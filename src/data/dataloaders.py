import random

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset


class EpochShuffled(IterableDataset):
    """Buffer-shuffle a stream with a per-epoch seed.

    ``datasets`` forbids ``shuffle`` after ``skip``/``take`` (reordering the
    shards would change which examples get skipped), so epoch-wise reshuffle
    cannot be expressed with HF primitives once the val split is taken.
    This wrapper keeps the HF chain order-frozen — membership of the split
    never changes — and does the buffer shuffling itself, reseeded by
    ``set_epoch`` so every repetition of the stream yields a new order.
    """

    def __init__(self, ds, seed: int, buffer_size: int, size: int | None = None):
        self.ds, self.seed, self.buffer_size = ds, seed, buffer_size
        self.size = size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        # Known only when the pool is bounded (data.train_size). With a known
        # length the Trainer derives steps per epoch and the cosine-LR horizon
        # itself — data.train_size + trainer.max_epochs needs no manual
        # max_steps arithmetic.
        if self.size is None:
            raise TypeError("unbounded stream has no length")
        return self.size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        buffer = []
        for example in self.ds:
            if len(buffer) < self.buffer_size:
                buffer.append(example)
                continue
            i = rng.randrange(self.buffer_size)
            yield buffer[i]
            buffer[i] = example
        rng.shuffle(buffer)
        yield from buffer


class PackedTokenStream(IterableDataset):
    """Pack rendered examples into fixed-length token sequences.

    Orthrus trains on packed 2048-token instances.  Packing after stream
    shuffling preserves the requested training-mixture order while ensuring
    every emitted item is a real fixed-size sequence (no padding budget).
    """

    def __init__(self, ds, tokenize, max_length: int, size: int | None = None):
        self.ds, self.tokenize, self.max_length, self.size = ds, tokenize, max_length, size

    def set_epoch(self, epoch: int):
        if hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(epoch)

    def __len__(self):
        if self.size is None:
            raise TypeError("an unbounded packed stream has no length")
        return self.size

    def __iter__(self):
        pending, emitted = [], 0
        for example in self.ds:
            pending.extend(self.tokenize(example))
            while len(pending) >= self.max_length:
                ids, pending = pending[: self.max_length], pending[self.max_length :]
                yield {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                }
                emitted += 1
                if self.size is not None and emitted >= self.size:
                    return


def quiet_download_logs():
    """Keep ALL transfer chatter out of the training/eval console.

    Streaming pulls parquet shards over HTTP for the whole run; with hydra's
    root logger at INFO that means one line with a signed URL per range
    request (httpx), plus "Resolving data files" / "Fetching N files"
    progress bars from datasets and huggingface_hub. None of it is training
    signal. Errors (level >= ERROR) still come through.
    """
    import logging

    for name in ("httpx", "httpcore", "datasets", "huggingface_hub", "fsspec", "filelock", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)
    from datasets import disable_progress_bars
    from huggingface_hub.utils import disable_progress_bars as hub_disable_progress_bars

    disable_progress_bars()
    hub_disable_progress_bars()


def build_dataloaders(cfg: DictConfig, tokenizer, df_processor):
    """``cfg.data`` dataset -> ``(train_loader, val_loader)``.

    Streaming — no full download: the requested splits are interleaved and
    shuffled with a buffer; the first ``data.val_size`` samples become
    validation, the rest stream into training. Two row formats are supported:

    * ``messages`` (Nemotron-style chat) — rendered with the tokenizer's
      chat template: the drafter must see the same formatting the verifier
      will be served at inference;
    * plain text (evaluation benches: MATH-500 etc.) — the column named by
      ``data.text_field`` (fallback: prompt/problem/question/text) is wrapped
      as a single user turn with the generation prompt appended, i.e. exactly
      what an instruct verifier receives at inference.

    Repeating the stream (multi-epoch training): every new Trainer epoch
    re-opens the stream, and ``ReshuffleStreamingData`` (src/train.py) calls
    ``set_epoch`` so the sample ORDER differs between repetitions. The
    val/train split is taken BEFORE the shuffle on purpose: membership of the
    validation slice must not depend on the epoch, otherwise reshuffling
    would leak validation samples into training (see ``EpochShuffled``).

    Batch contract (what the models consume): ``input_ids [B, T]`` +
    ``attention_mask [B, T]``. The ``[B, T, V]`` simplex is built on-device
    inside the model — it never rides the DataLoader.
    """
    from datasets import interleave_datasets, load_dataset

    d = cfg.data
    streams = [
        load_dataset(d.dataset, split=split, streaming=d.get("streaming", True))
        for split in d.splits
    ]
    ds = streams[0] if len(streams) == 1 else interleave_datasets(streams)

    use_template = getattr(tokenizer, "chat_template", None) is not None
    text_field = d.get("text_field", None)

    def extract_messages(example):
        if "messages" in example:
            return [m for m in example["messages"] if m.get("content")]
        for field in ([text_field] if text_field else ["prompt", "problem", "question", "text"]):
            if example.get(field):
                return [{"role": "user", "content": example[field]}]
        raise KeyError(f"no text column found in row (keys: {list(example)}); set data.text_field")

    def render(example) -> str:
        messages = extract_messages(example)
        if use_template:
            return tokenizer.apply_chat_template(
                messages, tokenize=False,
                # bare-prompt rows: end with the assistant header so the
                # continuation starts where inference would
                add_generation_prompt="messages" not in example,
            )
        return "\n".join(m["content"] for m in messages)

    def collate(examples):
        enc = df_processor(
            [render(e) for e in examples],
            return_simplex=False,
            truncation=True,
            max_length=d.max_length,
            # the chat template already carries BOS & co — adding them again
            # would double <|begin_of_text|> and shift every position
            add_special_tokens=not use_template,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def make_loader(split_ds, *, packed=False):
        return DataLoader(
            split_ds,
            batch_size=d.batch_size,
            collate_fn=None if packed else collate,
            num_workers=d.get("num_workers", 0),
        )

    val_size = d.get("val_size", 256)
    train_ds = ds.skip(val_size)
    # data.train_size bounds the training pool to a FIXED set of samples, so
    # trainer.max_epochs repeats exactly that set (an epoch in the strict
    # sense) — still streaming, nothing is downloaded ahead. null/0 = the
    # whole stream: repetitions then draw fresh samples in a fresh order.
    train_size = d.get("train_size", None)
    pack_sequences = d.get("pack_sequences", False)
    if train_size and not pack_sequences:
        train_ds = train_ds.take(train_size)
    train_ds = EpochShuffled(train_ds, seed=cfg.seed, buffer_size=d.get("shuffle_buffer", 1000),
                             size=None if pack_sequences else train_size)

    if pack_sequences:
        def tokenize_for_pack(example):
            text = render(example)
            ids = tokenizer(
                text,
                add_special_tokens=not use_template,
                truncation=False,
            )["input_ids"]
            # Keep separate conversations from becoming one synthetic turn.
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            return ids

        train_ds = PackedTokenStream(
            train_ds, tokenize_for_pack, max_length=d.max_length, size=train_size
        )
    return make_loader(train_ds, packed=pack_sequences), make_loader(ds.take(val_size))
