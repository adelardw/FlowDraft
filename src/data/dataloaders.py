from omegaconf import DictConfig
from torch.utils.data import DataLoader


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
    ds = ds.shuffle(seed=cfg.seed, buffer_size=d.get("shuffle_buffer", 1000))

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

    def make_loader(split_ds):
        return DataLoader(
            split_ds,
            batch_size=d.batch_size,
            collate_fn=collate,
            num_workers=d.get("num_workers", 0),
        )

    val_size = d.get("val_size", 256)
    return make_loader(ds.skip(val_size)), make_loader(ds.take(val_size))
