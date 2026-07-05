from omegaconf import DictConfig
from torch.utils.data import DataLoader


def build_dataloaders(cfg: DictConfig, tokenizer, df_processor):
    """nvidia/Nemotron-Post-Training-Dataset-v2 -> ``(train_loader, val_loader)``.

    Streaming — no full download: the requested category splits (``chat``,
    ``code``, ``math``, ``stem``, ``multilingual_*``) are interleaved and
    shuffled with a buffer; the first ``data.val_size`` samples become
    validation, the rest stream into training. ``messages`` are rendered
    with the tokenizer's chat template — the drafter must see the same
    formatting the verifier will be served at inference.

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

    def render(example) -> str:
        messages = [m for m in example["messages"] if m.get("content")]
        if use_template:
            return tokenizer.apply_chat_template(messages, tokenize=False)
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
