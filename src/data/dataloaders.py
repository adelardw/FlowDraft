from omegaconf import DictConfig


def build_dataloaders(cfg: DictConfig, tokenizer, df_processor):
    """YOUR data goes here — the only missing piece of the pipeline.

    Return ``(train_loader, val_loader)``; ``val_loader`` may be ``None``.

    Batch contract (what ``FlowMapOrthrus`` consumes):
      * ``input_ids [B, T]`` — long
      * ``attention_mask [B, T]`` — long, 1 = live token, 0 = padding
      * NOTHING ELSE is required. Do NOT put ``simplex`` in the batch — the
        one-hot ``[B, T, 128k]`` endpoints are built on-device inside the
        module; shipping them through the DataLoader wastes gigabytes.

    A minimal recipe::

        def collate(texts):
            enc = df_processor(texts, return_simplex=False,
                               truncation=True, max_length=cfg.data.max_length)
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]}

        train = DataLoader(MyTextDataset(...), batch_size=cfg.data.batch_size,
                           shuffle=True, num_workers=cfg.data.num_workers,
                           collate_fn=collate)

    Design note (README, open choices): corpus text trains the drafter on
    the corpus distribution; AR-generated continuations (on-policy) match
    the inference-time distribution — worth building both.
    """
    raise NotImplementedError("plug your Dataset/collate_fn/DataLoader in here")
