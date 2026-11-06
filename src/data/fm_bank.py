"""Fixed token pool and persistent prefix-specific AR continuation banks."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class EndpointBank:
    def __init__(self, path, specification):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS banks (key TEXT PRIMARY KEY, prefix BLOB, samples BLOB)")
        encoded = json.dumps(specification, sort_keys=True)
        saved = self.db.execute("SELECT value FROM metadata").fetchone()
        if saved is None:
            self.db.execute("INSERT INTO metadata VALUES (?)", (encoded,))
            self.db.commit()
        elif saved[0] != encoded:
            self.db.close()
            raise ValueError("AR bank specification differs; refusing to reuse incompatible endpoints")
        self.specification = specification

    @staticmethod
    def identity(prefix):
        raw = prefix.detach().cpu().numpy().astype('<i4').tobytes()
        return hashlib.sha256(raw).hexdigest(), raw

    def get(self, prefix):
        key, raw = self.identity(prefix)
        row = self.db.execute("SELECT prefix, samples FROM banks WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row[0] != raw:
            raise RuntimeError("AR bank prefix mismatch")
        shape = (self.specification['size'], self.specification['length'])
        return torch.from_numpy(np.frombuffer(row[1], dtype='<i4').copy().reshape(shape)).long()

    def put(self, prefix, samples):
        key, raw = self.identity(prefix)
        expected = (self.specification['size'], self.specification['length'])
        if tuple(samples.shape) != expected:
            raise ValueError("AR bank has the wrong shape")
        data = samples.detach().cpu().numpy().astype('<i4').tobytes()
        self.db.execute("INSERT INTO banks VALUES (?, ?, ?)", (key, raw, data))
        self.db.commit()

    def close(self):
        self.db.close()


class FixedTokenPool(IterableDataset):
    """An infinite deterministic sequence of permutations of the SAME finite pool.

    Resume offset counts consumed examples globally, including full epochs.
    No source rows are removed or substituted when resuming.
    """
    def __init__(self, rows, seed, skip=0):
        self.rows, self.seed, self.skip = rows, int(seed), int(skip)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("fixed FM pool currently requires num_workers=0")
        count = self.rows['input_ids'].size(0)
        epoch, offset = divmod(self.skip, count)
        while True:
            order = torch.randperm(count, generator=torch.Generator().manual_seed(self.seed + epoch))
            for index in order[offset:]:
                yield {key: value[index].long() for key, value in self.rows.items()}
            epoch, offset = epoch + 1, 0


def build_bank_loaders(cfg):
    if cfg.trainer.get('devices', 1) != 1 or cfg.data.get('num_workers', 0) != 0:
        raise ValueError("fixed-bank experiment currently supports one device and no loader workers")
    data = torch.load(cfg.data.fm_tokens_path, map_location='cpu', weights_only=True)
    train, val = data['train'], data['validation']
    if train['input_ids'].size(0) != int(cfg.data.train_size):
        raise ValueError("token pool size differs from data.train_size")
    pool = FixedTokenPool(train, cfg.seed, cfg.data.get('stream_skip', 0))
    validation = [{k: v[i].long() for k, v in val.items()} for i in range(val['input_ids'].size(0))]
    return DataLoader(pool, batch_size=cfg.data.batch_size), DataLoader(validation, batch_size=cfg.data.batch_size)
