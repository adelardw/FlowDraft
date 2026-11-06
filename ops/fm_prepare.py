"""Prepare the authorized fixed 80k raw-example pool, without generating banks."""
import argparse
import hashlib
import json
import os
import itertools
from pathlib import Path

os.environ.setdefault('HF_DATASETS_CACHE', '/private/tmp/flowdraft-fm-hf-datasets')

import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    def rows(path):
        for chunk in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from chunk.to_pylist()
    source_rows = itertools.chain.from_iterable(zip(*(rows(source / f'{k}.parquet') for k in ['chat', 'code', 'math'])))
    selected = list(itertools.islice(source_rows, 80032))
    if len(selected) != 80032:
        raise ValueError('not enough source examples')
    result = {}
    for name, examples in [('train', selected[32:]), ('validation', selected[:32])]:
        collected, count = {}, 0
        for start in range(0, len(examples), 128):
            texts = [tokenizer.apply_chat_template([m for m in e['messages'] if m.get('content')],
                     tokenize=False, add_generation_prompt=False) for e in examples[start:start+128]]
            batch = tokenizer(texts, return_tensors='pt', add_special_tokens=False,
                              truncation=True, max_length=256, padding='max_length')
            for k, v in batch.items():
                collected.setdefault(k, []).append(v.to(torch.int32 if k == 'input_ids' else torch.uint8))
            count += batch['input_ids'].size(0)
            if count % 8192 == 0:
                print(name, count, flush=True)
        result[name] = {k: torch.cat(v) for k, v in collected.items()}
        if int(result[name]['attention_mask'].sum(-1).min()) < 6:
            raise ValueError('source includes rows too short for four distinct anchors')
        print(name, count, flush=True)
    assert result['train']['input_ids'].size(0) == 80000
    torch.save(result, output / 'tokens.pt')
    manifest = {
        'source': 'original Nemotron chat/code/math, interleaved',
        'train_examples': 80000, 'validation_examples': 32, 'max_length': 256,
        'packed': False, 'seed': 42, 'tokenizer': 'Qwen/Qwen3-0.6B',
        'tokens_sha256': hashlib.sha256((output / 'tokens.pt').read_bytes()).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.parquet')},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
