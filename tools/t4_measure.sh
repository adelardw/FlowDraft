#!/bin/bash
# Замер декода по протоколу статьи на любом T4 вне Kaggle (например, Colab).
#
#   bash tools/t4_measure.sh <чекпоинт.ckpt> <папка вывода> [расписания]
#
# расписания — через запятую из n1,n2,n3,n4v (по умолчанию n1). Протокол тот
# же, что в ops/body_eval.py: шесть наборов, 100 промптов (AIME — все 30),
# блок 32, 64 новых токена, float32, плотное внимание. Ускорение по времени
# считается внутри одного прогона против обычного жадного декода той же
# модели, поэтому числа с другого T4 сравнимы с таблицей статьи.
set -euo pipefail
CKPT=$1; OUT=$2; SCHED=${3:-n1}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
declare -A JUMPS=([n1]="1" [n2]="[[0,1],[0.5,1]]" [n3]="[[0,1],[0.5,1],[0.75,1]]" [n4v]="[[0,1],[0.5,1],[0.7,1],[0.85,1]]")
DATA="gsm8k:100 math500:100 humaneval:100 mbpp:100 aime24:30 aime25:30"
for tag in ${SCHED//,/ }; do
  for item in $DATA; do
    name=${item%%:*}; n=${item##*:}
    echo "=== $tag $name"
    FLOWDRAFT_DF_ATTENTION=dense PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=. python -u src/eval.py checkpoint="$CKPT" data=$name decode.n_prompts=$n \
      decode.block_size=32 decode.max_new_tokens=64 "decode.jumps=${JUMPS[$tag]}" \
      model.backbone.dtype=float32 \
      results_file="$OUT/t4.jsonl" per_prompt_file="$OUT/pp-t4.jsonl" 2>&1 | grep -E "speedup|lossless|Error" | tail -3
  done
done
python - "$OUT/t4.jsonl" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows:
    by[json.dumps(r["jumps"])].append(r)
for j, rs in by.items():
    m = lambda k: sum(r[k] for r in rs) / len(rs)
    print(f"jumps {j}: наборов {len(rs)}, A {m('acceptance'):.3f}, tpf_steady {m('tpf_steady'):.3f}, "
          f"ускорение {m('speedup'):.3f}x, без потерь: {all(r['lossless'] for r in rs)}")
PY
