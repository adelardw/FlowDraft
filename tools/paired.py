"""Парные контрасты двух систем по промптам, с 95% доверительным интервалом.

  python tools/paired.py <pp-A.jsonl> <pp-B.jsonl> [--metric acceptance] [--cap 30]
                         [--fa qwen06_orthrus/] [--fb qwen06_flowdraft_multistep_qkv/]

Файлы — построчные замеры `src/eval.py` (per_prompt_file). Строки сопоставляются
по (расписание, набор, индекс промпта); --cap оставляет первые N промптов
каждого набора, чтобы сравнить замеры разной длины на одних и тех же задачах.
--fa/--fb оставляют строки, у которых путь чекпоинта содержит подстроку: так
из одного файла с несколькими конфигурациями берётся нужная.
Единица наблюдения — промпт: интервал отвечает на вопрос «устойчиво ли это по
задачам», а не «повторится ли на другом зерне обучения».
"""
import argparse
import json
import math
from collections import defaultdict


def load(path, cap, needle=None):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if needle and needle not in r["checkpoint"]:
            continue
        if cap is not None and r["prompt_index"] >= cap:
            continue
        rows[(json.dumps(r["jumps"]), r["dataset"], r["prompt_index"])] = r
    return rows


def ci(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return mu, 1.96 * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default="acceptance")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--fa", default=None)
    ap.add_argument("--fb", default=None)
    args = ap.parse_args()
    A, B = load(args.a, args.cap, args.fa), load(args.b, args.cap, args.fb)
    by_sched = defaultdict(list)
    for key in sorted(set(A) & set(B)):
        by_sched[key[0]].append(key)
    print(f"метрика {args.metric}; B − A, 95% ДИ по промптам")
    for sched, keys in by_sched.items():
        a = [A[k][args.metric] for k in keys]
        b = [B[k][args.metric] for k in keys]
        d, h = ci([y - x for x, y in zip(a, b)])
        win = sum(y > x for x, y in zip(a, b)) / len(keys)
        lossless = all(A[k]["lossless"] and B[k]["lossless"] for k in keys)
        print(f"  jumps {sched}: n={len(keys)}  A {sum(a)/len(a):.3f}  B {sum(b)/len(b):.3f}  "
              f"Δ {d:+.3f} ± {h:.3f}  (B выше на {win:.0%} промптов)  lossless={lossless}")


if __name__ == "__main__":
    main()
