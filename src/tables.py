"""Render the Orthrus validation 2x2 from results/validation.jsonl.

    uv run python src/tables.py --results results/validation.jsonl \
        --audit results/validation-lossless.jsonl --out results/validation.md

The layout encodes the one rule that makes the grid readable: ``acceptance``,
``tpf`` and ``lossless`` are properties of the method and compare across
codebases, while ``tokens_per_s`` and ``speedup`` also carry each harness's
Python overhead and compare only within one. So the cross-codebase sections
carry the former and the wall-clock columns stay grouped under their own
codebase, labelled as such.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

# (codebase, weights_source) -> (cell, weights label, code label)
CELLS = OrderedDict(
    [
        (("reference", "authors"), ("A", "authors", "authors")),
        (("ours", "authors"), ("B", "authors", "ours")),
        (("ours", "ours"), ("C", "ours", "ours")),
        (("reference", "ours"), ("D", "ours", "authors")),
    ]
)


def load(path: Path) -> dict:
    """Latest row per (codebase, weights, dataset, block size)."""
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # An interrupted append can truncate the last line; render what
            # completed rather than refusing to produce a table at all.
            continue
        key = (
            row.get("codebase"),
            row.get("weights_source"),
            row.get("dataset"),
            row.get("block_size"),
        )
        rows[key] = row
    return rows


def num(value, digits=2, dash="—"):
    return dash if value is None else f"{value:.{digits}f}"


def pm(row, key, digits=2):
    if row is None or row.get(key) is None:
        return "—"
    std = row.get(f"{key}_std")
    base = f"{row[key]:.{digits}f}"
    return base if std is None else f"{base} ± {std:.{digits}f}"


def flag(value):
    return {True: "yes", False: "**no**", None: "n/a"}[value]


def axes(rows):
    datasets = sorted({key[2] for key in rows})
    blocks = sorted({key[3] for key in rows if key[3] is not None})
    return datasets, blocks


def implementation_section(rows, datasets, blocks):
    """Cell A vs cell B: identical weights, different decode loop."""
    lines = [
        "## 1. Is our implementation correct?",
        "",
        "The authors' diffusion head run through **their** decode loop (cell A) and",
        "through **ours** (cell B). Same weights, same prompts, same block size — so a",
        "gap here is our bug and nothing else.",
        "",
        "| benchmark | K | acceptance A | acceptance B | Δ | TPF A | TPF B | Δ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset in datasets:
        for block in blocks:
            a = rows.get(("reference", "authors", dataset, block))
            b = rows.get(("ours", "authors", dataset, block))
            if a is None and b is None:
                continue
            delta_acc = (
                num(b["acceptance"] - a["acceptance"], 2, "—")
                if a and b and a.get("acceptance") is not None and b.get("acceptance") is not None
                else "—"
            )
            delta_tpf = (
                num(b["tpf"] - a["tpf"], 3, "—")
                if a and b and a.get("tpf") is not None and b.get("tpf") is not None
                else "—"
            )
            lines.append(
                f"| {dataset} | {block} | {pm(a, 'acceptance')} | {pm(b, 'acceptance')} | "
                f"{delta_acc} | {pm(a, 'tpf', 3)} | {pm(b, 'tpf', 3)} | {delta_tpf} |"
            )
    lines.append("")
    return lines


def training_section(rows, datasets, blocks):
    """Authors' head vs ours, held inside each codebase."""
    lines = [
        "## 2. How does our trained head compare?",
        "",
        "Authors' weights vs ours, each measured inside a single codebase so the",
        "comparison is not contaminated by harness overhead.",
        "",
        "| benchmark | K | codebase | acceptance (authors) | acceptance (ours) | TPF (authors) | TPF (ours) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset in datasets:
        for block in blocks:
            for codebase, label in (("reference", "authors"), ("ours", "ours")):
                authors = rows.get((codebase, "authors", dataset, block))
                ours = rows.get((codebase, "ours", dataset, block))
                if authors is None and ours is None:
                    continue
                lines.append(
                    f"| {dataset} | {block} | {label} | {pm(authors, 'acceptance')} | "
                    f"{pm(ours, 'acceptance')} | {pm(authors, 'tpf', 3)} | {pm(ours, 'tpf', 3)} |"
                )
    lines.append("")
    return lines


def detail_section(rows, datasets, blocks):
    lines = [
        "## 3. Full grid",
        "",
        "`tok/s` and `speedup` are **within-codebase** numbers (each row's speedup is",
        "against that same harness's own autoregressive baseline).",
        "",
    ]
    for dataset in datasets:
        for block in blocks:
            present = [
                (key, rows[(key[0], key[1], dataset, block)])
                for key in CELLS
                if (key[0], key[1], dataset, block) in rows
            ]
            if not present:
                continue
            sample = present[0][1]
            lines += [
                f"### {dataset} · K={block}"
                f" · {sample.get('n_prompts')} prompts"
                f" · {sample.get('max_new_tokens')} new tokens"
                f" · {sample.get('attention_backend')}",
                "",
                "| cell | weights | code | acceptance ↑ | TPF ↑ | tok/s | tok/s AR | speedup |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for key, row in present:
                cell, weights, code = CELLS[key]
                lines.append(
                    f"| {cell} | {weights} | {code} | {pm(row, 'acceptance')} | "
                    f"{pm(row, 'tpf', 3)} | {num(row.get('tokens_per_s'), 1)} | "
                    f"{num(row.get('tokens_per_s_ar'), 1)} | "
                    f"{num(row.get('speedup'), 2)}× |"
                )
            lines.append("")
    return lines


def audit_section(audit):
    lines = [
        "## 4. Losslessness audit (float32 / eager)",
        "",
        "The throughput grid runs bf16 with fused kernels, where bitwise equality",
        "with autoregressive decoding fails for numerical reasons alone. This audit",
        "re-runs each cell in the one configuration where the claim is meaningful.",
        "",
        "| cell | weights | code | benchmark | K | prompts | bitwise lossless |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not audit:
        lines.append("| — | — | — | — | — | — | not run |")
    for key, row in audit.items():
        entry = CELLS.get((key[0], key[1]))
        if entry is None:
            continue
        cell, weights, code = entry
        lines.append(
            f"| {cell} | {weights} | {code} | {row.get('dataset')} | "
            f"{row.get('block_size')} | {row.get('n_prompts')} | "
            f"{flag(row.get('lossless'))} |"
        )
    lines.append("")
    return lines


def render(rows, audit, source=None) -> str:
    datasets, blocks = axes(rows)
    profiles = sorted({r.get("run_id") for r in rows.values() if r.get("run_id")})
    sample = next(iter(rows.values()), {})
    header = [
        "# Orthrus validation: authors' weights and code vs ours",
        "",
        f"Generated by `src/tables.py` from `{source or 'results/validation-quick.jsonl'}`"
        + (f" (profile: {', '.join(profiles)}" if profiles else "")
        + (f", {sample.get('n_prompts')} prompts x {sample.get('max_new_tokens')} new tokens)."
           if profiles else "."),
        "Reproduce the whole thing with `./scripts/run_validation.sh`.",
        "",
        "| cell | diffusion weights | decode loop | entry point |",
        "| --- | --- | --- | --- |",
        "| A | authors | authors | `src/eval_reference.py` on the release |",
        "| B | authors | ours | `src/eval.py` on the converted checkpoint |",
        "| C | ours | ours | `src/eval.py` on our trained baseline |",
        "| D | ours | authors | `src/eval_reference.py` on the reverse-converted repo |",
        "",
        "**acceptance** — drafted tokens accepted per cycle (the block's clean anchor",
        "is not counted, so K=32 admits at most 31). **TPF** — generated tokens per",
        "forward pass; one cycle costs two forwards, and autoregressive decoding is",
        "1.0 by construction. Both are hardware-independent and compare across",
        "codebases; wall-clock columns do not.",
        "",
    ]
    if not rows:
        return "\n".join(header + ["_No results yet — run `./scripts/run_validation.sh`._", ""])
    body = (
        implementation_section(rows, datasets, blocks)
        + training_section(rows, datasets, blocks)
        + detail_section(rows, datasets, blocks)
        + audit_section(audit)
    )
    return "\n".join(header + body)


def update_readme(readme: Path, rows) -> bool:
    """Replace the README's `## Results` section with a pointer plus headline."""
    if not readme.is_file() or not rows:
        return False
    text = readme.read_text()
    start = text.find("\n## Results\n")
    if start < 0:
        return False
    nxt = text.find("\n## ", start + 1)
    end = len(text) if nxt < 0 else nxt

    datasets, blocks = axes(rows)
    block = blocks[-1] if blocks else None
    lines = [
        "",
        "## Results",
        "",
        "Full 2x2 validation of our Orthrus re-implementation against the authors'",
        "released weights and reference code: **[results/validation.md](results/validation.md)**,",
        "reproducible with `./scripts/run_validation.sh`.",
        "",
        f"Headline, K={block} (acceptance = drafted tokens accepted per cycle; TPF = tokens per forward):",
        "",
        "| benchmark | A: authors weights, authors code | B: authors weights, our code | C: our weights, our code |",
        "| --- | --- | --- | --- |",
    ]
    for dataset in datasets:
        cells = [
            rows.get(("reference", "authors", dataset, block)),
            rows.get(("ours", "authors", dataset, block)),
            rows.get(("ours", "ours", dataset, block)),
        ]
        rendered = " | ".join(
            "—" if c is None else f"{pm(c, 'acceptance')} / {num(c.get('tpf'), 3)}"
            for c in cells
        )
        lines.append(f"| {dataset} | {rendered} |")
    lines.append("")
    readme.write_text(text[:start] + "\n".join(lines) + text[end:])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/validation-quick.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("results/validation-lossless.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/validation.md"))
    parser.add_argument("--readme", type=Path, help="also refresh this README's Results section")
    args = parser.parse_args()

    rows = load(args.results)
    audit = load(args.audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows, audit, source=str(args.results)))
    print(f"{len(rows)} rows -> {args.out}")
    if args.readme and update_readme(args.readme, rows):
        print(f"Results section refreshed -> {args.readme}")


if __name__ == "__main__":
    main()
