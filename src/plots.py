"""Analysis plots over eval results (``results/eval.jsonl``).

Every ``src/eval.py`` run appends one JSON row; this CLI turns the
accumulated rows into the report figures:

  1. ``frontier.png``   — acceptance vs jumps (the acceptance-vs-passes
     frontier), one line per (variant, block_size);
  2. ``tpf.png``        — TPF ± std bars per configuration, with the AR
     reference (tpf_ar) as a horizontal line;
  3. ``block_size.png`` — TPF vs block size K, one line per variant.

    uv run python src/plots.py                       # all figures
    uv run python src/plots.py --results path.jsonl --out-dir results
"""
import json
from collections import defaultdict
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _label(row):
    tag = row["variant"]
    if row.get("dataset"):
        tag += f" @{row['dataset'].split('/')[-1]}"  # unseen bench vs train distribution
    if row.get("temperature", 0) > 0:
        tag += f" T={row['temperature']}" + ("" if row.get("coupled", True) else " (uncoupled)")
    return tag


@app.command()
def main(
    results: str = typer.Option("results/eval.jsonl", help="JSONL written by src/eval.py."),
    out_dir: str = typer.Option("results", help="Where the PNGs go."),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in Path(results).read_text().splitlines() if line.strip()]
    if not rows:
        raise typer.BadParameter(f"no rows in {results} — run src/eval.py first")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. acceptance-vs-passes frontier (integer jump counts only)
    frontier = defaultdict(list)
    for r in rows:
        if isinstance(r["jumps"], int):
            frontier[(_label(r), r["block_size"])].append((r["jumps"], r["acceptance"]))
    if frontier:
        fig, ax = plt.subplots(figsize=(7, 5))
        for (label, block), pts in sorted(frontier.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=f"{label}, K={block}")
        ax.set_xlabel("jumps (draft forwards per block)")
        ax.set_ylabel("acceptance (tokens per cycle)")
        ax.set_title("Acceptance-vs-passes frontier")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.savefig(out / "frontier.png", dpi=150, bbox_inches="tight")
        typer.echo(f"-> {out / 'frontier.png'}")

    # 2. TPF comparison with the AR reference
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, tpfs, errs = [], [], []
    for r in rows:
        labels.append(f"{_label(r)}\nK={r['block_size']} j={r['jumps']}")
        tpfs.append(r["tpf"])
        errs.append(r.get("tpf_std", 0.0))
    ax.bar(range(len(rows)), tpfs, yerr=errs, capsize=3)
    ar_ref = sum(r.get("tpf_ar", 1.0) for r in rows) / len(rows)
    ax.axhline(ar_ref, color="gray", linestyle="--", label=f"AR reference ({ar_ref:.2f})")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("TPF (tokens per forward)")
    ax.set_title("TPF per configuration (± std over prompts)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out / "tpf.png", dpi=150, bbox_inches="tight")
    typer.echo(f"-> {out / 'tpf.png'}")

    # 3. TPF vs block size
    by_block = defaultdict(list)
    for r in rows:
        if isinstance(r["jumps"], int):
            by_block[(_label(r), r["jumps"])].append((r["block_size"], r["tpf"]))
    if by_block:
        fig, ax = plt.subplots(figsize=(7, 5))
        for (label, jumps), pts in sorted(by_block.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s",
                    label=f"{label}, jumps={jumps}")
        ax.set_xlabel("block size K")
        ax.set_ylabel("TPF")
        ax.set_title("TPF vs block size")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.savefig(out / "block_size.png", dpi=150, bbox_inches="tight")
        typer.echo(f"-> {out / 'block_size.png'}")


if __name__ == "__main__":
    app()
