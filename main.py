"""FlowDraft playground: generate from YOUR prompts with lossless drafting.

    uv run python main.py -p "Once upon a time" -p "def main():"
    uv run python main.py -p "..." --variant orthrus --checkpoint path.ckpt

Dataset evaluation with metrics lives in src/eval.py; this CLI is for
eyeballing generations.
"""
import hydra
import typer
from loguru import logger

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def generate(
    prompt: list[str] = typer.Option(..., "--prompt", "-p", help="Repeat -p for several prompts."),
    block_size: int = typer.Option(8, help="Drafted block size K."),
    jumps: int = typer.Option(1, help="Flow-map jumps per block."),
    max_new_tokens: int = typer.Option(64),
    model_cfg: str = typer.Option(
        None, "--model",
        help="Model config name (llama3_3b | qwen2_0.5b) or an HF id with '/'.",
    ),
    variant: str = typer.Option(
        "flowdraft",
        help="flowdraft | flowdraft_block_wise | orthrus | orthrus_block_wise",
    ),
    checkpoint: str = typer.Option(None, help="Trained DF-head .ckpt; omit for the raw drafter."),
    temperature: float = typer.Option(0.0, help="0 = greedy; >0 = sampling (coupled: bitwise lossless)."),
    top_k: int = typer.Option(None, help="Sampling only."),
    top_p: float = typer.Option(None, help="Sampling only."),
    coupled: bool = typer.Option(True, help="T>0: Gumbel-coupled sampling (bit-exact vs AR)."),
    sampling_seed: int = typer.Option(0, help="Coupled sampling seed."),
    lossless_check: bool = typer.Option(True, help="Compare bitwise vs AR (greedy or coupled sampling)."),
):
    import torch

    from src.models.factory import build_lit

    overrides = [f"variant={variant}"]
    if checkpoint:
        overrides.append(f"checkpoint={checkpoint}")
    if model_cfg:
        # config-group name (qwen2_0.5b) or a raw HF id (org/name)
        overrides.append(f"model.name={model_cfg}" if "/" in model_cfg else f"model={model_cfg}")
    with hydra.initialize(version_base="1.3", config_path="src/configs"):
        cfg = hydra.compose(config_name="eval", overrides=overrides)
    torch.manual_seed(cfg.seed)
    model = build_lit(cfg)

    sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                    coupled=coupled, sampling_seed=sampling_seed)
    for text in prompt:
        out = model.generate(
            text, block_size=block_size, jumps=jumps, max_new_tokens=max_new_tokens, **sampling
        )
        mean_acc = sum(out["acceptance"]) / max(len(out["acceptance"]), 1)
        typer.echo(f"\n>>> {text}")
        typer.echo(out.get("text", out["new_tokens"]))
        typer.echo(
            f"[acceptance={mean_acc:.2f}, forwards={out['n_forwards']}, {out['seconds']:.2f}s]"
        )
        if lossless_check and (temperature == 0 or coupled):
            ar = model.ar_generate(text, max_new_tokens=max_new_tokens, **sampling)
            ok = out["new_tokens"] == ar["new_tokens"]
            typer.echo(f"[lossless vs AR (bitwise): {'PASS' if ok else 'FAIL'}]")
            if not ok:
                logger.error("flow-draft output diverged from AR!")
                raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
