#!/usr/bin/env bash
#
# One command for the whole Orthrus validation 2x2.
#
#   ./scripts/run_validation.sh                 # single-GPU profile (~2-4 h on a 12 GB card)
#   PROFILE=full ./scripts/run_validation.sh    # paper protocol: whole splits, 2048 new tokens
#   DRY_RUN=1 ./scripts/run_validation.sh       # print the run plan and exit
#
# The grid swaps weights and code independently, which is the only way to tell a
# training gap from an implementation bug:
#
#   cell  weights   code        entry point
#   A     authors   authors     src/eval_reference.py on the release as-is
#   B     authors   ours        src/eval.py on the converted .ckpt
#   C     ours      ours        src/eval.py on our trained baseline
#   D     ours      authors     src/eval_reference.py on the reverse-converted repo
#
# Every stage is resume-safe: a (cell, benchmark, block size) whose row is
# already in $RESULTS is skipped, so the script can be interrupted and re-run,
# and re-running a finished grid is a no-op. Nothing here is machine-specific —
# override DTYPE/ATTN/PROFILE to move it to another box.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROFILE=${PROFILE:-quick}
# gsm8k and math500 are deliberately out of the default suite; add them back
# here to restore the full six-benchmark Orthrus protocol.
BENCHMARKS=${BENCHMARKS:-"humaneval mbpp aime24 aime25"}
# 8 is our checkpoint's trained block, 32 is the release's. Running both at both
# separates "trained for this K" from "better drafter".
BLOCK_SIZES=${BLOCK_SIZES:-"8 32"}
DTYPE=${DTYPE:-bfloat16}
ATTN=${ATTN:-sdpa}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
case "$PROFILE" in
    quick) PRESET=validation ;;
    full)  PRESET=validation_full ;;
    *) echo "PROFILE must be 'quick' or 'full', got '$PROFILE'" >&2; exit 2 ;;
esac

# Per profile, so a quick sweep and a paper-scale sweep never share a file: they
# differ only in n_prompts/max_new_tokens and would otherwise land in one table
# as if they were comparable rows. The audit is profile-independent (it pins its
# own prompt count and length), so it stays shared.
RESULTS=${RESULTS:-results/validation-$PROFILE.jsonl}
PER_PROMPT=${PER_PROMPT:-results/validation-$PROFILE-prompts.jsonl}
AUDIT=${AUDIT:-results/validation-lossless.jsonl}

REFERENCE=${REFERENCE:-weights/Orthrus-Qwen3-1.7B}
OURS_CKPT=${OURS_CKPT:-weights/qwen3-1.7-baseline-block8.ckpt}
OURS_ZIP=${OURS_ZIP:-weights/qwen3-1.7-baseline-block8.ckpt.zip}
AUTHORS_CKPT=${AUTHORS_CKPT:-weights/orthrus-authors.ckpt}
OURS_AS_REFERENCE=${OURS_AS_REFERENCE:-weights/ours-as-orthrus}

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- stage 0
log "stage 0/4  preflight"

if [[ ! -d "$REFERENCE" ]]; then
    log "downloading the Orthrus release -> $REFERENCE"
    $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('chiennv/Orthrus-Qwen3-1.7B', local_dir='$REFERENCE')
"
fi
if [[ ! -f "$OURS_CKPT" && -f "$OURS_ZIP" ]]; then
    log "extracting $OURS_ZIP"
    unzip -o -q "$OURS_ZIP" -x '__MACOSX/*' -d "$(dirname "$OURS_CKPT")"
fi
if [[ ! -f "$AUTHORS_CKPT" ]]; then
    log "converting the release into our checkpoint layout"
    $PYTHON -m src.tools.convert_orthrus to-ours --reference "$REFERENCE" --out "$AUTHORS_CKPT"
fi
if [[ ! -d "$OURS_AS_REFERENCE" ]]; then
    log "converting our checkpoint into the release layout"
    $PYTHON -m src.tools.convert_orthrus to-theirs \
        --checkpoint "$OURS_CKPT" --reference "$REFERENCE" --out "$OURS_AS_REFERENCE"
fi
$PYTHON -m src.tools.convert_orthrus verify --reference "$REFERENCE"

# ---------------------------------------------------------------- stage 1
log "stage 1/4  equivalence gate"
# Fatal on purpose: if our diffusion forward disagrees with the reference on
# identical weights and an identical block, the grid below would be measuring a
# bug rather than a drafter, and every downstream number would be meaningless.
$PYTHON -m src.tools.equivalence_gate --reference "$REFERENCE" --checkpoint "$AUTHORS_CKPT"

# ---------------------------------------------------------------- stage 2
log "stage 2/4  benchmark grid (profile=$PROFILE, $DTYPE/$ATTN)"

have_row() {  # file run_id codebase weights_source dataset block_size
    # run_id carries the profile, so a `quick` grid is never mistaken for a
    # finished `full` one — the two differ only in n_prompts/max_new_tokens,
    # which are otherwise invisible in the resume key.
    $PYTHON - "$@" <<'PY'
import json, sys
path, run_id, codebase, weights, dataset, block = sys.argv[1:7]
try:
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A kill during append can leave a truncated last line. That run
                # is simply not done, so ignore the fragment and let it re-run.
                continue
            if (row.get("run_id") == run_id
                    and row.get("codebase") == codebase
                    and row.get("weights_source") == weights
                    and row.get("dataset") == dataset
                    and str(row.get("block_size")) == block):
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PY
}

FAILED=()

run_cell() {  # cell codebase weights_source dataset block_size
    local cell=$1 codebase=$2 weights=$3 dataset=$4 block=$5
    if have_row "$RESULTS" "$PROFILE" "$codebase" "$weights" "$dataset" "$block"; then
        echo "  [skip] cell $cell  $dataset  K=$block  (already in $RESULTS)"
        return 0
    fi
    echo "  [run ] cell $cell  $dataset  K=$block"
    if [[ "$DRY_RUN" == "1" ]]; then return 0; fi

    local common=(
        "+benchmark=$PRESET"
        "data=$dataset"
        "decode.block_size=$block"
        "model.backbone.dtype=$DTYPE"
        "model.backbone.attn_implementation=$ATTN"
        # The grid is a throughput audit: bf16 plus fused kernels break bitwise
        # equality for numerical reasons, so losslessness is certified in stage 3
        # under fp32/eager instead of aborting the grid here.
        "lossless_policy=diagnose"
        "weights_source=$weights"
        "run_id=$PROFILE"
        "split_label=cell-$cell"
        "results_file=$RESULTS"
        "per_prompt_file=$PER_PROMPT"
    )
    # One bad leg (a dataset that fails to download, a transient OOM) must not
    # discard the hours already spent: record it, keep going, report at the end.
    local status=0
    if [[ "$codebase" == "reference" ]]; then
        local path=$REFERENCE
        [[ "$weights" == "ours" ]] && path=$OURS_AS_REFERENCE
        $PYTHON src/eval_reference.py "${common[@]}" "reference_path=$path" || status=$?
    else
        local ckpt=$AUTHORS_CKPT
        [[ "$weights" == "ours" ]] && ckpt=$OURS_CKPT
        $PYTHON src/eval.py "${common[@]}" "checkpoint=$ckpt" "variant=orthrus" || status=$?
    fi
    if [[ $status -ne 0 ]]; then
        echo "  [FAIL] cell $cell  $dataset  K=$block  (exit $status)" >&2
        FAILED+=("cell $cell $dataset K=$block")
    fi
    return 0
}

for dataset in $BENCHMARKS; do
    for block in $BLOCK_SIZES; do
        run_cell A reference authors "$dataset" "$block"
        run_cell B ours      authors "$dataset" "$block"
        run_cell C ours      ours    "$dataset" "$block"
        run_cell D reference ours    "$dataset" "$block"
    done
done

# ---------------------------------------------------------------- stage 3
log "stage 3/4  losslessness audit (float32/eager)"
# Short and narrow on purpose: fp32 weights plus eager attention is the only
# configuration where bitwise equality with AR is a meaningful claim, and it is
# ~3x slower and close to the VRAM ceiling on a 12 GB card.
AUDIT_DATASET=${AUDIT_DATASET:-$(echo "$BENCHMARKS" | awk '{print $1}')}
audit_cell() {  # cell codebase weights_source
    local cell=$1 codebase=$2 weights=$3
    if have_row "$AUDIT" "audit" "$codebase" "$weights" "$AUDIT_DATASET" 8; then
        echo "  [skip] audit cell $cell  (already in $AUDIT)"
        return 0
    fi
    echo "  [run ] audit cell $cell"
    if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
    local common=(
        "+benchmark=$PRESET"
        "data=$AUDIT_DATASET"
        "decode.block_size=8"
        "decode.n_prompts=3"
        "decode.max_new_tokens=64"
        "model.backbone.dtype=float32"
        "model.backbone.attn_implementation=eager"
        "lossless_policy=diagnose"
        "weights_source=$weights"
        "run_id=audit"
        "split_label=audit-$cell"
        "results_file=$AUDIT"
        "per_prompt_file=null"
    )
    local status=0
    if [[ "$codebase" == "reference" ]]; then
        local path=$REFERENCE
        [[ "$weights" == "ours" ]] && path=$OURS_AS_REFERENCE
        $PYTHON src/eval_reference.py "${common[@]}" "reference_path=$path" || status=$?
    else
        local ckpt=$AUTHORS_CKPT
        [[ "$weights" == "ours" ]] && ckpt=$OURS_CKPT
        $PYTHON src/eval.py "${common[@]}" "checkpoint=$ckpt" "variant=orthrus" || status=$?
    fi
    if [[ $status -ne 0 ]]; then
        echo "  [FAIL] audit cell $cell (exit $status)" >&2
        FAILED+=("audit cell $cell")
    fi
    return 0
}
audit_cell A reference authors
audit_cell B ours      authors
audit_cell C ours      ours
audit_cell D reference ours

# ---------------------------------------------------------------- stage 4
log "stage 4/4  table"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry run — no table written)"
    exit 0
fi
$PYTHON src/tables.py --results "$RESULTS" --audit "$AUDIT" \
    --out results/validation.md --readme README.md
echo
echo "wrote results/validation.md"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '\n\033[1m%d run(s) failed — re-run this script to retry just those:\033[0m\n' "${#FAILED[@]}"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
