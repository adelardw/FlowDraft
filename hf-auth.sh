#!/usr/bin/env bash
#
# Load HF credentials from .env and run a command with them available.
# huggingface_hub / transformers auto-read $HF_TOKEN from the environment,
# so exporting it is all the "login" that is needed — no token written to disk.
#
# Usage:
#   ./hf-auth.sh uv run src/train.py
#   ./hf-auth.sh uv run src/models/model.py
#   ./hf-auth.sh            # just load + verify auth, run nothing
#
set -euo pipefail

# .env lives next to this script (repo root).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

[[ -f "$ENV_FILE" ]] || { echo "✗ $ENV_FILE not found" >&2; exit 1; }

# Export every KEY=VALUE from .env into the environment (comments/blank lines ok).
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

[[ -n "${HF_TOKEN:-}" ]] || { echo "✗ HF_TOKEN not set in $ENV_FILE" >&2; exit 1; }

# Verify the token actually authenticates (network round-trip).
uv run python -c "from huggingface_hub import whoami; print('✓ HF auth as', whoami()['name'])"

# Run whatever was passed (if anything), inheriting the exported env.
if [[ $# -gt 0 ]]; then
  exec "$@"
fi
