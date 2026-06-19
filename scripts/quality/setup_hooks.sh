#!/usr/bin/env bash
# One-time developer setup: venv + pre-commit hook environments + .githooks wrappers.
#
# Git uses core.hooksPath=.githooks (shell scripts that call `pre-commit run`).
# pre-commit install is not used because it refuses to run when hooksPath is set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required." >&2
    exit 1
fi

VENV="${ROOT}/.venv"
if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

if [[ -f requirements-dev.txt ]]; then
    pip install -q -r requirements-dev.txt
else
    pip install -q 'pre-commit>=3.5.0'
fi

pre-commit install-hooks

chmod +x .githooks/pre-commit .githooks/pre-push
chmod +x scripts/quality/*.sh
git config core.hooksPath .githooks

echo ""
echo "Git hooks: core.hooksPath=.githooks (calls pre-commit via .venv)"
echo "Activate tools:  source .venv/bin/activate"
echo "Manual checks:   ./scripts/quality/run.sh"
echo "All hooks:       pre-commit run --all-files"
echo "Push-stage:      pre-commit run --hook-stage pre-push"
