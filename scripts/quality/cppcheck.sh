#!/usr/bin/env bash
# Static analysis with cppcheck (matches CI cppcheck job).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT="$(quality_repo_root)"

if ! command -v cppcheck >/dev/null 2>&1; then
    echo "cppcheck not found. Install with: sudo apt install cppcheck" >&2
    exit 1
fi

echo "[cppcheck] Analyzing src/ and backends/ ..."
cd "$ROOT"
cppcheck --enable=warning --std=c++17 --error-exitcode=1 \
    --suppress=missingIncludeSystem \
    --suppress=unmatchedSuppression \
    -I include -I backends/src \
    src/ backends/
echo "[cppcheck] Passed."
