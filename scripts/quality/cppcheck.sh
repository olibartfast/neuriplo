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
# TestTemplateCompileTest.cpp is excluded: cppcheck cannot expand gmock's
# MOCK_METHOD. Without the gmock headers it reports a syntax error in that
# file; with them, one inside gmock's own preprocessor machinery. It is
# compiled and run by ctest instead.
cppcheck --enable=warning --std=c++17 --error-exitcode=1 \
    --suppress=missingIncludeSystem \
    --suppress=unmatchedSuppression \
    -i backends/src/test/TestTemplateCompileTest.cpp \
    -I include -I backends/src \
    src/ backends/
echo "[cppcheck] Passed."
