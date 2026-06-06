#!/usr/bin/env bash
# Run clang-tidy using compile_commands.json from a configured build tree.
#
# Usage:
#   ./scripts/quality/clang_tidy.sh [BUILD_DIR]
#
# Example:
#   cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
#   ./scripts/quality/clang_tidy.sh build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT="$(quality_repo_root)"
BUILD_DIR="${1:-build}"

TIDY="$(quality_clang_tidy)" || {
    echo "clang-tidy not found. Install clang-tidy-18 (Ubuntu: apt install clang-tidy-18)." >&2
    exit 1
}

DB="${ROOT}/${BUILD_DIR}/compile_commands.json"
if [[ ! -f "$DB" ]]; then
    echo "Missing ${DB}. Configure the build first, e.g.:" >&2
    echo "  cmake -S . -B ${BUILD_DIR} -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" >&2
    exit 1
fi

mapfile -t FILES < <(python3 - "$DB" <<'PY'
import json, sys
db = json.load(open(sys.argv[1]))
seen = set()
for entry in db:
    path = entry.get("file", "")
    if not path or path in seen:
        continue
    if "/backends/" in path or path.startswith("src/") or "/src/" in path or "/include/" in path:
        seen.add(path)
for path in sorted(seen):
    print(path)
PY
)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No translation units found in compile_commands.json." >&2
    exit 1
fi

echo "[${TIDY}] Checking ${#FILES[@]} file(s) (build dir: ${BUILD_DIR}) ..."
"$TIDY" -p "${ROOT}/${BUILD_DIR}" "${FILES[@]}"
echo "[${TIDY}] Passed."
