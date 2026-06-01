#!/usr/bin/env bash
# Format C++ sources with clang-format (matches CI format-check job).
# Usage:
#   ./scripts/quality/format.sh          # check (--dry-run --Werror)
#   ./scripts/quality/format.sh --fix    # rewrite files in place
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT="$(quality_repo_root)"
MODE=check

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) MODE=check; shift ;;
        --fix) MODE=fix; shift ;;
        -h | --help)
            echo "Usage: $0 [--check|--fix]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

CFMT="$(quality_clang_format)" || {
    echo "clang-format not found. Install clang-format-18 (Ubuntu: apt install clang-format-18)." >&2
    exit 1
}

mapfile -t FILES < <(quality_cpp_sources "$ROOT")
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No C++ sources found under src/, include/, backends/." >&2
    exit 1
fi

if [[ "$MODE" == fix ]]; then
    echo "[$CFMT] Formatting ${#FILES[@]} file(s)..."
    xargs -a <(printf '%s\n' "${FILES[@]}") "$CFMT" -i
    echo "Format complete."
else
    echo "[$CFMT] Checking ${#FILES[@]} file(s) (--dry-run --Werror)..."
    xargs -a <(printf '%s\n' "${FILES[@]}") "$CFMT" --dry-run --Werror
    echo "Format check passed."
fi
