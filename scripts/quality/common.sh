#!/usr/bin/env bash
# Shared helpers for scripts/quality/*.sh
set -euo pipefail

quality_repo_root() {
    local root
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo "$root"
}

quality_clang_format() {
    if command -v clang-format-21 >/dev/null 2>&1; then
        echo clang-format-21
    elif command -v clang-format >/dev/null 2>&1; then
        local version
        version="$(clang-format --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)"
        if [[ "${version%%.*}" != "21" ]]; then
            echo "clang-format 21 is required for CI parity; found ${version:-unknown}." >&2
            echo "Install clang-format-21 or run 'pip install clang-format==21.1.8'." >&2
            return 1
        fi
        echo clang-format
    else
        return 1
    fi
}

quality_clang_tidy() {
    if command -v clang-tidy-18 >/dev/null 2>&1; then
        echo clang-tidy-18
    elif command -v clang-tidy >/dev/null 2>&1; then
        echo clang-tidy
    else
        return 1
    fi
}

quality_cpp_sources() {
    local root="$1"
    find "$root/src" "$root/include" "$root/backends" "$root/test" \
        \( -name '*.cpp' -o -name '*.hpp' \) -print 2>/dev/null | sort
}
