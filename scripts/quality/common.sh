#!/usr/bin/env bash
# Shared helpers for scripts/quality/*.sh
set -euo pipefail

quality_repo_root() {
    local root
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo "$root"
}

quality_clang_format() {
    if command -v clang-format-18 >/dev/null 2>&1; then
        echo clang-format-18
    elif command -v clang-format >/dev/null 2>&1; then
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
    find "$root/src" "$root/include" "$root/backends" \
        \( -name '*.cpp' -o -name '*.hpp' \) -print 2>/dev/null | sort
}
