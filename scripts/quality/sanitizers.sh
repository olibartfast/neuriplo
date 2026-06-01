#!/usr/bin/env bash
# Configure, build, and test with AddressSanitizer + UndefinedBehaviorSanitizer.
# Mirrors the CI sanitizers job for backends that build without vendor SDKs locally.
#
# Usage:
#   ./scripts/quality/sanitizers.sh [OPTIONS]
#
# Options:
#   --backend NAME   DEFAULT_BACKEND (default: OPENCV_DNN)
#   --build-dir DIR  CMake build directory (default: build-asan)
#   --no-test        Build only; skip ctest
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT="$(quality_repo_root)"
BACKEND="${DEFAULT_BACKEND:-OPENCV_DNN}"
BUILD_DIR="build-asan"
RUN_TESTS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --no-test)
            RUN_TESTS=0
            shift
            ;;
        -h | --help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:abort_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"

echo "[sanitizers] Backend=${BACKEND} build_dir=${BUILD_DIR}"
echo "[sanitizers] ASAN_OPTIONS=${ASAN_OPTIONS}"
echo "[sanitizers] UBSAN_OPTIONS=${UBSAN_OPTIONS}"

cd "$ROOT"
cmake -S . -B "$BUILD_DIR" \
    -DDEFAULT_BACKEND="$BACKEND" \
    -DSANITIZERS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_INFERENCE_ENGINE_TESTS=ON

cmake --build "$BUILD_DIR" --parallel

if [[ "$RUN_TESTS" -eq 1 ]]; then
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo "[sanitizers] Passed."
