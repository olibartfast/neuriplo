#!/usr/bin/env bash
# Run local code-quality checks (fast subset by default).
#
# Usage:
#   ./scripts/quality/run.sh              # format + cppcheck
#   ./scripts/quality/run.sh --all        # format + cppcheck + clang-tidy + sanitizers
#   ./scripts/quality/run.sh --format     # format check only
#   ./scripts/quality/run.sh --fix        # clang-format -i
#   ./scripts/quality/run.sh --sanitizers # ASan+UBSan build+test (OPENCV_DNN default)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_FORMAT=0
RUN_CPPCHECK=0
RUN_TIDY=0
RUN_SAN=0
FORMAT_MODE=check
BUILD_DIR=build

if [[ $# -eq 0 ]]; then
    RUN_FORMAT=1
    RUN_CPPCHECK=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            RUN_FORMAT=1
            RUN_CPPCHECK=1
            RUN_TIDY=1
            RUN_SAN=1
            shift
            ;;
        --format) RUN_FORMAT=1; shift ;;
        --fix)
            RUN_FORMAT=1
            FORMAT_MODE=fix
            shift
            ;;
        --cppcheck) RUN_CPPCHECK=1; shift ;;
        --clang-tidy)
            RUN_TIDY=1
            shift
            ;;
        --sanitizers) RUN_SAN=1; shift ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
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

if [[ "$RUN_FORMAT" -eq 1 ]]; then
    if [[ "$FORMAT_MODE" == fix ]]; then
        "${SCRIPT_DIR}/format.sh" --fix
    else
        "${SCRIPT_DIR}/format.sh" --check
    fi
fi

if [[ "$RUN_CPPCHECK" -eq 1 ]]; then
    "${SCRIPT_DIR}/cppcheck.sh"
fi

if [[ "$RUN_TIDY" -eq 1 ]]; then
    "${SCRIPT_DIR}/clang_tidy.sh" "$BUILD_DIR"
fi

if [[ "$RUN_SAN" -eq 1 ]]; then
    "${SCRIPT_DIR}/sanitizers.sh" --build-dir "${BUILD_DIR}-asan"
fi

echo "[quality] All requested checks passed."
