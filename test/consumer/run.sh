#!/usr/bin/env bash
# Consumer proof ([T-14], [V-12]): installs <build-dir> to a temporary prefix,
# builds test/consumer against it with only CMAKE_PREFIX_PATH set, and runs
# the C and C++ programs; then builds the C program again through pkg-config,
# and finally drives the library from Python ctypes. Every program must print exactly "OK FIXTURE_GOOD 2 4 6 8".
#
#   ./test/consumer/run.sh <build-dir>
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <build-dir>" >&2
    exit 2
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$1" && pwd)"
PLUGIN_DIR="${BUILD_DIR}/plugin_fixtures"
EXPECTED="OK FIXTURE_GOOD 2 4 6 8"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
PREFIX="${WORK}/prefix"

expect_ok() {
    local name="$1"
    shift
    local output
    output="$("$@")"
    if [[ "${output}" != "${EXPECTED}" ]]; then
        echo "consumer check: FAIL (${name} printed '${output}', expected '${EXPECTED}')" >&2
        exit 1
    fi
    echo "${name}: ${output}"
}

[[ -d "${PLUGIN_DIR}" ]] || { echo "no fixture plugins at ${PLUGIN_DIR}" >&2; exit 1; }

cmake --install "${BUILD_DIR}" --prefix "${PREFIX}" > "${WORK}/install.log"

# CMake package: nothing but the prefix.
cmake -S "${SOURCE_DIR}" -B "${WORK}/build" -DCMAKE_PREFIX_PATH="${PREFIX}" > "${WORK}/configure.log"
cmake --build "${WORK}/build" > "${WORK}/build.log"
expect_ok "cmake consumer_c" "${WORK}/build/consumer_c" "${PLUGIN_DIR}"
expect_ok "cmake consumer_cpp" "${WORK}/build/consumer_cpp" "${PLUGIN_DIR}"

# pkg-config.
PC_FILE="$(find "${PREFIX}" -name neuriplo.pc -print -quit)"
[[ -n "${PC_FILE}" ]] || { echo "neuriplo.pc not installed under ${PREFIX}" >&2; exit 1; }
PC_DIR="$(dirname "${PC_FILE}")"
LIB_DIR="$(dirname "${PC_DIR}")"
read -r -a PC_FLAGS <<< "$(PKG_CONFIG_PATH="${PC_DIR}" pkg-config --cflags --libs neuriplo)"
cc -std=c99 "${SOURCE_DIR}/consumer.c" -o "${WORK}/consumer_c_pkgconfig" "${PC_FLAGS[@]}"
expect_ok "pkg-config consumer_c" env LD_LIBRARY_PATH="${LIB_DIR}" "${WORK}/consumer_c_pkgconfig" "${PLUGIN_DIR}"

# Python ctypes ([T-16], [V-13]), against the same installed prefix.
if command -v python3 > /dev/null; then
    expect_ok "python ctypes" python3 "${SOURCE_DIR}/python/smoke_ctypes.py" "${PREFIX}" "${PLUGIN_DIR}"
else
    echo "consumer check: FAIL (python3 not found for the ctypes smoke test)" >&2
    exit 1
fi

echo "consumer check: PASS"
