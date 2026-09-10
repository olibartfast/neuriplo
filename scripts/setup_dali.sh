#!/usr/bin/env bash
# Extract the C++ NVIDIA DALI distribution.
#
# NVIDIA publishes no standalone C++ DALI package: the headers and shared
# libraries ship inside the nvidia-dali pip wheel, whose filename carries an
# opaque build number, so there is no stable URL to pin. This downloads the
# wheel for the pinned version and lays out the include/ and lib files where
# -DDALI_DIR expects them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load versions from versions.env unless the caller pinned one explicitly
if [ -z "${DALI_VERSION:-}" ] && [ -f "${ROOT_DIR}/versions.env" ]; then
    source "${ROOT_DIR}/versions.env"
fi

if [ -z "${DALI_VERSION:-}" ]; then
    echo "Error: DALI_VERSION is not set and versions.env was not found" >&2
    exit 1
fi
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$HOME/dependencies}"
TARGET="${1:-$DEPENDENCY_ROOT/dali}"

echo "Installing NVIDIA DALI ${DALI_VERSION} into ${TARGET}"
mkdir -p "${TARGET}"
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

python3 -m pip download --no-deps --dest "${workdir}" \
    --extra-index-url https://pypi.nvidia.com \
    "nvidia-dali-cuda120==${DALI_VERSION}"

wheel="$(find "${workdir}" -name 'nvidia_dali*.whl' -print -quit)"
if [[ -z "${wheel}" ]]; then
    echo "error: no nvidia-dali wheel downloaded" >&2
    exit 1
fi

python3 -m zipfile -e "${wheel}" "${workdir}/extracted"
src="${workdir}/extracted/nvidia/dali"

cp -a "${src}/include" "${TARGET}/"
# libdali_operators.so must come across too: libdali.so does not reference it
# (DALI's Python bindings dlopen it), and a C++ host has to link it explicitly.
for lib in libdali.so libdali_core.so libdali_kernels.so libdali_operators.so; do
    cp -a "${src}/${lib}" "${TARGET}/"
done
# Bundled third-party libraries resolved through rpath.
[[ -d "${src}/.libs" ]] && cp -a "${src}/.libs" "${TARGET}/"

echo "DALI ready. Configure with -DDALI_DIR=${TARGET}"
