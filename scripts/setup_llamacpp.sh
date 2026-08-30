#!/usr/bin/env bash
# Builds and installs llama.cpp shared libraries from source.
# Usage: ./scripts/setup_llamacpp.sh [--install-dir <path>]
# Default install dir: ~/dependencies/llamacpp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${ROOT_DIR}/versions.env" ]; then
    source "${ROOT_DIR}/versions.env"
else
    echo "Error: versions.env not found" >&2
    exit 1
fi

source "${SCRIPT_DIR}/lib/version_stamp.sh"

INSTALL_DIR="${HOME}/dependencies/llamacpp"
if [[ "${1:-}" == "--install-dir" ]]; then
    INSTALL_DIR="${2:?--install-dir requires a path argument}"
fi

SRC_DIR="/tmp/llamacpp-src"
FORCE="${FORCE:-false}"

# ── Already installed? ────────────────────────────────────────────────────────
# libllama.so existing does not say which build tag produced it, and
# INSTALL_DIR carries no version, so any past build satisfied every later pin.
if [ -f "${INSTALL_DIR}/lib/libllama.so" ] && [ -f "${INSTALL_DIR}/include/llama.h" ] && [ "${FORCE}" != "true" ]; then
    if neuriplo_stamp_matches "${INSTALL_DIR}" "${LLAMACPP_VERSION}" "llama.cpp"; then
        echo "✓ llama.cpp ${LLAMACPP_VERSION} already installed at ${INSTALL_DIR}"
    fi
    exit 0
fi

echo "Building llama.cpp ${LLAMACPP_VERSION} → ${INSTALL_DIR}"

for cmd in cmake git; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found" >&2; exit 1; }
done

# ── Clone ─────────────────────────────────────────────────────────────────────
rm -rf "${SRC_DIR}"
git clone https://github.com/ggerganov/llama.cpp.git "${SRC_DIR}"
cd "${SRC_DIR}"
git checkout "${LLAMACPP_VERSION}"

# ── Build ─────────────────────────────────────────────────────────────────────
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF
cmake --build build -j"$(nproc)"

# cmake --install merges into whatever the prefix already holds, so a forced
# rebuild after a version change would leave the previous tag's libraries in
# place and the stamp below would certify the mixture as the new version.
# llama.cpp does rename libraries across releases -- libggml.so became the split
# libggml-base/libggml-cpu pair -- and cmake/LinkBackend.cmake would still find
# the stale one alongside the new ABI. Install into a staging tree and swap, so
# INSTALL_DIR only ever holds one build; staging also means a failed build
# leaves the existing installation untouched rather than half-replaced.
STAGE_DIR="${INSTALL_DIR}.incoming"
rm -rf "${STAGE_DIR}"
cmake --install build --prefix "${STAGE_DIR}"
rm -rf "${INSTALL_DIR}"
mv "${STAGE_DIR}" "${INSTALL_DIR}"

rm -rf "${SRC_DIR}"

# Only now that the install is complete: a stamp always describes a usable tree.
neuriplo_write_stamp "${INSTALL_DIR}" "${LLAMACPP_VERSION}"

echo ""
echo "✓ llama.cpp ${LLAMACPP_VERSION} installed to ${INSTALL_DIR}"
echo ""
echo "Configure neuriplo with:"
echo "  cmake -S . -B build -DDEFAULT_BACKEND=LLAMACPP -DLLAMACPP_DIR=${INSTALL_DIR} -DBUILD_INFERENCE_ENGINE_TESTS=ON"
