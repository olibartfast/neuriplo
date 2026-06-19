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

INSTALL_DIR="${HOME}/dependencies/llamacpp"
if [[ "${1:-}" == "--install-dir" ]]; then
    INSTALL_DIR="${2:?--install-dir requires a path argument}"
fi

SRC_DIR="/tmp/llamacpp-src"

# ── Already installed? ────────────────────────────────────────────────────────
if [ -f "${INSTALL_DIR}/lib/libllama.so" ] && [ -f "${INSTALL_DIR}/include/llama.h" ]; then
    echo "✓ llama.cpp ${LLAMACPP_VERSION} already installed at ${INSTALL_DIR}"
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
cmake --install build

rm -rf "${SRC_DIR}"

echo ""
echo "✓ llama.cpp ${LLAMACPP_VERSION} installed to ${INSTALL_DIR}"
echo ""
echo "Configure neuriplo with:"
echo "  cmake -S . -B build -DDEFAULT_BACKEND=LLAMACPP -DLLAMACPP_DIR=${INSTALL_DIR} -DBUILD_INFERENCE_ENGINE_TESTS=ON"
