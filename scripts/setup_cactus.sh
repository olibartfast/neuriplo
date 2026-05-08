#!/usr/bin/env bash
# Builds and installs the Cactus shared library from source.
# Usage: ./scripts/setup_cactus.sh [--install-dir <path>]
# Default install dir: ~/dependencies/cactus
#
# IMPORTANT: Cactus requires an ARM64 (aarch64) host.
# The library uses ARM NEON intrinsics unconditionally and cannot be
# compiled on x86_64.  Tested targets: Jetson Orin, Raspberry Pi 5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Architecture guard ────────────────────────────────────────────────────────
arch=$(uname -m)
if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then
    echo ""
    echo "Error: Cactus requires an ARM64 host."
    echo "  Detected: $arch"
    echo "  Cactus uses ARM NEON intrinsics unconditionally and cannot be compiled on x86_64."
    echo "  Run this on a Jetson Orin, Raspberry Pi 5, or any aarch64 host."
    echo ""
    exit 1
fi

# ── Version ───────────────────────────────────────────────────────────────────
if [ -f "${ROOT_DIR}/versions.env" ]; then
    source "${ROOT_DIR}/versions.env"
else
    echo "Error: versions.env not found" >&2
    exit 1
fi

INSTALL_DIR="${HOME}/dependencies/cactus"
if [[ "${1:-}" == "--install-dir" ]]; then
    INSTALL_DIR="${2:?--install-dir requires a path argument}"
fi

SRC_DIR="/tmp/cactus-src"

# ── Already installed? ────────────────────────────────────────────────────────
if [ -f "${INSTALL_DIR}/lib/libcactus.so" ] && [ -f "${INSTALL_DIR}/include/cactus.h" ]; then
    echo "✓ Cactus ${CACTUS_VERSION} already installed at ${INSTALL_DIR}"
    exit 0
fi

echo "Building Cactus ${CACTUS_VERSION} → ${INSTALL_DIR}"

for cmd in cmake git python3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found" >&2; exit 1; }
done

# ── Clone ─────────────────────────────────────────────────────────────────────
rm -rf "${SRC_DIR}"
git clone https://github.com/cactus-compute/cactus.git "${SRC_DIR}"
cd "${SRC_DIR}"
git checkout "${CACTUS_VERSION}"
git submodule update --init --recursive

# ── Apply arch-detection patch (removes hard-coded ARM flags for generic build)
PATCH_SCRIPT="${ROOT_DIR}/patches/cactus-v1.14-arch-detect.py"
if [ -f "${PATCH_SCRIPT}" ]; then
    python3 "${PATCH_SCRIPT}" "${SRC_DIR}/cactus/CMakeLists.txt"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir -p "${INSTALL_DIR}/include" "${INSTALL_DIR}/lib"
mkdir -p "${SRC_DIR}/cactus/build"
cd "${SRC_DIR}/cactus/build"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON
make -j"$(nproc)"

# ── Install ───────────────────────────────────────────────────────────────────
cp "${SRC_DIR}/cactus/cactus.h" "${INSTALL_DIR}/include/"
find "${SRC_DIR}/cactus/build" -name 'libcactus.so*' -exec cp {} "${INSTALL_DIR}/lib/" \;
# Ensure the unversioned symlink exists
if [ ! -f "${INSTALL_DIR}/lib/libcactus.so" ]; then
    versioned=$(ls "${INSTALL_DIR}/lib/libcactus.so."* 2>/dev/null | head -1)
    [ -n "${versioned}" ] && ln -s "${versioned}" "${INSTALL_DIR}/lib/libcactus.so"
fi

rm -rf "${SRC_DIR}"

echo ""
echo "✓ Cactus ${CACTUS_VERSION} installed to ${INSTALL_DIR}"
echo ""
echo "Configure neuriplo with:"
echo "  cmake -S . -B build -DDEFAULT_BACKEND=CACTUS -DCACTUS_DIR=${INSTALL_DIR} -DBUILD_INFERENCE_ENGINE_TESTS=ON"
