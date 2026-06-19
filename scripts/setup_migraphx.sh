#!/usr/bin/env bash
# Installs MIGraphX via the ROCm apt repository.
# Usage: ./scripts/setup_migraphx.sh
#
# MIGraphX ships as part of ROCm — there is no separate source build.
# This script installs the apt packages and validates the installation.
# Requires: ROCm already installed at /opt/rocm, AMD GPU with ROCm support.

set -euo pipefail

ROCM_ROOT="/opt/rocm"
MIGRAPHX_INCLUDE="${ROCM_ROOT}/include/migraphx"
MIGRAPHX_LIB="${ROCM_ROOT}/lib/libmigraphx.so"

# ── Already installed? ────────────────────────────────────────────────────────
if [ -d "${MIGRAPHX_INCLUDE}" ] && [ -f "${MIGRAPHX_LIB}" ]; then
    echo "✓ MIGraphX already installed at ${ROCM_ROOT}"
    exit 0
fi

# ── ROCm prerequisite ─────────────────────────────────────────────────────────
if [ ! -d "${ROCM_ROOT}" ]; then
    echo "Error: ROCm not found at ${ROCM_ROOT}." >&2
    echo "Install ROCm first: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html" >&2
    exit 1
fi

echo "Installing MIGraphX from ROCm apt repository..."

# ── Install ───────────────────────────────────────────────────────────────────
sudo apt-get update
sudo apt-get install -y migraphx migraphx-dev

# ── Validate ──────────────────────────────────────────────────────────────────
[ -d "${MIGRAPHX_INCLUDE}" ] || { echo "Error: MIGraphX headers not found after install." >&2; exit 1; }
[ -f "${MIGRAPHX_LIB}" ]    || { echo "Error: libmigraphx.so not found after install." >&2; exit 1; }

echo ""
echo "✓ MIGraphX installed at ${ROCM_ROOT}"
echo ""
echo "Configure neuriplo with:"
echo "  cmake -S . -B build -DDEFAULT_BACKEND=MIGRAPHX -DMIGRAPHX_ROOT=${ROCM_ROOT} -DBUILD_INFERENCE_ENGINE_TESTS=ON"
echo ""
echo "Run the Docker test image on a ROCm-capable host:"
echo "  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video neuriplo:migraphx"
