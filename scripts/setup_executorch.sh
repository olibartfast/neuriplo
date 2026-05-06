#!/bin/bash
# Builds and installs ExecuTorch C++ runtime from source.
# Usage: ./scripts/setup_executorch.sh [--install-dir <path>]
# Default install dir: ~/dependencies/executorch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${ROOT_DIR}/versions.env" ]; then
    source "${ROOT_DIR}/versions.env"
else
    echo "Error: versions.env not found" >&2
    exit 1
fi

INSTALL_DIR="${HOME}/dependencies/executorch"
if [[ "${1:-}" == "--install-dir" ]]; then
    INSTALL_DIR="${2:?--install-dir requires a path argument}"
fi

SRC_DIR="/tmp/executorch-${EXECUTORCH_VERSION}-src"

# ── Already installed? ────────────────────────────────────────────────────────
if [ -f "${INSTALL_DIR}/lib/libexecutorch.a" ]; then
    echo "✓ ExecuTorch ${EXECUTORCH_VERSION} already installed at ${INSTALL_DIR}"
    exit 0
fi

echo "Building ExecuTorch ${EXECUTORCH_VERSION} → ${INSTALL_DIR}"

# ── System prerequisites ──────────────────────────────────────────────────────
for cmd in cmake ninja python3 pip3 git; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found" >&2; exit 1; }
done

python3 - <<'EOF'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    print(f"Error: Python 3.10+ required, got {major}.{minor}", file=sys.stderr)
    sys.exit(1)
EOF

# ── Clone ─────────────────────────────────────────────────────────────────────
if [ ! -d "${SRC_DIR}/.git" ]; then
    git clone \
        --branch "${EXECUTORCH_VERSION}" \
        --depth 1 \
        https://github.com/pytorch/executorch.git \
        "${SRC_DIR}"
fi

cd "${SRC_DIR}"

# Submodules needed for the C++ build (flatbuffers, cpuinfo, etc.)
git submodule sync --recursive
git submodule update --init --recursive --depth 1

# ── Python build dependencies ─────────────────────────────────────────────────
# ExecuTorch uses Python for flatbuffers codegen during cmake configure.
pip3 install --quiet tomli zstd setuptools wheel

# install_requirements.sh installs the executorch Python package and its deps.
if [ -f "${SRC_DIR}/install_requirements.sh" ]; then
    bash "${SRC_DIR}/install_requirements.sh" --pybind off 2>&1 | tail -5
fi

# ── CMake configure ───────────────────────────────────────────────────────────
mkdir -p cmake-out
cmake -S . -B cmake-out \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_TESTS=OFF \
    -DEXECUTORCH_BUILD_EXAMPLES=OFF \
    -DPYTHON_EXECUTABLE="$(command -v python3)"

# ── Build & install ───────────────────────────────────────────────────────────
cmake --build cmake-out --config Release -j"$(nproc)"
cmake --install cmake-out

echo ""
echo "✓ ExecuTorch ${EXECUTORCH_VERSION} installed to ${INSTALL_DIR}"
echo ""
echo "Configure neuriplo with:"
echo "  cmake -S . -B build -DDEFAULT_BACKEND=EXECUTORCH -DEXECUTORCH_DIR=${INSTALL_DIR} -DBUILD_INFERENCE_ENGINE_TESTS=ON"
