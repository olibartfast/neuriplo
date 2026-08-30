#!/bin/bash
# Setup script for TensorRT backend
# This script is called by the unified setup_dependencies.sh
set -e

echo "Setting up TensorRT backend..."

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

# Default installation directory
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$HOME/dependencies}"
version="$TENSORRT_VERSION"
dir="$DEPENDENCY_ROOT/TensorRT-$version"

# The CUDA suffix selects which CUDA line the TensorRT build targets. CUDA_VERSION
# in versions.env is the version the CI images are built on; a developer machine
# may be on a different line, and a TensorRT built for CUDA 13 will not run
# against a CUDA 12 toolkit. Prefer the locally installed CUDA, falling back to
# the pinned value. Set TRT_CUDA_VERSION to override.
detect_local_cuda_major() {
    if command -v nvcc &> /dev/null; then
        nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1
    fi
}

if [ -z "${TRT_CUDA_VERSION:-}" ]; then
    TRT_CUDA_VERSION="$CUDA_VERSION"
    local_cuda_major="$(detect_local_cuda_major)"
    pinned_cuda_major="${CUDA_VERSION%%.*}"
    if [ -n "$local_cuda_major" ] && [ "$local_cuda_major" != "$pinned_cuda_major" ]; then
        case "$local_cuda_major" in
            12) TRT_CUDA_VERSION=12.9 ;;
            13) TRT_CUDA_VERSION=13.0 ;;
            *)
                echo "Warning: local CUDA $local_cuda_major.x has no known TensorRT build;" >&2
                echo "         falling back to the pinned CUDA $CUDA_VERSION." >&2
                ;;
        esac
        if [ "$TRT_CUDA_VERSION" != "$CUDA_VERSION" ]; then
            echo "Note: local CUDA toolkit is ${local_cuda_major}.x but versions.env pins CUDA ${CUDA_VERSION};"
            echo "      selecting the TensorRT build for CUDA ${TRT_CUDA_VERSION}."
            echo "      Set TRT_CUDA_VERSION to override."
        fi
    fi
fi

# Check if already installed. The install directory is named for the TensorRT
# version only, so it cannot tell which CUDA line the tree inside was built for --
# say which one is expected, so that upgrading the local CUDA and re-running does
# not silently leave a TensorRT built for the previous line in place.
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    echo "✓ TensorRT already installed at $dir"
    echo "  (expected build: CUDA ${TRT_CUDA_VERSION}; re-run with FORCE=true to reinstall)"
    exit 0
fi

# Derive the URL components from versions.env. The download path uses the
# three-component version (10.14.1) while the tarball keeps the full build
# number (10.14.1.48).
trt_short="$(echo "$version" | cut -d. -f1-3)"
tarball="TensorRT-${version}.Linux.x86_64-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz"
download_url="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_short}/tars/${tarball}"

echo "Installing TensorRT $version (CUDA $TRT_CUDA_VERSION)..."

# Create directory and download
mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
wget -q -O "$tarball" "$download_url" || {
    echo "Error: Failed to download TensorRT from $download_url"
    echo "Please download manually from https://developer.nvidia.com/tensorrt"
    rm -f "$tarball"
    exit 1
}

# Extract TensorRT
tar -xzf "$tarball" -C "$DEPENDENCY_ROOT" || {
    echo "Error: Failed to extract TensorRT"
    rm -f "$tarball"
    exit 1
}
rm -f "$tarball"

# Verify installation
if [[ ! -f "$dir/include/NvInfer.h" || ! -f "$dir/lib/libnvinfer.so" ]]; then
    echo "Error: TensorRT installation incomplete at $dir"
    exit 1
fi

echo "✓ TensorRT $version installed successfully at $dir"
