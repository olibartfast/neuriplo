#!/bin/bash

# Setup script for ONNX Runtime backend
# This script is called by the unified setup_dependencies.sh

set -e

echo "Setting up ONNX Runtime backend..."

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

# Default installation directory
version="$ONNX_RUNTIME_VERSION"
dir="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-$version"

# Check if already installed
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    echo "✓ ONNX Runtime already installed at $dir"
    exit 0
fi

echo "Installing ONNX Runtime $version..."

# ONNX Runtime 1.28 split the GPU build per CUDA major and renamed it
# (onnxruntime-linux-x64-gpu_cuda13-<ver>); older releases publish a single
# onnxruntime-linux-x64-gpu-<ver>. Try the build matching the pinned CUDA
# first, then the old name. Either way the install lands at $dir, the path
# cmake/versions.cmake and the other scripts expect.
cuda_major="${CUDA_VERSION%%.*}"
base_url="https://github.com/microsoft/onnxruntime/releases/download/v$version"
mkdir -p "$DEPENDENCY_ROOT"
tmp_dir="$(mktemp -d "$DEPENDENCY_ROOT/.onnxruntime.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

downloaded=""
for archive in "onnxruntime-linux-x64-gpu_cuda${cuda_major}-$version" "onnxruntime-linux-x64-gpu-$version"; do
    if wget -q "$base_url/$archive.tgz" -O "$tmp_dir/ort.tgz"; then
        downloaded="$archive"
        break
    fi
done
if [[ -z "$downloaded" ]]; then
    echo "Error: no ONNX Runtime $version GPU build found for CUDA $cuda_major under $base_url" >&2
    exit 1
fi
echo "Downloaded $downloaded"

tar -xzf "$tmp_dir/ort.tgz" -C "$tmp_dir"
rm -rf "$dir"
mv "$tmp_dir/$downloaded" "$dir"

echo "✓ ONNX Runtime $version installed successfully at $dir"
