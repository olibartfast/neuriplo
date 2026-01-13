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
version="$TENSORRT_VERSION"
dir="$DEPENDENCY_ROOT/TensorRT-$version"

# Check if already installed
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    echo "✓ TensorRT already installed at $dir"
    exit 0
fi

# Parse version components
export TRT_MAJOR=10
export TRT_MINOR=.13
export TRT_PATCH=.3
export TRT_BUILD=.9
export TRT_VERSION=${TRT_MAJOR}${TRT_MINOR}${TRT_PATCH}${TRT_BUILD}
export TRT_CUDA_VERSION=13.0

# Download TensorRT
echo "Downloading TensorRT ${TRT_VERSION}..."
download_url="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TRT_MAJOR}${TRT_MINOR}${TRT_PATCH}/tars/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz"
tarball="TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz"

wget -O "$tarball" "$download_url" || {
    echo "Error: Failed to download TensorRT"
    echo "Please download manually from https://developer.nvidia.com/tensorrt"
    exit 1
}

# Extract TensorRT
# echo "Extracting TensorRT to $DEPENDENCY_ROOT..."
# tar -xzf "$tarball" -C "$DEPENDENCY_ROOT" || {
#     echo "Error: Failed to extract TensorRT"
#     exit 1
# }

# # Clean up tarball
# rm "$tarball"

# # Verify installation
# if [[ -d "$dir" ]]; then
#     echo "✓ TensorRT $version successfully installed at $dir"
# else
#     echo "Error: TensorRT directory not found after extraction"
#     exit 1
# fi