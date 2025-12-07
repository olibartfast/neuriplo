#!/bin/bash

# Setup script for LibTorch backend
# This script is called by the unified setup_dependencies.sh

set -e

echo "Setting up LibTorch backend..."

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

# Default installation directory
version="$PYTORCH_VERSION"
dir="$DEPENDENCY_ROOT/libtorch"

# Check if already installed
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    echo "✓ LibTorch already installed at $dir"
    exit 0
fi

echo "Installing LibTorch $version..."

# Create directory and download
mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
wget -q "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip" -O tmp.zip
unzip -q tmp.zip && rm tmp.zip

echo "✓ LibTorch $version installed successfully at $dir"
