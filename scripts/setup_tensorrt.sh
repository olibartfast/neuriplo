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
local version="$TENSORRT_VERSION"
local dir="$DEPENDENCY_ROOT/TensorRT-$version"

# Check if already installed
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    echo "✓ TensorRT already installed at $dir"
    exit 0
fi

echo "Error: Install TensorRT $version manually from https://developer.nvidia.com/tensorrt to $dir"
echo "Please download TensorRT from NVIDIA Developer Portal and extract it to $dir"
[[ -d "$dir" ]] || exit 1

echo "✓ TensorRT $version found at $dir"
