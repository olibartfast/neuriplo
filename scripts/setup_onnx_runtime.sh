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

# Create directory and download
mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
wget -q "https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-linux-x64-gpu-$version.tgz" -O tmp.tgz
tar -xzf tmp.tgz && rm tmp.tgz

echo "✓ ONNX Runtime $version installed successfully at $dir"
