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

# The install directory carries no version, so "does the directory exist" is
# satisfied by any LibTorch forever and the pin can never take effect -- that is
# how an install drifts away from versions.env unnoticed. Compare the version
# LibTorch stamps in build-version instead.
installed_version=""
if [ -f "$dir/build-version" ]; then
    # build-version looks like "2.3.0+cpu" or "2.0.1+cu118".
    installed_version="$(tr -d '[:space:]' < "$dir/build-version" | cut -d+ -f1)"
fi

if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    if [ "$installed_version" = "$version" ]; then
        echo "✓ LibTorch $version already installed at $dir"
        exit 0
    fi
    echo "LibTorch at $dir is ${installed_version:-an unrecognised version}, but versions.env pins $version."
    echo "Re-run with FORCE=true to replace it (this removes the existing $dir)."
    exit 0
fi

echo "Installing LibTorch $version..."

# Create directory and download
mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
wget -q "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip" -O tmp.zip
# Unzip merges into an existing tree, leaving a mix of both versions behind.
rm -rf "$dir"
unzip -q tmp.zip && rm tmp.zip

echo "✓ LibTorch $version installed successfully at $dir"
