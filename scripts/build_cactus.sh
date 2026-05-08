#!/usr/bin/env bash
# Build the Cactus Docker image.
# Cactus v1.14 requires an ARM64 host — fails early with a clear message on x86_64.

set -e

arch=$(uname -m)
if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then
    echo ""
    echo "Cactus backend requires an ARM64 host."
    echo "  Detected: $arch"
    echo "  Cactus v1.14 uses ARM NEON intrinsics unconditionally and cannot be compiled on x86_64."
    echo "  Run this on a Jetson Orin, Raspberry Pi 5, or any aarch64 host."
    echo ""
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker build --rm -t neuriplo:cactus -f "$REPO_ROOT/docker/Dockerfile.cactus" "$REPO_ROOT" "$@"
