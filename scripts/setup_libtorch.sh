#!/bin/bash

# Individual LibTorch setup script for InferenceEngines
# This script is a convenience wrapper around the unified setup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFIED_SCRIPT="$SCRIPT_DIR/setup_dependencies.sh"

if [[ ! -f "$UNIFIED_SCRIPT" ]]; then
    echo "Error: Unified setup script not found at $UNIFIED_SCRIPT"
    exit 1
fi

echo "LibTorch Setup for InferenceEngines"
echo "==================================="
echo "This script will install LibTorch dependencies."
echo ""

# Run the unified script with LibTorch backend
exec "$UNIFIED_SCRIPT" --backend LIBTORCH "$@" 