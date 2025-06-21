#!/bin/bash

# Individual TensorRT setup script for InferenceEngines
# This script is a convenience wrapper around the unified setup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFIED_SCRIPT="$SCRIPT_DIR/setup_dependencies.sh"

if [[ ! -f "$UNIFIED_SCRIPT" ]]; then
    echo "Error: Unified setup script not found at $UNIFIED_SCRIPT"
    exit 1
fi

echo "TensorRT Setup for InferenceEngines"
echo "==================================="
echo "This script will install TensorRT dependencies."
echo "Note: TensorRT requires manual download from NVIDIA website."
echo ""

# Run the unified script with TensorRT backend
exec "$UNIFIED_SCRIPT" --backend TENSORRT "$@" 