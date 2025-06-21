#!/bin/bash

# Individual OpenVINO setup script for InferenceEngines
# This script is a convenience wrapper around the unified setup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFIED_SCRIPT="$SCRIPT_DIR/setup_dependencies.sh"

if [[ ! -f "$UNIFIED_SCRIPT" ]]; then
    echo "Error: Unified setup script not found at $UNIFIED_SCRIPT"
    exit 1
fi

echo "OpenVINO Setup for InferenceEngines"
echo "==================================="
echo "This script will install OpenVINO dependencies."
echo "Note: OpenVINO requires manual download from Intel website."
echo ""

# Run the unified script with OpenVINO backend
exec "$UNIFIED_SCRIPT" --backend OPENVINO "$@" 