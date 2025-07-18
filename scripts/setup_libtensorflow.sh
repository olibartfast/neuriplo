#!/bin/bash

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SETUP_PIP_SCRIPT="$SCRIPT_DIR/setup_tensorflow_pip.sh"

# Parse arguments
CLEAN_INSTALL=false
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version) TENSORFLOW_VERSION="$2"; shift 2 ;;
        -c|--clean) CLEAN_INSTALL=true; shift ;;
        -f|--force) FORCE=true; shift ;;
        -h|--help) 
            echo "Usage: $0 [-v VERSION] [-c] [-f] [-h]"
            echo "Options:"
            echo "  -v, --version VERSION  TensorFlow version (default: from cmake/versions.cmake)"
            echo "  -c, --clean            Clean install (remove existing installation)"
            echo "  -f, --force            Force reinstall even if already installed"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "This script sets up TensorFlow C++ libraries using the pip-based installation method."
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check if setup_tensorflow_pip.sh exists
if [[ ! -f "$SETUP_PIP_SCRIPT" ]]; then
    echo "Error: TensorFlow pip setup script not found at $SETUP_PIP_SCRIPT"
    exit 1
fi

echo "TensorFlow C++ Setup for InferenceEngines"
echo "=========================================="
echo "This script will install TensorFlow C++ library dependencies using the pip-based method."
echo ""

# Build arguments for setup_tensorflow_pip.sh
PIP_ARGS=()
if [[ "$CLEAN_INSTALL" == "true" ]]; then
    PIP_ARGS+=("-c")
fi

if [[ "$FORCE" == "true" ]]; then
    PIP_ARGS+=("-f")
fi

if [[ -n "${TENSORFLOW_VERSION:-}" ]]; then
    PIP_ARGS+=("-v" "$TENSORFLOW_VERSION")
fi

# Run the pip setup script
echo "Running TensorFlow pip setup script..."
exec "$SETUP_PIP_SCRIPT" "${PIP_ARGS[@]}" 