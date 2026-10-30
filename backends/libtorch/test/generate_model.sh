#!/usr/bin/env bash
# Use a matching Python environment; never trace with an unpinned latest image.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/export_torchvision_classifier.py" "$@"
