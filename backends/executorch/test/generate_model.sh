#!/bin/bash
# Usage: ./generate_model.sh [delegate]
# delegate: xnnpack (default) or portable - must match the EXECUTORCH_DELEGATE
# neuriplo was configured with.

set -e

DELEGATE="${1:-xnnpack}"
case "${DELEGATE}" in
    xnnpack|portable) ;;
    *)
        echo "Unsupported ExecuTorch delegate: ${DELEGATE}" >&2
        echo "Valid values: xnnpack, portable" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm -v "${SCRIPT_DIR}:/workspace" python:3.11 bash -lc \
    "pip install --no-cache-dir torch torchvision executorch >/tmp/executorch-pip.log 2>&1 && python /workspace/export_executorch_classifier.py --delegate '${DELEGATE}'"
