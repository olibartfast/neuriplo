#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm -v "${SCRIPT_DIR}:/workspace" python:3.11 bash -lc \
    "pip install --no-cache-dir torch torchvision executorch >/tmp/executorch-pip.log 2>&1 && python /workspace/export_executorch_classifier.py"
