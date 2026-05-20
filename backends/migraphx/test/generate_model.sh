#!/bin/bash
# Generate a minimal ONNX ResNet-18 model for MIGraphX integration tests.
# Requires: docker with pytorch/pytorch image (no local GPU needed).
docker run --rm -v "$(pwd):/workspace" pytorch/pytorch:latest \
    bash -c "cd /workspace && python export_onnx_classifier.py"
