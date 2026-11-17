#!/bin/bash

# Generate a GGUF model for GGML backend testing.
# This script converts a ResNet18 ONNX model to the GGUF format read by
# GGMLInfer (strict-GGUF-only: the backend rejects anything but .gguf).

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Generating GGUF Test Model ===${NC}"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: This script should be run from the test directory"
    exit 1
fi

# Check if the conversion script exists
CONVERSION_SCRIPT="../../../../scripts/convert_onnx_to_gguf.py"
if [ ! -f "$CONVERSION_SCRIPT" ]; then
    echo -e "${YELLOW}Conversion script not found, creating placeholder model...${NC}"
    echo "/tmp/test_ggml_model.bin" > model_path.txt
    echo "Note: This is a placeholder. Run the conversion script to create a real model:"
    echo "  python3 ../../../../scripts/convert_onnx_to_gguf.py --input <model.onnx> --output resnet18.gguf"
    exit 0
fi

if [ -z "$ONNX_MODEL" ]; then
    echo -e "${YELLOW}ONNX_MODEL not set, creating placeholder model...${NC}"
    echo "Usage: ONNX_MODEL=/path/to/resnet18.onnx $0"
    echo "/tmp/test_ggml_model.bin" > model_path.txt
    exit 0
fi

# Run the conversion script
echo "Converting ResNet18 to GGUF format..."
python3 "$CONVERSION_SCRIPT" --input "$ONNX_MODEL" --output resnet18.gguf

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GGUF model generated successfully${NC}"
    echo "Model: resnet18.gguf"
    echo "$(pwd)/resnet18.gguf" > model_path.txt
    echo "Test path: model_path.txt"
else
    echo -e "${YELLOW}Conversion failed, creating placeholder...${NC}"
    echo "/tmp/test_ggml_model.bin" > model_path.txt
fi
