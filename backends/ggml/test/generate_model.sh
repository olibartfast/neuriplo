#!/bin/bash

# Generate a GGML model for testing
# This script converts ResNet18 PyTorch model to GGML format

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Generating GGML Test Model ===${NC}"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: This script should be run from the test directory"
    exit 1
fi

# Check if the conversion script exists
CONVERSION_SCRIPT="../../../../scripts/convert_to_ggml.sh"
if [ ! -f "$CONVERSION_SCRIPT" ]; then
    echo -e "${YELLOW}Conversion script not found, creating placeholder model...${NC}"
    echo "/tmp/test_ggml_model.bin" > model_path.txt
    echo "Note: This is a placeholder. Run the conversion script to create a real model:"
    echo "  ../../../../scripts/convert_to_ggml.sh -o resnet18.ggml -t ."
    exit 0
fi

# Run the conversion script
echo "Converting ResNet18 to GGML format..."
"$CONVERSION_SCRIPT" -o resnet18.ggml -t .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GGML model generated successfully${NC}"
    echo "Model: resnet18.ggml"
    echo "Test path: model_path.txt"
else
    echo -e "${YELLOW}Conversion failed, creating placeholder...${NC}"
    echo "/tmp/test_ggml_model.bin" > model_path.txt
fi
