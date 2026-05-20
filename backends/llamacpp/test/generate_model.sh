#!/bin/bash

# Generate a llama.cpp (GGUF) model for testing.
# Downloads a small quantised model from Hugging Face if available,
# otherwise writes a placeholder model_path.txt so tests fall back to mock.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Generating llama.cpp Test Model ===${NC}"

MODEL_DIR="${1:-.}"
MODEL_FILE="$MODEL_DIR/gemma-4-E2B-it-Q4_K_M.gguf"
HF_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo -e "${GREEN}✓ Model already present: $MODEL_FILE${NC}"
    echo "$MODEL_FILE" > model_path.txt
    exit 0
fi

echo "Attempting to download small GGUF model..."
if command -v wget >/dev/null 2>&1; then
    wget -q -O "$MODEL_FILE" "$HF_URL" && {
        echo -e "${GREEN}✓ Model downloaded successfully${NC}"
        echo "$MODEL_FILE" > model_path.txt
        exit 0
    }
fi

echo -e "${YELLOW}Download failed or wget not available. Creating placeholder.${NC}"
echo "/tmp/test_llamacpp_model.gguf" > model_path.txt
echo "Note: Run the following to get a real model:"
echo "  wget -O $MODEL_FILE $HF_URL"
