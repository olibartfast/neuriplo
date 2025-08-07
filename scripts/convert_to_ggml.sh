#!/bin/bash

# Convert ResNet18 PyTorch model to GGML format
# This script downloads a pretrained ResNet18 model and converts it to GGML format

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OUTPUT_MODEL="resnet18.ggml"
TEST_DIR="backends/ggml/test"
USE_VENV=true
VENV_DIR="temp_ggml_venv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_MODEL="$2"
            shift 2
            ;;
        -t|--test-dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Convert ResNet18 PyTorch model to GGML format"
            echo ""
            echo "Options:"
            echo "  -o, --output PATH    Output GGML model path (default: resnet18.ggml)"
            echo "  -t, --test-dir PATH  Test directory to copy model to (default: backends/ggml/test)"
            echo "  --no-venv           Don't use virtual environment (use system Python)"
            echo "  --venv-dir PATH     Virtual environment directory (default: temp_ggml_venv)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== ResNet18 to GGML Converter ===${NC}"

# Function to cleanup virtual environment
cleanup_venv() {
    if [ "$USE_VENV" = true ] && [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Cleaning up virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    fi
}

# Set trap to cleanup on exit
trap cleanup_venv EXIT

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed${NC}"
    exit 1
fi

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

# Check if venv module is available
if ! python3 -c "import venv" 2>/dev/null; then
    echo -e "${RED}Error: Python venv module is not available${NC}"
    echo "Install it with: sudo apt-get install python3-venv"
    exit 1
fi

if [ "$USE_VENV" = true ]; then
    echo -e "${BLUE}Setting up temporary virtual environment...${NC}"
    
    # Create virtual environment
    python3 -m venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
    
    # Install required packages
    echo "Installing PyTorch dependencies..."
    pip install --upgrade pip
    pip install torch torchvision numpy
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install dependencies${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Using system Python (--no-venv specified)${NC}"
    
    # Check if required packages are available in system Python
    echo "Checking system Python dependencies..."
    if ! python3 -c "import torch, torchvision, numpy" 2>/dev/null; then
        echo -e "${RED}Error: PyTorch, torchvision, or numpy not found in system Python${NC}"
        echo "Install them with: pip3 install torch torchvision numpy"
        echo "Or use virtual environment (default behavior)"
        exit 1
    fi
    
    echo -e "${GREEN}✓ System Python dependencies check passed${NC}"
fi

# Run the conversion script
echo -e "${BLUE}Starting conversion...${NC}"
python3 scripts/convert_resnet18_to_ggml.py --output "$OUTPUT_MODEL" --test-dir "$TEST_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Conversion completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Set DEFAULT_BACKEND=GGML in your CMake configuration"
    echo "2. Set GGML_DIR environment variable to point to your GGML installation"
    echo "3. Build the project with GGML backend"
    echo "4. Run tests with the converted model"
else
    echo -e "${RED}✗ Conversion failed${NC}"
    exit 1
fi
