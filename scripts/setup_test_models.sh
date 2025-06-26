#!/bin/bash

# Model Setup Script for InferenceEngines Testing
# This script sets up test models for all backends

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to setup models for each backend
setup_models() {
    log_info "Setting up test models for all backends..."
    
    cd "$PROJECT_ROOT"
    
    # Create a temporary directory for model generation
    TEMP_MODEL_DIR="temp_models"
    mkdir -p "$TEMP_MODEL_DIR"
    cd "$TEMP_MODEL_DIR"
    
    # Generate base ONNX model (used by multiple backends)
    log_info "Generating base ResNet-18 ONNX model..."
    cat > generate_base_model.py << 'EOF'
import torch
import torchvision.models as models
import torch.onnx as onnx
import os

# Load pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Example input to trace the model
example_input = torch.rand(1, 3, 224, 224)

# Export the model to ONNX
model_name = "resnet18.onnx"
with torch.no_grad():
    onnx.export(
        model,
        example_input,
        model_name,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )

print(f"Base ONNX model exported to {model_name}")

# Also create model_path.txt for consistency
with open("model_path.txt", "w") as f:
    f.write(model_name)
EOF
    
    if python3 generate_base_model.py; then
        log_success "Base ONNX model generated successfully"
    else
        log_error "Failed to generate base ONNX model"
        return 1
    fi
    
    # Copy base model to each backend test directory
    for backend_dir in "$PROJECT_ROOT"/backends/*/test/; do
        if [ -d "$backend_dir" ]; then
            backend_name=$(basename "$(dirname "$backend_dir")")
            log_info "Setting up models for $backend_name backend..."
            
            # Copy base files
            cp resnet18.onnx "$backend_dir/" 2>/dev/null || true
            cp model_path.txt "$backend_dir/" 2>/dev/null || true
            
            # Make generation scripts executable
            chmod +x "$backend_dir"/*.sh 2>/dev/null || true
        fi
    done
    
    # Generate TensorFlow SavedModel
    log_info "Generating TensorFlow SavedModel..."
    tf_test_dir="$PROJECT_ROOT/backends/libtensorflow/test"
    if [ -d "$tf_test_dir" ]; then
        cd "$tf_test_dir"
        if [ -f "generate_tf_model.py" ]; then
            python3 generate_tf_model.py || log_warning "Failed to generate TensorFlow model"
        fi
    fi
    
    # Cleanup
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_MODEL_DIR"
    
    log_success "Model setup completed for all backends"
}

# Function to check dependencies for model generation
check_dependencies() {
    log_info "Checking dependencies for model generation..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required for model generation"
        return 1
    fi
    
    # Check PyTorch
    if ! python3 -c "import torch; import torchvision" 2>/dev/null; then
        log_warning "PyTorch/torchvision not found. Some model generation may fail."
        log_info "Install with: pip install torch torchvision"
    fi
    
    # Check TensorFlow (optional)
    if ! python3 -c "import tensorflow" 2>/dev/null; then
        log_warning "TensorFlow not found. TensorFlow model generation will be skipped."
        log_info "Install with: pip install tensorflow"
    fi
    
    log_success "Dependency check completed"
}

# Function to verify generated models
verify_models() {
    log_info "Verifying generated models..."
    
    total_models=0
    valid_models=0
    
    for backend_dir in "$PROJECT_ROOT"/backends/*/test/; do
        if [ -d "$backend_dir" ]; then
            backend_name=$(basename "$(dirname "$backend_dir")")
            cd "$backend_dir"
            
            case $backend_name in
                "opencv-dnn"|"onnx-runtime")
                    if [ -f "resnet18.onnx" ]; then
                        log_success "$backend_name: ONNX model found"
                        ((valid_models++))
                    else
                        log_warning "$backend_name: ONNX model missing"
                    fi
                    ((total_models++))
                    ;;
                "libtorch")
                    if [ -f "resnet18.pt" ] || [ -f "resnet18.onnx" ]; then
                        log_success "$backend_name: Model found"
                        ((valid_models++))
                    else
                        log_warning "$backend_name: Model missing"
                    fi
                    ((total_models++))
                    ;;
                "libtensorflow")
                    if [ -d "saved_model" ]; then
                        log_success "$backend_name: SavedModel found"
                        ((valid_models++))
                    else
                        log_warning "$backend_name: SavedModel missing"
                    fi
                    ((total_models++))
                    ;;
                "tensorrt")
                    if [ -f "resnet18.engine" ] || [ -f "resnet18.onnx" ]; then
                        log_success "$backend_name: Model found"
                        ((valid_models++))
                    else
                        log_warning "$backend_name: Model missing"
                    fi
                    ((total_models++))
                    ;;
                "openvino")
                    if [ -f "resnet18.xml" ] && [ -f "resnet18.bin" ]; then
                        log_success "$backend_name: IR files found"
                        ((valid_models++))
                    elif [ -f "resnet18.onnx" ]; then
                        log_success "$backend_name: ONNX model found"
                        ((valid_models++))
                    else
                        log_warning "$backend_name: Model missing"
                    fi
                    ((total_models++))
                    ;;
            esac
        fi
    done
    
    log_info "Model verification: $valid_models/$total_models backends have models"
}

# Main function
main() {
    log_info "Starting model setup for InferenceEngines backends"
    
    # Parse command line arguments
    local verify_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verify-only)
                verify_only=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --verify-only    Only verify existing models, don't generate new ones"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    if [ "$verify_only" = true ]; then
        verify_models
    else
        check_dependencies
        setup_models
        verify_models
    fi
    
    log_success "Model setup script completed"
}

# Run main function with all arguments
main "$@"
