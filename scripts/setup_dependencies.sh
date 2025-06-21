#!/bin/bash

# Unified setup script for InferenceEngines library dependencies
# This script installs inference backend dependencies based on the selected backend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BACKEND=""
DEPENDENCY_ROOT="$HOME/dependencies"
FORCE=false
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --backend BACKEND    Specify inference backend to setup"
    echo "                           Supported: ONNX_RUNTIME, TENSORRT, LIBTORCH, OPENVINO"
    echo "  -r, --root PATH          Set dependency installation root (default: $HOME/dependencies)"
    echo "  -f, --force              Force reinstallation of dependencies"
    echo "  -v, --verbose            Enable verbose output"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --backend ONNX_RUNTIME"
    echo "  $0 --backend TENSORRT --root /opt/dependencies"
    echo "  $0 --backend LIBTORCH --force"
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--backend)
                BACKEND="$2"
                shift 2
                ;;
            -r|--root)
                DEPENDENCY_ROOT="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to check if running in Docker
is_docker() {
    [[ -f /.dockerenv ]]
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ -f /etc/os-release ]]; then
            . /etc/os-release
            echo "$ID"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    local os=$(detect_os)
    print_status "Installing system dependencies for $os..."
    
    case $os in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                wget \
                curl \
                unzip \
                pkg-config \
                libopencv-dev \
                libglog-dev
            ;;
        centos|rhel|fedora)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                git \
                wget \
                curl \
                unzip \
                pkg-config \
                opencv-devel \
                glog-devel
            ;;
        *)
            print_warning "Unsupported OS: $os. Please install dependencies manually."
            ;;
    esac
}

# Function to setup ONNX Runtime
setup_onnx_runtime() {
    local version="1.19.2"
    local dir="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-$version"
    
    print_status "Setting up ONNX Runtime $version..."
    
    if [[ -d "$dir" && "$FORCE" != "true" ]]; then
        print_success "ONNX Runtime already installed at $dir"
        return 0
    fi
    
    mkdir -p "$DEPENDENCY_ROOT"
    cd "$DEPENDENCY_ROOT"
    
    local url="https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-linux-x64-gpu-$version.tgz"
    local archive="onnxruntime-linux-x64-gpu-$version.tgz"
    
    print_status "Downloading ONNX Runtime from $url..."
    wget -q "$url" -O "$archive"
    
    print_status "Extracting ONNX Runtime..."
    tar -xzf "$archive"
    rm "$archive"
    
    print_success "ONNX Runtime installed at $dir"
}

# Function to setup TensorRT
setup_tensorrt() {
    local version="10.7.0.23"
    local dir="$DEPENDENCY_ROOT/TensorRT-$version"
    
    print_status "Setting up TensorRT $version..."
    
    if [[ -d "$dir" && "$FORCE" != "true" ]]; then
        print_success "TensorRT already installed at $dir"
        return 0
    fi
    
    print_warning "TensorRT requires manual installation from NVIDIA website."
    print_status "Please download TensorRT $version from: https://developer.nvidia.com/tensorrt"
    print_status "Extract it to: $dir"
    
    if [[ ! -d "$dir" ]]; then
        print_error "TensorRT not found at $dir. Please install manually."
        exit 1
    fi
    
    print_success "TensorRT found at $dir"
}

# Function to setup LibTorch
setup_libtorch() {
    local version="2.0.0"
    local dir="$DEPENDENCY_ROOT/libtorch"
    
    print_status "Setting up LibTorch $version..."
    
    if [[ -d "$dir" && "$FORCE" != "true" ]]; then
        print_success "LibTorch already installed at $dir"
        return 0
    fi
    
    mkdir -p "$DEPENDENCY_ROOT"
    cd "$DEPENDENCY_ROOT"
    
    local url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip"
    local archive="libtorch-$version.zip"
    
    print_status "Downloading LibTorch from $url..."
    wget -q "$url" -O "$archive"
    
    print_status "Extracting LibTorch..."
    unzip -q "$archive"
    rm "$archive"
    
    print_success "LibTorch installed at $dir"
}

# Function to setup OpenVINO
setup_openvino() {
    local version="2023.1.0"
    local dir="$DEPENDENCY_ROOT/openvino-$version"
    
    print_status "Setting up OpenVINO $version..."
    
    if [[ -d "$dir" && "$FORCE" != "true" ]]; then
        print_success "OpenVINO already installed at $dir"
        return 0
    fi
    
    print_warning "OpenVINO requires manual installation from Intel website."
    print_status "Please download OpenVINO $version from: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html"
    print_status "Extract it to: $dir"
    
    if [[ ! -d "$dir" ]]; then
        print_error "OpenVINO not found at $dir. Please install manually."
        exit 1
    fi
    
    print_success "OpenVINO found at $dir"
}

# Function to setup CUDA (if needed)
setup_cuda() {
    if command -v nvcc &> /dev/null; then
        print_success "CUDA already installed: $(nvcc --version | head -n1)"
        return 0
    fi
    
    print_warning "CUDA not found. Please install CUDA manually from NVIDIA website."
    print_status "Download from: https://developer.nvidia.com/cuda-downloads"
}

# Function to validate installation
validate_installation() {
    local backend="$1"
    print_status "Validating $backend installation..."
    
    case $backend in
        ONNX_RUNTIME)
            local dir="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-1.19.2"
            if [[ -f "$dir/include/onnxruntime_cxx_api.h" && -f "$dir/lib/libonnxruntime.so" ]]; then
                print_success "ONNX Runtime validation passed"
            else
                print_error "ONNX Runtime validation failed"
                exit 1
            fi
            ;;
        TENSORRT)
            local dir="$DEPENDENCY_ROOT/TensorRT-10.7.0.23"
            if [[ -f "$dir/include/NvInfer.h" && -f "$dir/lib/libnvinfer.so" ]]; then
                print_success "TensorRT validation passed"
            else
                print_error "TensorRT validation failed"
                exit 1
            fi
            setup_cuda
            ;;
        LIBTORCH)
            local dir="$DEPENDENCY_ROOT/libtorch"
            if [[ -f "$dir/share/cmake/Torch/TorchConfig.cmake" ]]; then
                print_success "LibTorch validation passed"
            else
                print_error "LibTorch validation failed"
                exit 1
            fi
            ;;
        OPENVINO)
            local dir="$DEPENDENCY_ROOT/openvino-2023.1.0"
            if [[ -f "$dir/include/openvino/openvino.hpp" && -f "$dir/lib/libopenvino.so" ]]; then
                print_success "OpenVINO validation passed"
            else
                print_error "OpenVINO validation failed"
                exit 1
            fi
            ;;
    esac
}

# Function to create environment setup script
create_env_setup() {
    local backend="$1"
    local env_file="$DEPENDENCY_ROOT/setup_env.sh"
    
    print_status "Creating environment setup script..."
    
    cat > "$env_file" << EOF
#!/bin/bash
# Environment setup script for InferenceEngines dependencies
# Generated by setup_dependencies.sh

export DEPENDENCY_ROOT="$DEPENDENCY_ROOT"

# ONNX Runtime
export ONNX_RUNTIME_DIR="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-1.19.2"
export LD_LIBRARY_PATH="\$ONNX_RUNTIME_DIR/lib:\$LD_LIBRARY_PATH"

# TensorRT
export TENSORRT_DIR="$DEPENDENCY_ROOT/TensorRT-10.7.0.23"
export LD_LIBRARY_PATH="\$TENSORRT_DIR/lib:\$LD_LIBRARY_PATH"

# LibTorch
export LIBTORCH_DIR="$DEPENDENCY_ROOT/libtorch"
export LD_LIBRARY_PATH="\$LIBTORCH_DIR/lib:\$LD_LIBRARY_PATH"

# OpenVINO
export OPENVINO_DIR="$DEPENDENCY_ROOT/openvino-2023.1.0"
export LD_LIBRARY_PATH="\$OPENVINO_DIR/lib:\$LD_LIBRARY_PATH"

echo "InferenceEngines environment variables set"
EOF
    
    chmod +x "$env_file"
    print_success "Environment setup script created: $env_file"
}

# Main function
main() {
    print_status "InferenceEngines Dependency Setup Script"
    print_status "========================================"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Check if backend is specified
    if [[ -z "$BACKEND" ]]; then
        print_error "Backend must be specified. Use --help for usage information."
        exit 1
    fi
    
    # Validate backend
    case $BACKEND in
        ONNX_RUNTIME|TENSORRT|LIBTORCH|OPENVINO)
            ;;
        *)
            print_error "Unsupported backend: $BACKEND"
            print_status "Supported backends: ONNX_RUNTIME, TENSORRT, LIBTORCH, OPENVINO"
            exit 1
            ;;
    esac
    
    print_status "Backend: $BACKEND"
    print_status "Dependency root: $DEPENDENCY_ROOT"
    print_status "Force reinstall: $FORCE"
    
    # Install system dependencies
    install_system_dependencies
    
    # Setup specific backend
    case $BACKEND in
        ONNX_RUNTIME)
            setup_onnx_runtime
            ;;
        TENSORRT)
            setup_tensorrt
            ;;
        LIBTORCH)
            setup_libtorch
            ;;
        OPENVINO)
            setup_openvino
            ;;
    esac
    
    # Validate installation
    validate_installation "$BACKEND"
    
    # Create environment setup script
    create_env_setup "$BACKEND"
    
    print_success "Setup completed successfully!"
    print_status "To use the dependencies, source the environment setup script:"
    print_status "  source $DEPENDENCY_ROOT/setup_env.sh"
}

# Run main function with all arguments
main "$@" 