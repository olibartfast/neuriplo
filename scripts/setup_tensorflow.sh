#!/bin/bash

# TensorFlow Build Setup Script
# This script automates the TensorFlow build process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TF_VERSION="r2.19"
PYTHON_VERSION="3.12"
BAZEL_VERSION="6.5.0"
CLANG_VERSION="17"

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root"
   exit 1
fi

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check available memory (at least 8GB)
    MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEM_GB" -lt 8 ]; then
        log_warning "Less than 8GB RAM available. Build may fail."
    fi
    
    # Check available disk space (at least 20GB)
    DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$DISK_GB" -lt 20 ]; then
        log_warning "Less than 20GB free disk space. Build may fail."
    fi
    
    log_success "System requirements check completed"
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    
    sudo apt update
    sudo apt install -y build-essential git curl wget python3-dev python3-pip python3-venv
    
    log_success "System dependencies installed"
}

# Setup Bazel
setup_bazel() {
    log_info "Setting up Bazel $BAZEL_VERSION..."
    
    # Add Bazel repository
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
    sudo mv bazel.gpg /etc/apt/keyrings/
    
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/bazel.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    
    sudo apt update
    sudo apt install -y bazel-$BAZEL_VERSION
    
    log_success "Bazel $BAZEL_VERSION installed"
}

# Setup Clang
setup_clang() {
    log_info "Setting up Clang $CLANG_VERSION..."
    
    sudo apt install -y llvm-$CLANG_VERSION clang-$CLANG_VERSION
    
    log_success "Clang $CLANG_VERSION installed"
}

# Setup virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."
    
    python3 -m venv tensorflow_build_env
    source tensorflow_build_env/bin/activate
    pip install -U pip
    
    log_success "Virtual environment created and activated"
}

# Clone and configure TensorFlow
setup_tensorflow() {
    log_info "Setting up TensorFlow $TF_VERSION..."
    
    if [ ! -d "tensorflow" ]; then
        git clone https://github.com/tensorflow/tensorflow.git
    fi
    
    cd tensorflow
    git checkout $TF_VERSION
    
    # Set environment variables
    export CC=/usr/bin/clang-$CLANG_VERSION
    export BAZEL_COMPILER=/usr/bin/clang-$CLANG_VERSION
    
    # Configure build
    log_info "Configuring TensorFlow build..."
    echo "N" | ./configure  # Non-interactive configuration for CPU-only build
    
    log_success "TensorFlow $TF_VERSION configured"
}

# Build TensorFlow
build_tensorflow() {
    log_info "Building TensorFlow (this may take 2-4 hours)..."
    
    bazel-$BAZEL_VERSION build //tensorflow/tools/pip_package:wheel \
        --repo_env=USE_PYWRAP_RULES=1 \
        --repo_env=WHEEL_NAME=tensorflow_cpu \
        --config=opt
    
    log_success "TensorFlow build completed"
}

# Install TensorFlow
install_tensorflow() {
    log_info "Installing TensorFlow..."
    
    WHEEL_PATH="bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl"
    
    if [ -f "$WHEEL_PATH" ]; then
        pip install "$WHEEL_PATH"
        log_success "TensorFlow installed successfully"
        
        # Verify installation
        python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
    else
        log_error "TensorFlow wheel not found at $WHEEL_PATH"
        exit 1
    fi
}

# Main execution
main() {
    log_info "Starting TensorFlow build setup..."
    
    check_requirements
    install_dependencies
    setup_bazel
    setup_clang
    setup_venv
    setup_tensorflow
    build_tensorflow
    install_tensorflow
    
    log_success "TensorFlow build and installation completed successfully!"
    log_info "To activate the environment in the future, run:"
    log_info "source tensorflow_build_env/bin/activate"
}

# Parse command line arguments
case "${1:-}" in
    --check-only)
        check_requirements
        exit 0
        ;;
    --setup-only)
        check_requirements
        install_dependencies
        setup_bazel
        setup_clang
        setup_venv
        setup_tensorflow
        log_success "Setup completed. Run the script again to build TensorFlow."
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo "Options:"
        echo "  --check-only    Check system requirements only"
        echo "  --setup-only    Setup environment without building"
        echo "  --help, -h      Show this help message"
        echo ""
        echo "Default: Complete setup and build process"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac 