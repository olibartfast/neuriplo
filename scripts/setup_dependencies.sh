#!/bin/bash

set -e

# Defaults
BACKEND=""
DEPENDENCY_ROOT="$HOME/dependencies"
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--backend) BACKEND="$2"; shift 2 ;;
        -r|--root) DEPENDENCY_ROOT="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        -h|--help) 
            echo "Usage: $0 -b BACKEND [-r PATH] [-f] [-h]"
            echo "Backends: ONNX_RUNTIME, TENSORRT, LIBTORCH, OPENVINO, LIBTENSORFLOW"
            exit 0 ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done

# Validate backend
[[ -z "$BACKEND" ]] && { echo "Error: Backend required"; exit 1; }
case $BACKEND in
    ONNX_RUNTIME|TENSORRT|LIBTORCH|OPENVINO|LIBTENSORFLOW) ;;
    *) echo "Error: Unsupported backend: $BACKEND"; exit 1 ;;
esac

# Install system dependencies
install_system_deps() {
    local os=$(grep -oP '^ID=\K.*' /etc/os-release 2>/dev/null || echo "linux")
    case $os in
        ubuntu|debian) sudo apt-get update && sudo apt-get install -y build-essential cmake git wget curl unzip pkg-config libopencv-dev ;;
        centos|rhel|fedora) sudo yum groupinstall -y "Development Tools" && sudo yum install -y cmake git wget curl unzip pkg-config opencv-devel ;;
        *) echo "Warning: Unsupported OS: $os. Install dependencies manually." ;;
    esac
}

# Setup ONNX Runtime
setup_onnx_runtime() {
    local version="1.19.2"
    local dir="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-$version"
    [[ -d "$dir" && "$FORCE" != "true" ]] && return 0
    mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-linux-x64-gpu-$version.tgz" -O tmp.tgz
    tar -xzf tmp.tgz && rm tmp.tgz
}

# Setup TensorRT
setup_tensorrt() {
    local version="10.7.0.23"
    local dir="$DEPENDENCY_ROOT/TensorRT-$version"
    [[ -d "$dir" && "$FORCE" != "true" ]] && return 0
    echo "Error: Install TensorRT $version manually from https://developer.nvidia.com/tensorrt to $dir"
    [[ -d "$dir" ]] || exit 1
}

# Setup LibTorch
setup_libtorch() {
    local version="2.0.0"
    local dir="$DEPENDENCY_ROOT/libtorch"
    [[ -d "$dir" && "$FORCE" != "true" ]] && return 0
    mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
    wget -q "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip" -O tmp.zip
    unzip -q tmp.zip && rm tmp.zip
}

# Setup OpenVINO
setup_openvino() {
    local version="2025.2.0"
    local dir="$DEPENDENCY_ROOT/openvino_$version"
    [[ -d "$dir" && "$FORCE" != "true" ]] && return 0
    
    echo "Installing OpenVINO $version to $dir..."
    mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
    
    # Download OpenVINO toolkit
    local tarball="openvino_2025.2.0.tgz"
    if [[ ! -f "$tarball" ]]; then
        echo "Downloading OpenVINO toolkit..."
        curl -L "https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/linux/openvino_toolkit_ubuntu24_2025.2.0.19140.c01cd93e24d_x86_64.tgz" --output "$tarball"
    fi
    
    # Extract and move to final location
    echo "Extracting OpenVINO..."
    tar -xf "$tarball"
    if [[ -d "$dir" ]]; then
        rm -rf "$dir"
    fi
    mv openvino_toolkit_ubuntu24_2025.2.0.19140.c01cd93e24d_x86_64 "$dir"
    rm -f "$tarball"
    
    # Create a local Python virtual environment for OpenVINO tools
    echo "Setting up OpenVINO Python tools..."
    local venv_dir="$dir/python_env"
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    pip install openvino-dev
    deactivate
    
    # Create wrapper script for ovc
    mkdir -p "$dir/bin"
    cat > "$dir/bin/ovc" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../python_env"
source "$VENV_DIR/bin/activate"
ovc "$@"
deactivate
EOF
    chmod +x "$dir/bin/ovc"
    
    echo "OpenVINO $version installed successfully to $dir"
}

# Setup TensorFlow C++ Libraries
setup_libtensorflow() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local setup_script="$script_dir/setup_libtensorflow.sh"
    
    if [[ ! -f "$setup_script" ]]; then
        echo "Error: TensorFlow setup script not found at $setup_script"
        exit 1
    fi
    
    echo "Setting up TensorFlow C++ libraries..."
    if [[ "$FORCE" == "true" ]]; then
        "$setup_script" --force
    else
        "$setup_script"
    fi
}

# Validate installation
validate_installation() {
    case $1 in
        ONNX_RUNTIME)
            [[ -f "$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-1.19.2/include/onnxruntime_cxx_api.h" && -f "$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-1.19.2/lib/libonnxruntime.so" ]] || { echo "Error: ONNX Runtime validation failed"; exit 1; } ;;
        TENSORRT)
            [[ -f "$DEPENDENCY_ROOT/TensorRT-10.7.0.23/include/NvInfer.h" && -f "$DEPENDENCY_ROOT/TensorRT-10.7.0.23/lib/libnvinfer.so" ]] || { echo "Error: TensorRT validation failed"; exit 1; }
            command -v nvcc &>/dev/null || echo "Warning: CUDA not found. Install from https://developer.nvidia.com/cuda-downloads" ;;
        LIBTORCH)
            [[ -f "$DEPENDENCY_ROOT/libtorch/share/cmake/Torch/TorchConfig.cmake" ]] || { echo "Error: LibTorch validation failed"; exit 1; } ;;
        OPENVINO)
            [[ -f "$DEPENDENCY_ROOT/openvino_2025.2.0/runtime/include/openvino/openvino.hpp" && -f "$DEPENDENCY_ROOT/openvino_2025.2.0/runtime/lib/intel64/libopenvino.so" ]] || { echo "Error: OpenVINO validation failed"; exit 1; } ;;
        LIBTENSORFLOW)
            [[ -f "$DEPENDENCY_ROOT/tensorflow/include/tensorflow/cc/saved_model/loader.h" && -f "$DEPENDENCY_ROOT/tensorflow/lib/libtensorflow_cc.so" ]] || { echo "Error: TensorFlow C++ validation failed"; exit 1; } ;;
    esac
}

# Create environment setup script
create_env_setup() {
    local env_file="$DEPENDENCY_ROOT/setup_env.sh"
    cat > "$env_file" << EOF
#!/bin/bash
export DEPENDENCY_ROOT="$DEPENDENCY_ROOT"
export ONNX_RUNTIME_DIR="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-1.19.2"
export TENSORRT_DIR="$DEPENDENCY_ROOT/TensorRT-10.7.0.23"
export LIBTORCH_DIR="$DEPENDENCY_ROOT/libtorch"
export OPENVINO_DIR="$DEPENDENCY_ROOT/openvino_2025.2.0"
export TENSORFLOW_DIR="$DEPENDENCY_ROOT/tensorflow"
export LD_LIBRARY_PATH="\$ONNX_RUNTIME_DIR/lib:\$TENSORRT_DIR/lib:\$LIBTORCH_DIR/lib:\$OPENVINO_DIR/runtime/lib/intel64:\$TENSORFLOW_DIR/lib:\$LD_LIBRARY_PATH"
export PATH="\$OPENVINO_DIR/bin:\$PATH"
EOF
    chmod +x "$env_file"
}

# Main
install_system_deps
case $BACKEND in
    ONNX_RUNTIME) setup_onnx_runtime ;;
    TENSORRT) setup_tensorrt ;;
    LIBTORCH) setup_libtorch ;;
    OPENVINO) setup_openvino ;;
    LIBTENSORFLOW) setup_libtensorflow ;;
esac
validate_installation "$BACKEND"
create_env_setup
echo "Setup complete. Source $DEPENDENCY_ROOT/setup_env.sh to use dependencies."