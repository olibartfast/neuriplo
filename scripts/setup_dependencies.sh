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
            echo "Backends: ONNX_RUNTIME, TENSORRT, LIBTORCH, OPENVINO"
            exit 0 ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done

# Validate backend
[[ -z "$BACKEND" ]] && { echo "Error: Backend required"; exit 1; }
case $BACKEND in
    ONNX_RUNTIME|TENSORRT|LIBTORCH|OPENVINO) ;;
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
    local version="2023.1.0"
    local dir="$DEPENDENCY_ROOT/openvino-$version"
    [[ -d "$dir" && "$FORCE" != "true" ]] && return 0
    echo "Error: Install OpenVINO $version manually from https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html to $dir"
    [[ -d "$dir" ]] || exit 1
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
            [[ -f "$DEPENDENCY_ROOT/openvino-2023.1.0/include/openvino/openvino.hpp" && -f "$DEPENDENCY_ROOT/openvino-2023.1.0/lib/libopenvino.so" ]] || { echo "Error: OpenVINO validation failed"; exit 1; } ;;
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
export OPENVINO_DIR="$DEPENDENCY_ROOT/openvino-2023.1.0"
export LD_LIBRARY_PATH="\$ONNX_RUNTIME_DIR/lib:\$TENSORRT_DIR/lib:\$LIBTORCH_DIR/lib:\$OPENVINO_DIR/lib:\$LD_LIBRARY_PATH"
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
esac
validate_installation "$BACKEND"
create_env_setup
echo "Setup complete. Source $DEPENDENCY_ROOT/setup_env.sh to use dependencies."