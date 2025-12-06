#!/bin/bash

set -e

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

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
            echo "Backends: ONNX_RUNTIME, TENSORRT, LIBTORCH, OPENVINO, LIBTENSORFLOW, GGML, TVM"
            exit 0 ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done

# Validate backend
[[ -z "$BACKEND" ]] && { echo "Error: Backend required"; exit 1; }
case $BACKEND in
    ONNX_RUNTIME|TENSORRT|LIBTORCH|OPENVINO|LIBTENSORFLOW|GGML|TVM) ;;
    *) echo "Error: Unsupported backend: $BACKEND"; exit 1 ;;
esac

# Install system dependencies
install_system_deps() {
    local os=$(grep -oP '^ID=\K.*' /etc/os-release 2>/dev/null || echo "linux")
    case $os in
        ubuntu|debian) sudo apt-get update && sudo apt-get install -y build-essential cmake git wget curl unzip pkg-config libopencv-dev libopenblas-dev ;;
        centos|rhel|fedora) sudo yum groupinstall -y "Development Tools" && sudo yum install -y cmake git wget curl unzip pkg-config opencv-devel openblas-devel ;;
        *) echo "Warning: Unsupported OS: $os. Install dependencies manually." ;;
    esac
}

# Setup ONNX Runtime
setup_onnx_runtime() {
    echo "Setting up ONNX Runtime..."
    DEPENDENCY_ROOT="$DEPENDENCY_ROOT" FORCE="$FORCE" ./scripts/setup_onnx_runtime.sh
}

# Setup TensorRT
setup_tensorrt() {
    echo "Setting up TensorRT..."
    DEPENDENCY_ROOT="$DEPENDENCY_ROOT" FORCE="$FORCE" ./scripts/setup_tensorrt.sh
}

# Setup LibTorch
setup_libtorch() {
    echo "Setting up LibTorch..."
    DEPENDENCY_ROOT="$DEPENDENCY_ROOT" FORCE="$FORCE" ./scripts/setup_libtorch.sh
}

# Setup OpenVINO
setup_openvino() {
    echo "Setting up OpenVINO..."
    DEPENDENCY_ROOT="$DEPENDENCY_ROOT" FORCE="$FORCE" ./scripts/setup_openvino.sh
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

# Setup GGML
setup_ggml() {
    echo "Setting up GGML library..."

    # Call the dedicated GGML setup script
    DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" FORCE="${FORCE}" ./scripts/setup_ggml.sh
}

# Setup TVM
setup_tvm() {
    echo "Setting up TVM library..."

    # Check if setup script exists
    if [[ -f "./scripts/setup_tvm.sh" ]]; then
        DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" FORCE="${FORCE}" ./scripts/setup_tvm.sh
    else
        echo "Note: scripts/setup_tvm.sh not found."
        echo "Please refer to docs/TVM_BUILD_GUIDE.md for manual installation instructions."
        echo "TVM should be installed to: $DEPENDENCY_ROOT/tvm"
        exit 1
    fi
}

# Validate installation
validate_installation() {
    case $1 in
        ONNX_RUNTIME)
            [[ -f "$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-$ONNX_RUNTIME_VERSION/include/onnxruntime_cxx_api.h" && -f "$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-$ONNX_RUNTIME_VERSION/lib/libonnxruntime.so" ]] || { echo "Error: ONNX Runtime validation failed"; exit 1; } ;;
        TENSORRT)
            [[ -f "$DEPENDENCY_ROOT/TensorRT-$TENSORRT_VERSION/include/NvInfer.h" && -f "$DEPENDENCY_ROOT/TensorRT-$TENSORRT_VERSION/lib/libnvinfer.so" ]] || { echo "Error: TensorRT validation failed"; exit 1; }
            command -v nvcc &>/dev/null || echo "Warning: CUDA not found. Install from https://developer.nvidia.com/cuda-downloads" ;;
        LIBTORCH)
            [[ -f "$DEPENDENCY_ROOT/libtorch/share/cmake/Torch/TorchConfig.cmake" ]] || { echo "Error: LibTorch validation failed"; exit 1; } ;;
        OPENVINO)
            [[ -f "$DEPENDENCY_ROOT/openvino_$OPENVINO_VERSION/runtime/include/openvino/openvino.hpp" && -f "$DEPENDENCY_ROOT/openvino_$OPENVINO_VERSION/runtime/lib/intel64/libopenvino.so" ]] || { echo "Error: OpenVINO validation failed"; exit 1; } ;;
        LIBTENSORFLOW)
            [[ -f "$DEPENDENCY_ROOT/tensorflow/include/tensorflow/cc/saved_model/loader.h" && -f "$DEPENDENCY_ROOT/tensorflow/lib/libtensorflow_cc.so" ]] || { echo "Error: TensorFlow C++ validation failed"; exit 1; } ;;
        GGML)
            [[ -f "$DEPENDENCY_ROOT/ggml/include/ggml.h" && -f "$DEPENDENCY_ROOT/ggml/lib/libggml.so" ]] || { echo "Error: GGML validation failed"; exit 1; } ;;
        TVM)
            [[ -f "$DEPENDENCY_ROOT/tvm/include/tvm/runtime/c_runtime_api.h" && -f "$DEPENDENCY_ROOT/tvm/build/libtvm_runtime.so" ]] || { echo "Error: TVM validation failed"; exit 1; } ;;
    esac
}

# Create environment setup script
create_env_setup() {
    local env_file="$DEPENDENCY_ROOT/setup_env.sh"
    cat > "$env_file" << EOF
#!/bin/bash
export DEPENDENCY_ROOT="$DEPENDENCY_ROOT"
export ONNX_RUNTIME_DIR="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-$ONNX_RUNTIME_VERSION"
export TENSORRT_DIR="$DEPENDENCY_ROOT/TensorRT-$TENSORRT_VERSION"
export LIBTORCH_DIR="$DEPENDENCY_ROOT/libtorch"
export OPENVINO_DIR="$DEPENDENCY_ROOT/openvino_$OPENVINO_VERSION"
export TENSORFLOW_DIR="$DEPENDENCY_ROOT/tensorflow"
export GGML_DIR="\$DEPENDENCY_ROOT/ggml"
export TVM_DIR="\$DEPENDENCY_ROOT/tvm"
export LD_LIBRARY_PATH="\$ONNX_RUNTIME_DIR/lib:\$TENSORRT_DIR/lib:\$LIBTORCH_DIR/lib:\$OPENVINO_DIR/runtime/lib/intel64:\$TENSORFLOW_DIR/lib:\$GGML_DIR/lib:\$TVM_DIR/build:\$LD_LIBRARY_PATH"
export PATH="\$OPENVINO_DIR/bin:\$TVM_DIR/bin:\$PATH"
export PYTHONPATH="\$TVM_DIR/python:\$PYTHONPATH"
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
    GGML) setup_ggml ;;
    TVM) setup_tvm ;;
esac
validate_installation "$BACKEND"
create_env_setup
echo "Setup complete. Source $DEPENDENCY_ROOT/setup_env.sh to use dependencies."