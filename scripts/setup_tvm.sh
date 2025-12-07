#!/bin/bash

# Setup script for TVM backend
# This script is called by the unified setup_dependencies.sh

set -e

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

# Default installation directory
TVM_DIR="${DEPENDENCY_ROOT}/tvm"
BUILD_DIR="${TVM_DIR}/build"

# Check if TVM is already installed
if [ -d "$TVM_DIR" ] && [ -f "$BUILD_DIR/libtvm_runtime.so" ] && [ -f "$TVM_DIR/include/tvm/runtime/c_runtime_api.h" ] && [ "$FORCE" != "true" ]; then
    echo "✓ TVM already installed at $TVM_DIR"
    exit 0
fi

echo "Setting up TVM library..."

# Install system dependencies
echo "Checking system dependencies..."

# Check for LLVM
if ! command -v llvm-config &> /dev/null && ! ls /usr/lib/llvm-*/bin/llvm-config 2>/dev/null | head -1 > /dev/null; then
    echo "Installing LLVM development packages..."
    sudo apt-get update
    sudo apt-get install -y llvm-18-dev || \
    sudo apt-get install -y llvm-17-dev || \
    sudo apt-get install -y llvm-16-dev || \
    sudo apt-get install -y llvm-15-dev || \
    sudo apt-get install -y llvm-14-dev || \
    sudo apt-get install -y llvm-13-dev || \
    sudo apt-get install -y llvm-12-dev || {
        echo "Warning: Could not install LLVM. TVM will be built without LLVM support."
    }
fi

# Check for other build dependencies
echo "Installing build dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config \
    libtinfo-dev \
    zlib1g-dev \
    libedit-dev \
    libxml2-dev \
    python3-dev || {
    echo "Warning: Some build dependencies may not have installed"
}

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not running in a virtual environment"
    echo "   It's recommended to create a virtual environment for TVM:"
    echo "   python3 -m venv tvm-env && source tvm-env/bin/activate"
    echo ""
    read -p "Continue without virtual environment? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Create a virtual environment and try again."
        exit 1
    fi
else
    echo "✓ Running in virtual environment: $VIRTUAL_ENV"
fi

# Create dependency root directory if it doesn't exist
mkdir -p "$DEPENDENCY_ROOT"
cd "$DEPENDENCY_ROOT"

# Clone TVM repository if not exists
if [ ! -d "tvm" ]; then
    echo "Cloning TVM repository (this may take a few minutes)..."
    git clone --recursive https://github.com/apache/tvm tvm
fi

cd tvm

# Checkout specific version if TVM_VERSION is set and not "latest"
if [ -n "$TVM_VERSION" ] && [ "$TVM_VERSION" != "latest" ]; then
    echo "Checking out TVM version v${TVM_VERSION}..."
    git fetch --tags
    git checkout "v${TVM_VERSION}"
    git submodule update --init --recursive
else
    echo "Using latest TVM from main branch..."
    git checkout main
    git pull origin main
    git submodule update --init --recursive
fi

# Create build directory
echo "Configuring TVM build..."
mkdir -p build
cd build

# Detect LLVM installation
LLVM_CONFIG=""
for version in 18 17 16 15 14 13 12 11 10; do
    if command -v llvm-config-${version} &> /dev/null; then
        LLVM_CONFIG=$(llvm-config-${version} --bindir)/llvm-config
        echo "Found LLVM ${version} at: $LLVM_CONFIG"
        break
    elif [ -f "/usr/lib/llvm-${version}/bin/llvm-config" ]; then
        LLVM_CONFIG="/usr/lib/llvm-${version}/bin/llvm-config"
        echo "Found LLVM ${version} at: $LLVM_CONFIG"
        break
    fi
done

if [ -z "$LLVM_CONFIG" ] && command -v llvm-config &> /dev/null; then
    LLVM_CONFIG=$(command -v llvm-config)
    echo "Found LLVM at: $LLVM_CONFIG"
fi

if [ -z "$LLVM_CONFIG" ]; then
    echo "Warning: LLVM not found. TVM will be built without LLVM support."
    echo "For better CPU performance, install LLVM: sudo apt-get install llvm-12-dev"
    USE_LLVM_FLAG="OFF"
else
    USE_LLVM_FLAG="$LLVM_CONFIG"
fi

# Configure CMake build
# Use build directory directly (no installation needed)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_LLVM="$USE_LLVM_FLAG" \
    -DUSE_CUDA=OFF \
    -DUSE_OPENCL=OFF \
    -DUSE_VULKAN=OFF \
    -DUSE_METAL=OFF \
    -DUSE_ROCM=OFF \
    -DUSE_RTTI=ON \
    -DUSE_GRAPH_EXECUTOR=ON \
    -DUSE_PROFILER=ON \
    -DUSE_RELAY_DEBUG=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_GTEST=OFF

# Fix DLPack header issue (aligned with Docker setup)
echo "Fixing DLPack header structure..."
if [ ! -f "${TVM_DIR}/3rdparty/dlpack/dlpack.h" ] && [ -f "${TVM_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h" ]; then
    mkdir -p "${TVM_DIR}/3rdparty/dlpack"
    ln -sf "${TVM_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h" "${TVM_DIR}/3rdparty/dlpack/dlpack.h"
    echo "✓ DLPack header symlink created"
fi

# Build TVM
echo "Building TVM (this may take 10-30 minutes)..."
make -j$(nproc)

# Verify installation
if [ -f "$BUILD_DIR/libtvm_runtime.so" ]; then
    echo "✓ TVM runtime library built successfully"
else
    echo "✗ TVM build failed - libtvm_runtime.so not found"
    exit 1
fi

if [ -f "$BUILD_DIR/libtvm.so" ]; then
    echo "✓ TVM full library built successfully"
else
    echo "Warning: libtvm.so not found (only runtime built)"
fi

# Install Python dependencies first (aligned with Docker setup)
echo "Installing TVM Python dependencies..."

# Detect pip flags based on environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Installing in virtual environment (no additional flags needed)"
    PIP_FLAGS=""
else
    echo "Installing with --user flag (not in virtual environment)"
    PIP_FLAGS="--user"
    # Check if we need --break-system-packages flag for newer pip versions
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        if pip3 install --help 2>/dev/null | grep -q "break-system-packages"; then
            echo "Detected newer pip version, using --break-system-packages flag"
            PIP_FLAGS="--break-system-packages"
        fi
    fi
fi

if command -v pip3 &> /dev/null; then
    echo "Installing base Python dependencies..."
    pip3 install $PIP_FLAGS numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle onnx onnxruntime || echo "Warning: Some Python dependencies may not have installed"
    
    # Install tvm-ffi (critical for TVM Python bindings)
    echo "Installing tvm-ffi..."
    cd ../3rdparty/tvm-ffi
    pip3 install $PIP_FLAGS .
    
    # Install TVM Python package
    echo "Installing TVM Python package..."
    cd ../../python
    pip3 install $PIP_FLAGS -e .
elif command -v pip &> /dev/null; then
    echo "Installing base Python dependencies..."
    pip install $PIP_FLAGS numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle onnx onnxruntime || echo "Warning: Some Python dependencies may not have installed"
    
    # Install tvm-ffi (critical for TVM Python bindings)
    echo "Installing tvm-ffi..."
    cd ../3rdparty/tvm-ffi
    pip install $PIP_FLAGS .
    
    # Install TVM Python package
    echo "Installing TVM Python package..."
    cd ../../python
    pip install $PIP_FLAGS -e .
else
    echo "Warning: pip not found. Skipping Python package installation."
    echo "Install Python dependencies manually: pip install numpy decorator attrs tornado psutil xgboost cloudpickle onnx onnxruntime"
fi

# Create environment setup script
cat > "$TVM_DIR/setup_env.sh" << 'EOF'
#!/bin/bash
export TVM_HOME="$TVM_DIR"
export TVM_DIR="$TVM_DIR"
export PYTHONPATH="${TVM_DIR}/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${TVM_DIR}/build:${LD_LIBRARY_PATH}"
EOF

# Replace $TVM_DIR with actual path in setup_env.sh
sed -i "s|\$TVM_DIR|${TVM_DIR}|g" "$TVM_DIR/setup_env.sh"
chmod +x "$TVM_DIR/setup_env.sh"

# Verify installation (aligned with Docker setup)
echo "Verifying TVM installation..."
export TVM_HOME="${TVM_DIR}"
export PYTHONPATH="${TVM_DIR}/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${TVM_DIR}/build:${LD_LIBRARY_PATH}"

if [ -f "${TVM_DIR}/build/libtvm_runtime.so" ] && [ -f "${TVM_DIR}/build/libtvm.so" ]; then
    echo "✓ TVM libraries found"
else
    echo "✗ TVM library verification failed"
    exit 1
fi

if python3 -c "import tvm; print('TVM version:', tvm.__version__)" 2>/dev/null; then
    echo "✓ TVM Python package installed successfully"
else
    echo "Warning: TVM Python package verification failed"
    echo "You may need to manually set PYTHONPATH=${TVM_DIR}/python"
fi

echo "✓ TVM setup completed"
echo ""
echo "Installation directory: $TVM_DIR"
echo "Build directory: $BUILD_DIR"
echo ""
echo "To use TVM, source the environment setup script:"
echo "  source $TVM_DIR/setup_env.sh"
echo ""
echo "Or add to your shell profile (~/.bashrc or ~/.zshrc):"
echo "  export TVM_HOME=$TVM_DIR"
echo "  export PYTHONPATH=\${TVM_HOME}/python:\${PYTHONPATH}"
echo "  export LD_LIBRARY_PATH=\${TVM_HOME}/build:\${LD_LIBRARY_PATH}"
