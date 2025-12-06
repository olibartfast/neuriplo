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
# Basic configuration for CPU support with optional LLVM
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
    -DUSE_GRAPH_EXECUTOR_DEBUG=ON \
    -DUSE_RELAY_DEBUG=ON \
    -DBUILD_SHARED_LIBS=ON

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

# Install Python package
echo "Installing TVM Python package..."
cd ../python
if command -v pip3 &> /dev/null; then
    pip3 install -e . --user
elif command -v pip &> /dev/null; then
    pip install -e . --user
else
    echo "Warning: pip not found. Skipping Python package installation."
    echo "Install Python dependencies manually: pip install numpy decorator attrs tornado psutil xgboost cloudpickle"
fi

# Install additional Python dependencies
echo "Installing TVM Python dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install --user numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle || echo "Warning: Some Python dependencies may not have installed"
elif command -v pip &> /dev/null; then
    pip install --user numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle || echo "Warning: Some Python dependencies may not have installed"
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

# Verify Python installation
echo "Verifying TVM Python installation..."
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
