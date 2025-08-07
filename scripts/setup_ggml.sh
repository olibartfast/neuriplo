#!/bin/bash

# Setup script for GGML backend
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
GGML_DIR="${DEPENDENCY_ROOT}/ggml"
BUILD_DIR="${DEPENDENCY_ROOT}/ggml/build"

# Check if GGML is already installed
if [ -d "$GGML_DIR" ] && [ -f "$GGML_DIR/lib/libggml.so" ] && [ -f "$GGML_DIR/include/ggml.h" ] && [ "$FORCE" != "true" ]; then
    echo "✓ GGML already installed at $GGML_DIR"
    exit 0
fi

echo "Setting up GGML library..."

# Create directories
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone GGML repository if not exists
if [ ! -d "ggml" ]; then
    echo "Cloning GGML repository..."
    git clone https://github.com/ggerganov/ggml.git
fi

cd ggml

# Checkout stable version
echo "Checking out stable version..."
git checkout master
git pull origin master

# Build GGML
echo "Building GGML..."

# Create build directory
mkdir -p build
cd build

# Try with BLAS first
if cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$GGML_DIR" \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    -DGGML_AVX=ON \
    -DGGML_AVX2=ON \
    -DGGML_F16C=ON \
    -DGGML_FMA=ON; then
    
    echo "Building GGML with BLAS support..."
    make -j$(nproc)
    make install
else
    echo "BLAS not found, building GGML without BLAS support..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$GGML_DIR" \
        -DGGML_BLAS=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
        -DGGML_AVX=ON \
        -DGGML_AVX2=ON \
        -DGGML_F16C=ON \
        -DGGML_FMA=ON
    
    make -j$(nproc)
    make install
fi

# Verify installation
if [ -f "$GGML_DIR/lib/libggml.so" ] && [ -f "$GGML_DIR/include/ggml.h" ]; then
    echo "✓ GGML installed successfully at $GGML_DIR"
else
    echo "✗ GGML installation failed"
    exit 1
fi

# Create environment setup script
cat > "$GGML_DIR/setup_env.sh" << EOF
#!/bin/bash
export GGML_DIR="$GGML_DIR"
export LD_LIBRARY_PATH="\${GGML_DIR}/lib:\${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="\${GGML_DIR}/lib/pkgconfig:\${PKG_CONFIG_PATH}"
EOF

chmod +x "$GGML_DIR/setup_env.sh"

echo "✓ GGML setup completed"
