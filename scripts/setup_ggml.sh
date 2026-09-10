#!/bin/bash

# Setup script for GGML backend
# This script is called by the unified setup_dependencies.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

source "${SCRIPT_DIR}/lib/version_stamp.sh"

# Default installation directory
GGML_DIR="${DEPENDENCY_ROOT}/ggml"
BUILD_DIR="${DEPENDENCY_ROOT}/ggml/build"
FORCE="${FORCE:-false}"

# Check if GGML is already installed. The libraries existing is not enough:
# the install directory carries no version, so any past build answered that
# question yes and the pin in versions.env could never take effect.
if [ -f "$GGML_DIR/lib/libggml.so" ] && [ -f "$GGML_DIR/include/ggml.h" ] && [ "$FORCE" != "true" ]; then
    if neuriplo_stamp_matches "$GGML_DIR" "$GGML_VERSION" "GGML"; then
        echo "✓ GGML $GGML_VERSION already installed at $GGML_DIR"
    fi
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

# versions.env has pinned GGML_VERSION all along, but this script checked out
# master and pulled, so the pin never took effect and two runs a day apart
# produced two different GGMLs. Check out the pin, and fail loudly if it does
# not resolve -- silently falling back to a moving branch is what caused the
# drift in the first place.
echo "Checking out GGML $GGML_VERSION..."
git fetch --tags --force
if ! git checkout --detach "$GGML_VERSION"; then
    echo "Error: GGML_VERSION '$GGML_VERSION' does not resolve to a ref in the ggml repository." >&2
    echo "Correct the pin in versions.env." >&2
    exit 1
fi

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
fi

# Installing merges into whatever GGML_DIR already holds, so a rebuild after a
# version change would leave the previous tag's libraries in place and the stamp
# below would certify the mixture as the new version. ggml splits and renames
# libraries across releases -- libggml.so became the libggml-base/libggml-cpu
# pair -- and cmake/LinkBackend.cmake would still find the stale one alongside
# the new ABI. Install into a staging tree and swap, so GGML_DIR only ever holds
# one build; both configure branches above install the same way, so this runs
# once for either.
GGML_STAGE_DIR="${GGML_DIR}.incoming"
rm -rf "$GGML_STAGE_DIR"
cmake --install . --prefix "$GGML_STAGE_DIR"
rm -rf "$GGML_DIR"
mv "$GGML_STAGE_DIR" "$GGML_DIR"

# Verify installation
if [ -f "$GGML_DIR/lib/libggml.so" ] && [ -f "$GGML_DIR/include/ggml.h" ]; then
    # Only now that the tree is known good: a stamp always describes a usable install.
    neuriplo_write_stamp "$GGML_DIR" "$GGML_VERSION"
    echo "✓ GGML $GGML_VERSION installed successfully at $GGML_DIR"
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
