#!/bin/bash

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TENSORFLOW_VERSION=$(grep -m1 "set(TENSORFLOW_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g' || echo "2.19.0")
DEPENDENCIES_DIR="${DEPENDENCIES_DIR:-$HOME/dependencies}"
TENSORFLOW_DIR="$DEPENDENCIES_DIR/tensorflow"
VENV_DIR="$DEPENDENCIES_DIR/tensorflow_env"

# Parse arguments
CLEAN_INSTALL=false
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version) TENSORFLOW_VERSION="$2"; shift 2 ;;
        -c|--clean) CLEAN_INSTALL=true; shift ;;
        -f|--force) FORCE=true; shift ;;
        -h|--help) 
            echo "Usage: $0 [-v VERSION] [-c] [-f] [-h]"
            echo "Options:"
            echo "  -v, --version VERSION  TensorFlow version (default: $TENSORFLOW_VERSION)"
            echo "  -c, --clean            Clean install (remove existing installation)"
            echo "  -f, --force            Force reinstall even if already installed"
            echo "  -h, --help             Show this help message"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check if already installed and not forcing
if [[ -d "$TENSORFLOW_DIR" && "$FORCE" != "true" && "$CLEAN_INSTALL" != "true" ]]; then
    echo "TensorFlow C++ libraries already installed at $TENSORFLOW_DIR"
    echo "Use -f to force reinstall or -c for clean install"
    exit 0
fi

# Check Python
command -v python3 &>/dev/null || { echo "Python 3 required"; exit 1; }

# Clean if requested
if [[ "$CLEAN_INSTALL" == "true" ]]; then
    echo "Cleaning existing TensorFlow installation..."
    rm -rf "$TENSORFLOW_DIR" "$VENV_DIR"
fi

# Create virtual environment
mkdir -p "$DEPENDENCIES_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating TensorFlow virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install TensorFlow
echo "Installing TensorFlow $TENSORFLOW_VERSION..."
pip install --upgrade pip
pip install "tensorflow==$TENSORFLOW_VERSION" || { echo "TensorFlow installation failed"; exit 1; }

# Verify installation
echo "Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Setup TensorFlow C++ libraries
TF_SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages/tensorflow"
if [[ ! -d "$TF_SITE_PACKAGES" ]]; then
    echo "Error: TensorFlow not found in virtual environment"
    exit 1
fi

echo "Setting up TensorFlow C++ libraries..."
rm -rf "$TENSORFLOW_DIR"
mkdir -p "$TENSORFLOW_DIR/lib" "$TENSORFLOW_DIR/include"

# Copy libraries and headers
echo "Copying TensorFlow libraries..."
cp "$TF_SITE_PACKAGES/libtensorflow_cc.so.2" "$TENSORFLOW_DIR/lib/"
cp "$TF_SITE_PACKAGES/libtensorflow_framework.so.2" "$TENSORFLOW_DIR/lib/"
ln -sf "libtensorflow_cc.so.2" "$TENSORFLOW_DIR/lib/libtensorflow_cc.so"
ln -sf "libtensorflow_framework.so.2" "$TENSORFLOW_DIR/lib/libtensorflow_framework.so"

echo "Copying TensorFlow headers..."
cp -r "$TF_SITE_PACKAGES/include/"* "$TENSORFLOW_DIR/include/"

# Copy additional headers if available
if [[ -d "$TF_SITE_PACKAGES/core" ]]; then
    echo "Copying core headers..."
    cp -r "$TF_SITE_PACKAGES/core" "$TENSORFLOW_DIR/include/tensorflow/"
fi

if [[ -d "$TF_SITE_PACKAGES/cc" ]]; then
    echo "Copying cc headers..."
    cp -r "$TF_SITE_PACKAGES/cc" "$TENSORFLOW_DIR/include/tensorflow/"
fi

# Create pkg-config file
echo "Creating pkg-config file..."
mkdir -p "$TENSORFLOW_DIR/lib/pkgconfig"
cat > "$TENSORFLOW_DIR/lib/pkgconfig/tensorflow.pc" << EOF
prefix=$TENSORFLOW_DIR
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: TensorFlow
Version: $TENSORFLOW_VERSION
Libs: -L\${libdir} -ltensorflow_cc -ltensorflow_framework
Cflags: -I\${includedir}
EOF

# Setup environment
echo "Setting up environment variables..."
env_file="$DEPENDENCIES_DIR/setup_env.sh"

# Remove existing TensorFlow entries
if [[ -f "$env_file" ]]; then
    grep -v "TensorFlow\|TENSORFLOW" "$env_file" > "${env_file}.tmp" 2>/dev/null || true
    mv "${env_file}.tmp" "$env_file" 2>/dev/null || true
fi

# Add TensorFlow environment variables
cat >> "$env_file" << EOF

# TensorFlow C++ Libraries
export TENSORFLOW_DIR="$TENSORFLOW_DIR"
export LD_LIBRARY_PATH="\$TENSORFLOW_DIR/lib:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="\$TENSORFLOW_DIR/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
EOF

# Deactivate virtual environment
deactivate

echo "TensorFlow C++ setup complete!"
echo "Installation directory: $TENSORFLOW_DIR"
echo "Virtual environment: $VENV_DIR"
echo "Source $env_file to use TensorFlow C++ libraries"
echo ""
echo "To use TensorFlow C++ in your project:"
echo "  export TENSORFLOW_DIR=\"$TENSORFLOW_DIR\""
echo "  export LD_LIBRARY_PATH=\"\$TENSORFLOW_DIR/lib:\$LD_LIBRARY_PATH\"" 