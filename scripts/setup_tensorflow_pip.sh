#!/bin/bash

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TENSORFLOW_VERSION=$(grep -m1 "set(TENSORFLOW_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g' || echo "${TF_VERSION:-2.19.0}")
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

# Check Python
command -v python3 &>/dev/null || { echo "Python 3 required"; exit 1; }

# Check if already installed and not forcing
if [[ -d "$TENSORFLOW_DIR" && "$FORCE" != "true" && "$CLEAN_INSTALL" != "true" ]]; then
    echo "TensorFlow C++ libraries already installed at $TENSORFLOW_DIR"
    echo "Use -f to force reinstall or -c for clean install"
    exit 0
fi

# Clean if requested
[[ "$CLEAN_INSTALL" == "true" ]] && rm -rf "$TENSORFLOW_DIR" "$VENV_DIR"

# Create virtual environment
mkdir -p "$DEPENDENCIES_DIR"
[[ -d "$VENV_DIR" ]] || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install TensorFlow
pip install --upgrade pip
pip install "tensorflow==$TENSORFLOW_VERSION" || { echo "TensorFlow installation failed"; exit 1; }

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Setup TensorFlow
TF_SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages/tensorflow"
[[ -d "$TF_SITE_PACKAGES" ]] || { echo "TensorFlow not found"; exit 1; }

rm -rf "$TENSORFLOW_DIR"
mkdir -p "$TENSORFLOW_DIR/lib" "$TENSORFLOW_DIR/include"

# Copy libraries and headers
cp "$TF_SITE_PACKAGES/libtensorflow_cc.so.2" "$TENSORFLOW_DIR/lib/"
cp "$TF_SITE_PACKAGES/libtensorflow_framework.so.2" "$TENSORFLOW_DIR/lib/"
ln -sf "libtensorflow_cc.so.2" "$TENSORFLOW_DIR/lib/libtensorflow_cc.so"
ln -sf "libtensorflow_framework.so.2" "$TENSORFLOW_DIR/lib/libtensorflow_framework.so"
cp -r "$TF_SITE_PACKAGES/include/"* "$TENSORFLOW_DIR/include/"

# Copy additional headers if available
[[ -d "$TF_SITE_PACKAGES/core" ]] && cp -r "$TF_SITE_PACKAGES/core" "$TENSORFLOW_DIR/include/tensorflow/"
[[ -d "$TF_SITE_PACKAGES/cc" ]] && cp -r "$TF_SITE_PACKAGES/cc" "$TENSORFLOW_DIR/include/tensorflow/"

# Create pkg-config file
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
env_file="$DEPENDENCIES_DIR/setup_env.sh"
grep -v "TensorFlow\|TENSORFLOW" "$env_file" > "${env_file}.tmp" 2>/dev/null || true
mv "${env_file}.tmp" "$env_file" 2>/dev/null || true
cat >> "$env_file" << EOF
export TENSORFLOW_DIR="$TENSORFLOW_DIR"
export LD_LIBRARY_PATH="\$TENSORFLOW_DIR/lib:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="\$TENSORFLOW_DIR/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
EOF

echo "Setup complete. Source $env_file to use TensorFlow."