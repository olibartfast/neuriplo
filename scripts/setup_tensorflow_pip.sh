#!/bin/bash

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Load versions from versions.env (the single source of truth). cmake/versions.cmake
# only re-exports these as cache variables, so it cannot be scraped for a literal.
if [ -f "$PROJECT_ROOT/versions.env" ]; then
    source "$PROJECT_ROOT/versions.env"
else
    echo "Error: versions.env file not found" >&2
    exit 1
fi
DEPENDENCIES_DIR="${DEPENDENCIES_DIR:-$HOME/dependencies}"
TENSORFLOW_DIR="$DEPENDENCIES_DIR/tensorflow"
VENV_DIR="$DEPENDENCIES_DIR/tensorflow_env"

# Parse arguments
CLEAN_INSTALL=false
FORCE=false
GPU_CHECK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version) TENSORFLOW_VERSION="$2"; shift 2 ;;
        -c|--clean) CLEAN_INSTALL=true; shift ;;
        -f|--force) FORCE=true; shift ;;
        -g|--gpu) GPU_CHECK=true; shift ;;
        -h|--help) 
            echo "Usage: $0 [-v VERSION] [-c] [-f] [-g] [-h]"
            echo "Options:"
            echo "  -v, --version VERSION  TensorFlow version (default: $TENSORFLOW_VERSION)"
            echo "  -c, --clean            Clean install (remove existing installation)"
            echo "  -f, --force            Force reinstall even if already installed"
            echo "  -g, --gpu              Validate GPU runtime prerequisites (driver, CUDA, cuDNN)"
            echo "  -h, --help             Show this help message"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check Python
command -v python3 &>/dev/null || { echo "Python 3 required"; exit 1; }

# Opt-in GPU prerequisite validation. The pip wheel carries GPU kernels, but it
# needs the host NVIDIA driver and a CUDA 12.x / cuDNN 9.x runtime at run time.
# The default (CPU) path is unaffected: nothing is checked unless --gpu is set.
if [[ "$GPU_CHECK" == "true" ]]; then
    echo "Validating TensorFlow GPU runtime prerequisites..."
    command -v nvidia-smi &>/dev/null || {
        echo "Error: nvidia-smi not found; install the NVIDIA driver or use the GPU container with --gpus all" >&2
        exit 1
    }
    nvidia-smi >/dev/null 2>&1 || {
        echo "Error: nvidia-smi failed; the NVIDIA driver is not usable on this host" >&2
        exit 1
    }
    for lib in 'libcudart\.so\.12' 'libcublas\.so\.12' 'libcudnn\.so\.9'; do
        # The runtime may come from the system loader (ldconfig) or from pip's
        # nvidia-*-cu12 wheels, which install into the venv's site-packages and
        # are not registered with ldconfig. Accept either source.
        if ldconfig -p 2>/dev/null | grep -E "$lib" >/dev/null || \
           find "$VENV_DIR" -path '*/nvidia/*/lib/*' 2>/dev/null | grep -E "/${lib}\$" >/dev/null; then
            continue
        fi
        echo "Error: required GPU runtime library not found: $lib (TensorFlow 2.x needs CUDA 12.x and cuDNN 9.x)" >&2
        echo "Use the pinned image (nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04) or install the matching CUDA/cuDNN libraries." >&2
        exit 1
    done
    echo "GPU runtime prerequisites found (NVIDIA driver, CUDA 12.x, cuDNN 9.x)."
    echo "Note: device visibility is authoritative from TensorFlow Session::ListDevices();"
    echo "      a container still needs --gpus all and a host driver."
fi

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