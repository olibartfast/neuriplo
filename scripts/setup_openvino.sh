#!/bin/bash

# Setup script for OpenVINO backend
# This script is called by the unified setup_dependencies.sh

set -e

echo "Setting up OpenVINO backend..."

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

# Default installation directory
version="$OPENVINO_VERSION"
dir="$DEPENDENCY_ROOT/openvino_$version"

# Check if already installed
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    echo "✓ OpenVINO already installed at $dir"
    exit 0
fi

echo "Installing OpenVINO $version to $dir..."
mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"

# Download OpenVINO toolkit
tarball="openvino_${version}.tgz"
if [[ ! -f "$tarball" ]]; then
    echo "Downloading OpenVINO toolkit..."
    curl -L "https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/linux/openvino_toolkit_ubuntu24_${version}.19140.c01cd93e24d_x86_64.tgz" --output "$tarball"
fi

# Extract and move to final location
echo "Extracting OpenVINO..."
tar -xf "$tarball"
if [[ -d "$dir" ]]; then
    rm -rf "$dir"
fi
mv openvino_toolkit_ubuntu24_${version}.19140.c01cd93e24d_x86_64 "$dir"
rm -f "$tarball"

# Create a local Python virtual environment for OpenVINO tools
echo "Setting up OpenVINO Python tools..."
venv_dir="$dir/python_env"
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

echo "✓ OpenVINO $version installed successfully to $dir"
