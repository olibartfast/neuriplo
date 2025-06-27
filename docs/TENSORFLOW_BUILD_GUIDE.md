# TensorFlow Build and Installation Guide

This guide provides step-by-step instructions for building TensorFlow 2.19.0 from source on Ubuntu Linux systems.

## Prerequisites

- Ubuntu 22.04 or later
- At least 16GB RAM (32GB recommended)
- At least 50GB free disk space
- Python 3.12
- Git

## Step 1: System Preparation

### Update system packages
```bash
sudo apt update && sudo apt upgrade -y
```

### Install basic build dependencies
```bash
sudo apt install -y build-essential git curl wget
```

## Step 2: Set Up Python Virtual Environment

### Create and activate virtual environment
```bash
python3 -m venv tensorflow_build_env
source tensorflow_build_env/bin/activate
```

### Upgrade pip
```bash
pip install -U pip
```

## Step 3: Install Bazel Build System

### Add Bazel repository
```bash
# Download and add GPG key
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/keyrings/

# Add repository
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/bazel.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
```

### Install Bazel 6.5.0 (required for TensorFlow 2.19)
```bash
sudo apt update
sudo apt install -y bazel-6.5.0
```

### Verify installation
```bash
bazel-6.5.0 --version
# Expected output: bazel 6.5.0
```

## Step 4: Install Clang Compiler

### Install Clang 17 (recommended for TensorFlow 2.19)
```bash
sudo apt install -y llvm-17 clang-17
```

### Verify installation
```bash
clang-17 --version
# Expected output: Ubuntu clang version 17.0.6
```

## Step 5: Clone TensorFlow Repository

### Clone the repository
```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

### Checkout specific version
```bash
git checkout r2.19
```

## Step 6: Configure Build

### Set environment variables
```bash
export CC=/usr/bin/clang-17
export BAZEL_COMPILER=/usr/bin/clang-17
```

### Run configuration script
```bash
./configure
```

**Configuration options:**
- Python path: Use default (virtual environment)
- ROCm support: N (No)
- CUDA support: N (No) - for CPU-only build
- Use Clang: Y (Yes)
- Clang path: Use default (/usr/lib/llvm-17/bin/clang)
- Optimization flags: Use default (-Wno-sign-compare)
- Android builds: N (No)

## Step 7: Build TensorFlow

### Build CPU-only package
```bash
bazel-6.5.0 build //tensorflow/tools/pip_package:wheel \
    --repo_env=USE_PYWRAP_RULES=1 \
    --repo_env=WHEEL_NAME=tensorflow_cpu \
    --config=opt
```

**Build time:** Approximately 2-4 hours depending on system specifications.

**Expected output location:**
```
bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl
```

## Step 8: Install TensorFlow

### Install the built package
```bash
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl
```

### Verify installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Step 9: Build TensorFlow C++ Libraries (Optional)

If you need the C++ libraries for your inference engine:

### Build C++ libraries
```bash
bazel-6.5.0 build //tensorflow:libtensorflow_cc.so \
    --config=opt \
    --cxxopt=-std=c++17
```

### Copy libraries to system location
```bash
sudo cp bazel-bin/tensorflow/libtensorflow_cc.so.2 /usr/local/lib/
sudo cp bazel-bin/tensorflow/libtensorflow_framework.so.2 /usr/local/lib/
sudo ldconfig
```

### Copy headers
```bash
sudo cp -r tensorflow /usr/local/include/
sudo cp -r third_party /usr/local/include/
```

## Troubleshooting

### Common Issues

1. **Out of memory during build**
   ```bash
   # Limit Bazel memory usage
   bazel-6.5.0 build --local_ram_resources=2048 //tensorflow/tools/pip_package:wheel
   ```

2. **Build fails with compiler errors**
   - Ensure Clang 17 is properly installed
   - Check environment variables are set correctly
   - Verify Bazel version is 6.5.0

3. **Python version mismatch**
   - Ensure Python 3.12 is used
   - Check virtual environment is activated

### Performance Optimization

For faster builds:
```bash
# Use more CPU cores
bazel-6.5.0 build --jobs=8 //tensorflow/tools/pip_package:wheel

# Enable build cache
bazel-6.5.0 build --disk_cache=/path/to/cache //tensorflow/tools/pip_package:wheel
```

## GPU Support (Optional)

To build with GPU support:

1. Install CUDA and cuDNN
2. Configure with CUDA support: `y` when prompted
3. Build with GPU flags:
   ```bash
   bazel-6.5.0 build //tensorflow/tools/pip_package:wheel \
       --repo_env=USE_PYWRAP_RULES=1 \
       --repo_env=WHEEL_NAME=tensorflow \
       --config=cuda \
       --config=cuda_wheel \
       --config=opt
   ```

## Cleanup

### Remove build artifacts
```bash
bazel-6.5.0 clean --expunge
```

### Remove virtual environment
```bash
deactivate
rm -rf tensorflow_build_env
```

## Version Compatibility

| TensorFlow Version | Python | Bazel | Clang | CUDA | cuDNN |
|-------------------|--------|-------|-------|------|-------|
| 2.19.0            | 3.9-3.12 | 6.5.0 | 17.0.6 | 12.5 | 9.3 |
| 2.18.0            | 3.9-3.12 | 6.5.0 | 17.0.6 | 12.5 | 9.3 |
| 2.17.0            | 3.9-3.12 | 6.5.0 | 17.0.6 | 12.3 | 8.9 |

## References

- [Official TensorFlow Build Guide](https://www.tensorflow.org/install/source)
- [Bazel Installation Guide](https://bazel.build/install)
- [Clang Documentation](https://clang.llvm.org/)

## Notes

- Building TensorFlow from source is resource-intensive
- Consider using pre-built packages for production environments
- The build process may take several hours on slower systems
- Ensure adequate disk space for build artifacts and cache 