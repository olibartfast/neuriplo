# TVM Backend Build Guide for Neuriplo

This guide provides comprehensive instructions for building and using the TVM backend with neuriplo.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building TVM from Source](#building-tvm-from-source)
- [Configuring TVM for Neuriplo](#configuring-tvm-for-neuriplo)
- [Building Neuriplo with TVM](#building-neuriplo-with-tvm)
- [Model Compilation](#model-compilation)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

TVM (Tensor Virtual Machine) is an open-source deep learning compiler stack that optimizes models for various hardware backends. The neuriplo TVM backend enables:

- **Hardware Optimization**: Automatic optimization for CPUs, GPUs, and accelerators
- **Model Flexibility**: Support for models from TensorFlow, PyTorch, ONNX, and more
- **Performance Tuning**: Auto-tuning for optimal performance on target hardware
- **Quantization**: INT8/FP16 quantization support
- **Cross-platform**: Works on x86, ARM, and embedded devices

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS, or Windows (WSL recommended)
- **CMake**: 3.18 or higher
- **C++ Compiler**: GCC 7.0+, Clang 5.0+, or MSVC 2019+
- **Python**: 3.7 or higher
- **LLVM**: 10.0+ (recommended for CPU optimization)

### Optional Dependencies

- **CUDA Toolkit**: 11.0+ (for NVIDIA GPU support)
- **ROCm**: 4.0+ (for AMD GPU support)
- **OpenCL**: 1.2+ (for GPU support)
- **Vulkan**: 1.2+ (for Vulkan backend)
- **cuDNN**: 8.0+ (for GPU optimizations)

### Install System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    llvm-12-dev \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libtinfo-dev \
    zlib1g-dev \
    libedit-dev \
    libxml2-dev
```

#### macOS

```bash
brew install llvm cmake python@3.9
```

#### Windows (WSL2)

Use WSL2 with Ubuntu and follow Ubuntu instructions above.

## Building TVM from Source

### Step 1: Clone TVM Repository

```bash
cd $HOME/dependencies
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
```

**Note**: The `--recursive` flag is important to clone submodules.

### Step 2: Configure Build Options

Copy and edit the configuration file:

```bash
mkdir build
cp cmake/config.cmake build/
cd build
```

Edit `build/config.cmake` to enable desired features:

```cmake
# Basic Configuration
set(USE_LLVM ON)          # Required for CPU optimization
set(USE_CUDA OFF)         # Enable for NVIDIA GPU support
set(USE_OPENCL OFF)       # Enable for OpenCL support
set(USE_VULKAN OFF)       # Enable for Vulkan support
set(USE_METAL OFF)        # Enable for Apple Metal (macOS/iOS)
set(USE_ROCM OFF)         # Enable for AMD GPU support

# CUDA Configuration (if USE_CUDA is ON)
# set(USE_CUDA /usr/local/cuda)
# set(USE_CUDNN ON)
# set(USE_CUBLAS ON)
# set(USE_THRUST ON)

# LLVM Configuration
set(USE_LLVM /usr/lib/llvm-12/bin/llvm-config)  # Adjust path as needed

# Relay and Runtime
set(USE_RELAY_DEBUG ON)   # Enable for debugging
set(USE_RTTI ON)          # Enable RTTI for C++ interop

# Additional Libraries
set(USE_BLAS openblas)    # Use OpenBLAS for linear algebra
set(USE_RANDOM ON)        # Enable random number generation
set(USE_GRAPH_EXECUTOR ON) # Required for graph runtime
set(USE_GRAPH_EXECUTOR_DEBUG ON) # Enable for debugging

# Build shared library
set(BUILD_SHARED_LIBS ON)
```

### Step 3: Build TVM

```bash
# From tvm/build directory
cmake ..
make -j$(nproc)
```

Build time: 10-30 minutes depending on your system.

### Step 4: Install TVM Python Package

```bash
cd ../python
pip install -e .
```

Verify installation:

```bash
python -c "import tvm; print(tvm.__version__)"
```

### Step 5: Install Additional Python Dependencies

```bash
pip install numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle
```

## Configuring TVM for Neuriplo

### Set TVM Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export TVM_HOME=$HOME/dependencies/tvm
export TVM_DIR=$TVM_HOME
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export LD_LIBRARY_PATH=$TVM_HOME/build:${LD_LIBRARY_PATH}
```

Apply changes:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Verify TVM Installation

```bash
# Check library exists
ls -lh $TVM_HOME/build/libtvm.so  # Linux
ls -lh $TVM_HOME/build/libtvm.dylib  # macOS

# Check include files
ls $TVM_HOME/include/tvm/runtime/
```

## Building Neuriplo with TVM

### Configure Neuriplo Build

```bash
cd /path/to/neuriplo
mkdir build
cd build

cmake .. \
    -DDEFAULT_BACKEND=TVM \
    -DTVM_DIR=$HOME/dependencies/tvm \
    -DCMAKE_BUILD_TYPE=Release
```

### Build

```bash
cmake --build . -j$(nproc)
```

### Build with Tests

```bash
cmake .. \
    -DDEFAULT_BACKEND=TVM \
    -DTVM_DIR=$HOME/dependencies/tvm \
    -DBUILD_INFERENCE_ENGINE_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)

# Run tests
ctest --verbose
```

## Model Compilation

TVM requires models to be compiled before inference. Here's how to prepare models:

### From ONNX Model

```python
import tvm
from tvm import relay
import onnx

# Load ONNX model
onnx_model = onnx.load("model.onnx")

# Convert to Relay IR
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Compile for target
target = "llvm"  # CPU
# target = "cuda"  # NVIDIA GPU
# target = "opencl"  # OpenCL

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Save compiled model
lib.export_library("model.so")
```

### From PyTorch Model

```python
import torch
import tvm
from tvm import relay

# Load PyTorch model
model = torch.load("model.pth")
model.eval()

# Create example input
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)

# Trace the model
traced_model = torch.jit.trace(model, input_data)

# Convert to Relay IR
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

# Compile
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Save
lib.export_library("model.so")
```

### From TensorFlow Model

```python
import tensorflow as tf
import tvm
from tvm import relay

# Load TensorFlow model
model = tf.keras.models.load_model("model.h5")

# Convert to Relay
mod, params = relay.frontend.from_keras(model, shape_dict)

# Compile
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Save
lib.export_library("model.so")
```

## Performance Optimization

### Auto-Tuning

TVM's auto-tuning can significantly improve performance:

```python
import tvm
from tvm import autotvm, relay
from tvm.autotvm.tuner import XGBTuner

# Load model
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Extract tuning tasks
target = "llvm -mcpu=core-avx2"
tasks = autotvm.task.extract_from_program(
    mod["main"], target=target, params=params
)

# Configure tuning
tuning_option = {
    'log_filename': 'tuning.log',
    'tuner': 'xgb',
    'n_trial': 1000,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4)
    ),
}

# Run tuning
for i, task in enumerate(tasks):
    prefix = f"[Task {i+1}/{len(tasks)}] "
    tuner = XGBTuner(task, loss_type='rank')
    tuner.tune(
        n_trial=min(tuning_option['n_trial'], len(task.config_space)),
        early_stopping=tuning_option['early_stopping'],
        measure_option=tuning_option['measure_option'],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option['n_trial'], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option['log_filename'])
        ]
    )

# Compile with tuning log
with autotvm.apply_history_best(tuning_option['log_filename']):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
```

### Optimization Levels

```python
# opt_level=0: No optimization
# opt_level=1: Basic optimizations
# opt_level=2: Standard optimizations (default)
# opt_level=3: Aggressive optimizations

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
```

### Target-Specific Optimization

```python
# CPU with specific features
target = "llvm -mcpu=core-avx2 -mattr=+avx,+avx2,+fma"

# NVIDIA GPU
target = "cuda -arch=sm_75"  # For RTX 2080, Titan RTX
target = "cuda -arch=sm_80"  # For A100
target = "cuda -arch=sm_86"  # For RTX 3090

# ARM CPU
target = "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72"
```

## Troubleshooting

### Issue: TVM library not found

**Error**: `libtvm.so: cannot open shared object file`

**Solution**:
```bash
export LD_LIBRARY_PATH=$TVM_HOME/build:${LD_LIBRARY_PATH}
# Add to ~/.bashrc for persistence
```

### Issue: LLVM not found during build

**Error**: `Could NOT find LLVM`

**Solution**:
```bash
# Install LLVM
sudo apt-get install llvm-12-dev

# Update config.cmake
set(USE_LLVM /usr/lib/llvm-12/bin/llvm-config)
```

### Issue: CUDA compilation errors

**Error**: `nvcc fatal : Unsupported gpu architecture`

**Solution**:
```cmake
# In config.cmake, specify your GPU architecture
set(USE_CUDA /usr/local/cuda)
# Check your GPU compute capability: https://developer.nvidia.com/cuda-gpus
```

### Issue: Python import fails

**Error**: `ImportError: No module named 'tvm'`

**Solution**:
```bash
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
# Or reinstall Python package
cd $TVM_HOME/python
pip install -e .
```

### Issue: Compilation very slow

**Solution**:
```bash
# Use more cores
make -j$(nproc)

# Or limit cores if running out of memory
make -j4
```

### Issue: Runtime errors with compiled models

**Solution**:
```python
# Enable debug mode in config.cmake
set(USE_RELAY_DEBUG ON)
set(USE_GRAPH_EXECUTOR_DEBUG ON)

# Rebuild TVM
cd $TVM_HOME/build
make -j$(nproc)

# Check model compilation
import tvm.relay.testing
tvm.relay.testing.check_infer_type(mod)
```

## Advanced Features

### Quantization

```python
from tvm.relay import quantize

# Post-training quantization
with relay.quantize.qconfig(calibrate_mode='kl_divergence', weight_scale='max'):
    mod = relay.quantize.quantize(mod, params)
```

### Mixed Precision

```python
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end

# Convert specific ops to FP16
mod = relay.transform.ToMixedPrecision()(mod)
```

### Multi-Threading

```bash
# Set number of threads for TVM runtime
export TVM_NUM_THREADS=8
```

## Resources

- **Official TVM Documentation**: https://tvm.apache.org/docs/
- **TVM Installation Guide**: https://tvm.apache.org/docs/install/from_source.html
- **TVM Tutorials**: https://tvm.apache.org/docs/tutorials/
- **TVM Forum**: https://discuss.tvm.apache.org/
- **TVM GitHub**: https://github.com/apache/tvm
- **Auto-tuning Guide**: https://tvm.apache.org/docs/how_to/tune_with_autotvm/

## Performance Comparison

Expected performance improvements with TVM:

| Backend | Relative Performance | Use Case |
|---------|---------------------|----------|
| CPU (no optimization) | 1.0x | Baseline |
| TVM (opt_level=3) | 2-4x | CPU inference |
| TVM + Auto-tuning | 3-6x | Optimized CPU |
| TVM CUDA | 10-50x | GPU inference |
| TVM CUDA + Auto-tuning | 20-100x | Optimized GPU |

*Note: Actual performance depends on model architecture and hardware.*

## Next Steps

1. Build TVM following this guide
2. Compile a test model (ONNX, PyTorch, or TensorFlow)
3. Build neuriplo with TVM backend
4. Run inference tests
5. Optimize with auto-tuning for your target hardware
6. Deploy optimized models in your application

For issues or questions, refer to the [main documentation](DEPENDENCY_MANAGEMENT.md) or open an issue on GitHub.
