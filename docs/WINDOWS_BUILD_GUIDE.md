# Windows Build Guide for Neuriplo

This guide provides detailed instructions for building and using neuriplo on Windows.

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Dependencies](#system-dependencies)
- [Backend Setup](#backend-setup)
- [Building Neuriplo](#building-neuriplo)
- [Common Issues](#common-issues)

## Prerequisites

### Required Software

1. **Windows 10/11 (64-bit)**
   - Recommended: Windows 10 version 1809 or later

2. **Visual Studio 2019 or later**
   - Download from: https://visualstudio.microsoft.com/
   - Required components:
     - Desktop development with C++
     - C++ CMake tools for Windows
     - Windows 10 SDK

3. **CMake 3.10 or higher**
   - Download from: https://cmake.org/download/
   - Add to PATH during installation

4. **Git**
   - Download from: https://git-scm.com/download/win
   - Use Git Bash or Git for Windows

5. **vcpkg (Recommended)**
   - Package manager for C++ libraries
   - Installation:
     ```powershell
     git clone https://github.com/Microsoft/vcpkg.git
     cd vcpkg
     .\bootstrap-vcpkg.bat
     .\vcpkg integrate install
     ```

### Optional (for GPU Support)

- **NVIDIA CUDA Toolkit 12.6**
  - Download from: https://developer.nvidia.com/cuda-downloads
  - Required for: TensorRT, ONNX Runtime GPU, GGML GPU

- **cuDNN**
  - Download from: https://developer.nvidia.com/cudnn
  - Required for: TensorRT, LibTorch GPU

## System Dependencies

### Install OpenCV and glog via vcpkg

```powershell
# OpenCV with DNN and contrib modules
vcpkg install opencv[contrib,dnn]:x64-windows

# glog for logging
vcpkg install glog:x64-windows
```

### Configure vcpkg with CMake

```powershell
# Set vcpkg toolchain file
$env:CMAKE_TOOLCHAIN_FILE = "C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake"

# Or use it directly in CMake command
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake
```

## Backend Setup

### ONNX Runtime

ONNX Runtime provides the easiest setup on Windows with pre-built binaries.

```powershell
# Automatic setup
.\scripts\setup_dependencies.ps1 -Backend ONNX_RUNTIME

# Manual setup
# 1. Download from: https://github.com/microsoft/onnxruntime/releases
# 2. Download: onnxruntime-win-x64-gpu-{VERSION}.zip
# 3. Extract to: $env:USERPROFILE\dependencies\
```

**Features:**
- ✅ Pre-built binaries available
- ✅ GPU support (CUDA)
- ✅ CPU optimizations
- ✅ Easy installation

### LibTorch (PyTorch)

LibTorch provides official Windows builds with CUDA support.

```powershell
# Automatic setup (downloads with CUDA support)
.\scripts\setup_dependencies.ps1 -Backend LIBTORCH

# Manual setup
# 1. Download from: https://pytorch.org/get-started/locally/
# 2. Select: Windows, LibTorch, C++/Java, CUDA or CPU
# 3. Extract to: $env:USERPROFILE\dependencies\libtorch
```

**Features:**
- ✅ Official Windows builds
- ✅ GPU support (CUDA)
- ✅ Extensive model support
- ⚠️ Large download size (~2GB)

### TensorRT

TensorRT requires manual download from NVIDIA Developer.

```powershell
# Manual setup required
# 1. Register at: https://developer.nvidia.com/tensorrt
# 2. Download: TensorRT for Windows
# 3. Extract to: $env:USERPROFILE\dependencies\TensorRT-{VERSION}
# 4. Add to PATH: TensorRT\lib
```

**Requirements:**
- CUDA Toolkit 12.6
- cuDNN 8.x
- NVIDIA GPU (Compute Capability 5.0+)

**Features:**
- ✅ Best GPU performance
- ✅ INT8/FP16 optimization
- ❌ Manual download required
- ❌ NVIDIA hardware only

### OpenVINO

OpenVINO provides optimized inference for Intel hardware.

```powershell
# Manual setup
# 1. Download from: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
# 2. Install using the installer or extract archive
# 3. Run: setupvars.bat from the installation directory
```

**Features:**
- ✅ Optimized for Intel CPUs
- ✅ Support for Intel GPUs
- ✅ Model optimization toolkit
- ⚠️ Complex setup

### GGML

GGML requires building from source.

```powershell
# Automatic setup (clones and provides build instructions)
.\scripts\setup_dependencies.ps1 -Backend GGML

# Build GGML
cd $env:USERPROFILE\dependencies\ggml
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Features:**
- ✅ Lightweight
- ✅ CPU optimizations
- ✅ Quantization support
- ⚠️ Requires building from source

### OpenCV DNN

OpenCV DNN is included with OpenCV installation.

```powershell
# Already installed via vcpkg
vcpkg install opencv[contrib,dnn]:x64-windows
```

**Features:**
- ✅ Easy to use
- ✅ No additional dependencies
- ⚠️ Limited performance compared to dedicated backends

## Building Neuriplo

### Standard Build Process

```powershell
# 1. Clone the repository
git clone https://github.com/olibartfast/neuriplo.git
cd neuriplo

# 2. Setup dependencies for your chosen backend
.\scripts\setup_dependencies.ps1 -Backend ONNX_RUNTIME

# 3. Source environment variables
. $env:USERPROFILE\dependencies\setup_neuriplo_env.ps1

# 4. Configure with CMake
cmake -B build -DDEFAULT_BACKEND=ONNX_RUNTIME -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake

# 5. Build
cmake --build build --config Release

# 6. (Optional) Install
cmake --install build --prefix C:\neuriplo
```

### Build with Different Backends

```powershell
# ONNX Runtime
cmake -B build -DDEFAULT_BACKEND=ONNX_RUNTIME
cmake --build build --config Release

# LibTorch
cmake -B build -DDEFAULT_BACKEND=LIBTORCH
cmake --build build --config Release

# TensorRT
cmake -B build -DDEFAULT_BACKEND=TENSORRT
cmake --build build --config Release

# OpenVINO
cmake -B build -DDEFAULT_BACKEND=OPENVINO
cmake --build build --config Release

# GGML
cmake -B build -DDEFAULT_BACKEND=GGML
cmake --build build --config Release

# OpenCV DNN
cmake -B build -DDEFAULT_BACKEND=OPENCV_DNN
cmake --build build --config Release
```

### Build with Tests

```powershell
cmake -B build -DDEFAULT_BACKEND=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build --config Release

# Run tests
cd build
ctest -C Release
```

## Common Issues

### Issue: CMake cannot find OpenCV

**Solution:**
```powershell
# Ensure vcpkg toolchain is specified
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake

# Or set environment variable
$env:CMAKE_TOOLCHAIN_FILE = "C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake"
```

### Issue: CUDA/cuDNN not found

**Solution:**
```powershell
# Set CUDA path
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

# Verify installation
nvcc --version
```

### Issue: DLL not found when running

**Solution:**
```powershell
# Add dependencies to PATH
$env:Path = "$env:USERPROFILE\dependencies\onnxruntime-win-x64-gpu-1.19.2\lib;$env:Path"
$env:Path = "$env:USERPROFILE\dependencies\libtorch\lib;$env:Path"

# Or copy DLLs to executable directory
```

### Issue: Visual Studio version mismatch

**Solution:**
```powershell
# Specify Visual Studio version
cmake -B build -G "Visual Studio 16 2019" -A x64
# or
cmake -B build -G "Visual Studio 17 2022" -A x64
```

### Issue: Out of memory during build

**Solution:**
```powershell
# Limit parallel build jobs
cmake --build build --config Release -- /m:2

# Or use MSBuild directly
msbuild build\neuriplo.sln /p:Configuration=Release /m:2
```

### Issue: glog link errors

**Solution:**
```powershell
# Ensure glog is installed via vcpkg
vcpkg install glog:x64-windows

# Verify installation
vcpkg list | Select-String "glog"
```

## Performance Optimization

### Release Build

Always use Release configuration for production:
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Compiler Optimizations

The build system automatically applies Windows-specific optimizations:
- `/O2` - Maximum optimization
- `/fp:fast` - Fast floating-point model
- `/arch:AVX2` - AVX2 instructions (if supported)

### CUDA Optimization

For GPU backends, ensure proper CUDA architecture:
```powershell
# Set compute capability
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
```

## Integration with Your Project

### CMake Integration

```cmake
# Find neuriplo
find_package(neuriplo REQUIRED)

# Link your project
target_link_libraries(your_project PRIVATE neuriplo)
```

### Manual Integration

```cmake
# Include directories
target_include_directories(your_project PRIVATE C:/neuriplo/include)

# Link libraries
target_link_libraries(your_project PRIVATE 
    C:/neuriplo/lib/neuriplo.lib
    # Add backend-specific libraries
)
```

## Additional Resources

- **Main Documentation**: [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)
- **neuriplo README**: [README.md](../Readme.md)
- **vcpkg Documentation**: https://vcpkg.io/
- **CMake Documentation**: https://cmake.org/documentation/
- **Visual Studio**: https://docs.microsoft.com/en-us/visualstudio/

## Support

For issues specific to Windows builds:
1. Check this guide first
2. Review CMake error messages carefully
3. Ensure all prerequisites are installed
4. Check PATH environment variables
5. Open an issue on GitHub with detailed error information
