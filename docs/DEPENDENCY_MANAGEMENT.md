# Dependency Management for neuriplo

This document describes the dependency management system for the neuriplo library, which provides a unified approach to managing inference backend dependencies.

## Architecture

### Version Management

All inference backend versions are centrally managed in `cmake/versions.cmake`:

```cmake
# Inference Backend Versions
set(ONNX_RUNTIME_VERSION "1.19.2" CACHE STRING "ONNX Runtime version")
set(TENSORRT_VERSION "10.7.0.23" CACHE STRING "TensorRT version")
set(LIBTORCH_VERSION "2.0.0" CACHE STRING "LibTorch version")
set(OPENVINO_VERSION "2023.1.0" CACHE STRING "OpenVINO version")
set(TENSORFLOW_VERSION "2.19.0" CACHE STRING "TensorFlow version")

# CUDA Version (for GPU support)
set(CUDA_VERSION "12.6" CACHE STRING "CUDA version for GPU support")

# System Dependencies (minimum versions)
set(OPENCV_MIN_VERSION "4.6.0" CACHE STRING "Minimum OpenCV version")
set(GLOG_MIN_VERSION "0.6.0" CACHE STRING "Minimum glog version")
set(CMAKE_MIN_VERSION "3.10" CACHE STRING "Minimum CMake version")
```

### Dependency Validation

The `cmake/DependencyValidation.cmake` module provides validation:

- **System Dependencies**: OpenCV, glog, CMake version
- **Inference Backends**: ONNX Runtime, TensorRT, LibTorch, OpenVINO, TensorFlow
- **GPU Support**: CUDA validation for GPU-enabled backends
- **Installation Completeness**: Checks for required files and libraries

### Setup Scripts

#### Unified Setup Script

The main setup script `scripts/setup_dependencies.sh` supports the following inference backends:

```bash
# Setup any backend
./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
```

**Note**: TensorFlow (LIBTENSORFLOW) and OpenCV DNN (OPENCV_DNN) backends are supported by the CMake build system but require separate setup procedures (see individual backend scripts below).

#### Individual Backend Scripts

All backends can be set up using the unified script:

```bash
./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
```

Supported backend values: `OPENCV_DNN`, `ONNX_RUNTIME`, `LIBTORCH`, `TENSORRT`, `LIBTENSORFLOW`, `OPENVINO`, `GGML`, `TVM`

## Usage

### Building with CMake

The CMakeLists.txt automatically includes version management and validation:

```cmake
# Include centralized version management first
include(cmake/versions.cmake)

# Include dependency validation
include(cmake/DependencyValidation.cmake)

# Validate dependencies before proceeding
validate_all_dependencies()
```

### Setting Up Dependencies

1. **Choose your backend**:
   ```bash
   # Setup any backend
   ./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
   
   # Examples:
   ./scripts/setup_dependencies.sh --backend ONNX_RUNTIME
   ./scripts/setup_dependencies.sh --backend TENSORRT
   ./scripts/setup_dependencies.sh --backend LIBTORCH
   ./scripts/setup_dependencies.sh --backend OPENVINO
   ./scripts/setup_dependencies.sh --backend GGML
   ...
   ```

2. **Set environment variables**:
   ```bash
   source $HOME/dependencies/setup_env.sh
   ```

3. **Build the library**:
   ```bash
   mkdir build && cd build
   
   # Build with any backend
   cmake .. -DDEFAULT_BACKEND=<BACKEND_NAME> -DBUILD_INFERENCE_ENGINE_TESTS=ON
   make
   
   # Examples:
   cmake .. -DDEFAULT_BACKEND=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON
   cmake .. -DDEFAULT_BACKEND=TENSORRT -DBUILD_INFERENCE_ENGINE_TESTS=ON
   cmake .. -DDEFAULT_BACKEND=LIBTORCH -DBUILD_INFERENCE_ENGINE_TESTS=ON
   cmake .. -DDEFAULT_BACKEND=OPENVINO -DBUILD_INFERENCE_ENGINE_TESTS=ON
   cmake .. -DDEFAULT_BACKEND=GGML -DBUILD_INFERENCE_ENGINE_TESTS=ON
   cmake .. -DDEFAULT_BACKEND=TVM -DBUILD_INFERENCE_ENGINE_TESTS=ON
   ```

### Configuration Options

#### CMake Variables

- `DEFAULT_BACKEND`: Choose the inference backend (ONNX_RUNTIME, TENSORRT, LIBTORCH, OPENVINO, LIBTENSORFLOW, OPENCV_DNN, GGML, TVM)
- `BUILD_INFERENCE_ENGINE_TESTS`: Enable/disable test building (ON/OFF)
- `DEPENDENCY_ROOT`: Set custom dependency installation root (default: `$HOME/dependencies`)
- `ONNX_RUNTIME_VERSION`: Override ONNX Runtime version
- `TENSORRT_VERSION`: Override TensorRT version
- `LIBTORCH_VERSION`: Override LibTorch version
- `OPENVINO_VERSION`: Override OpenVINO version
- `TENSORFLOW_VERSION`: Override TensorFlow version
- `TVM_VERSION`: Override TVM version
- `CUDA_VERSION`: Override CUDA version for GPU support

#### Environment Variables

The setup script creates environment variables for each backend:

```bash
export DEPENDENCY_ROOT="$HOME/dependencies"
export ONNX_RUNTIME_DIR="$HOME/dependencies/onnxruntime-linux-x64-gpu-1.19.2"
export TENSORRT_DIR="$HOME/dependencies/TensorRT-10.7.0.23"
export LIBTORCH_DIR="$HOME/dependencies/libtorch"
export OPENVINO_DIR="$HOME/dependencies/openvino-2023.1.0"
export LD_LIBRARY_PATH="$ONNX_RUNTIME_DIR/lib:$TENSORRT_DIR/lib:$LIBTORCH_DIR/lib:$OPENVINO_DIR/lib:$LD_LIBRARY_PATH"
```

**Note**: TensorFlow environment variables are set by the individual TensorFlow setup scripts.

## Supported Platforms

### Linux (Ubuntu/Debian)
- Full support for all inference backends
- Automatic system dependency installation via apt-get
- OpenCV and glog installation
- TensorFlow C++ library support

### Linux (CentOS/RHEL/Fedora)
- Basic support with manual dependency installation
- Uses yum/dnf package manager
- May require additional configuration

### Other Linux Distributions
- Manual dependency installation required
- Refer to individual setup scripts for guidance

### Windows
- Not currently supported
- Future development planned

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   # Reinstall with force flag for unified script
   ./scripts/setup_dependencies.sh --backend <BACKEND_NAME> --force
   
   # Example:
   ./scripts/setup_dependencies.sh --backend ONNX_RUNTIME --force
   ```

2. **Version Conflicts**:
   ```bash
   # Override version in CMake
   cmake .. -DTENSORFLOW_VERSION=2.18.0
   ```

3. **CUDA Issues**:
   ```bash
   # Check CUDA installation
   nvcc --version
   nvidia-smi
   ```

4. **TensorFlow C++ Library Issues**:
   ```bash
   # Check TensorFlow installation
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

### Validation Errors

The validation system provides detailed error messages:

```
[ERROR] TensorFlow not found at /home/user/dependencies/tensorflow
Please ensure the inference backend is properly installed or run the setup script.
```

### Manual Installation

For backends requiring manual installation:

**TensorRT**:
1. Download from [NVIDIA Developer](https://developer.nvidia.com/tensorrt)
2. Extract to `$HOME/dependencies/TensorRT-<VERSION>`
3. Ensure CUDA is installed
4. Run validation: `./scripts/setup_dependencies.sh --backend TENSORRT`

**OpenVINO**:
1. Download from [Intel Developer Zone](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
2. Extract to `$HOME/dependencies/openvino-<VERSION>`
3. Run validation: `./scripts/setup_dependencies.sh --backend OPENVINO`

**TensorFlow**:
1. Install via pip: `pip install tensorflow==2.19.0`
2. Run setup script: `./scripts/setup_libtensorflow.sh`
3. Alternative: Use `./scripts/setup_tensorflow_pip.sh` for automated pip installation

**TVM**:
1. Clone TVM repository: `git clone --recursive https://github.com/apache/tvm tvm`
2. Build from source:
   ```bash
   cd tvm
   mkdir build
   cp cmake/config.cmake build/
   cd build
   # Edit config.cmake to enable desired features (LLVM, CUDA, etc.)
   cmake ..
   make -j$(nproc)
   ```
3. Install Python package:
   ```bash
   cd ../python
   pip install -e .
   ```
4. Set TVM directory: `export TVM_DIR=$HOME/dependencies/tvm`
5. For detailed instructions, see [TVM Installation Guide](https://tvm.apache.org/docs/install/from_source.html)

## Testing Integration

### Automated Testing

The testing framework supports all backends:

```bash
# Test specific backend
./scripts/test_backends.sh --backend <BACKEND_NAME>

# Examples:
./scripts/test_backends.sh --backend ONNX_RUNTIME
./scripts/test_backends.sh --backend TENSORRT
./scripts/test_backends.sh --backend LIBTORCH

# Test all backends
./scripts/test_backends.sh

# Run complete test suite with analysis
./scripts/run_complete_tests.sh

# Validate test system setup
./scripts/validate_test_system.sh
```

### Model Generation and Testing

- **TensorFlow**: Tests automatically generate SavedModel during execution using ResNet-50 from Keras Applications
- **ONNX Runtime**: Uses pre-downloaded ONNX models via `scripts/model_downloader.py`
- **LibTorch**: Tests with TorchScript models
- **TensorRT**: Requires model conversion from ONNX or other formats
- **OpenVINO**: Uses Intel OpenVINO IR format models
- **GGML**: Uses quantized GGML format models
- **TVM**: Requires model compilation using TVM compiler

Use `scripts/setup_test_models.sh` to download and prepare test models for all backends.

## Integration with Main Project

The neuriplo library is designed as a standalone component that can be integrated into larger projects:

1. **CMake Integration**: Can be included via `add_subdirectory()` or `FetchContent`
2. **Version Synchronization**: All backend versions are managed centrally in this library
3. **Dependency Isolation**: Each backend's dependencies are self-contained
4. **Testing Framework**: Comprehensive testing across all backends

Example integration:
```cmake
# In main project CMakeLists.txt
add_subdirectory(neuriplo)
target_link_libraries(my_project PRIVATE neuriplo)
```

## Contributing

When adding new inference backends:

1. **Update versions.cmake**: Add version variables for the new backend
2. **Update DependencyValidation.cmake**: Add validation functions for the backend
3. **Update setup_dependencies.sh**: Add installation logic (if automatic download is possible)
4. **Update unified setup**: Add installation logic to `scripts/setup_dependencies.sh` for the new backend
5. **Update CMakeLists.txt**: Add backend to `SUPPORTED_BACKENDS` list
6. **Create backend implementation**: Add source files in `backends/<backend>/src/`
7. **Add tests**: Create test files in `backends/<backend>/test/`
8. **Update documentation**: Document the new backend in this file
9. **Add testing integration**: Integrate with `scripts/test_backends.sh` framework

## Available Scripts

For reference, the following scripts are available in the `scripts/` directory:

- `setup_dependencies.sh` - Unified setup for ONNX Runtime, TensorRT, LibTorch, OpenVINO
- `setup_libtensorflow.sh` - TensorFlow C++ library setup
- `setup_tensorflow_pip.sh` - TensorFlow pip installation
- `setup_onnx_runtime.sh` - Individual ONNX Runtime setup
- `setup_tensorrt.sh` - Individual TensorRT setup  
- `setup_libtorch.sh` - Individual LibTorch setup
- `setup_openvino.sh` - Individual OpenVINO setup
- `test_backends.sh` - Backend testing framework
- `run_complete_tests.sh` - Complete test suite execution
- `validate_test_system.sh` - Test system validation
- `model_downloader.py` - Test model download utility
- `setup_test_models.sh` - Test model preparation
- `analyze_test_results.sh` - Test result analysis
