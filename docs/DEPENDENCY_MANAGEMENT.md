# Dependency Management for neuriplo

This document describes the dependency management system for the neuriplo library, which provides a unified approach to managing inference backend dependencies.

## Architecture

### Backend Registry and Version Management

The CMake backend registry is the source of truth for supported backend IDs and
their CMake metadata:

- `cmake/BackendRegistry.cmake`: supported `DEFAULT_BACKEND` values, backend
  CMake modules, test directories, and version-variable mapping.
- `versions.env`: dependency versions.
- `cmake/versions.cmake`: reads `versions.env` and validates that every
  registered backend has a version variable.

### Dependency Validation

The `cmake/DependencyValidation.cmake` module provides validation:

- **System Dependencies**: OpenCV, glog, CMake version
- **Inference Backends**: only the selected `DEFAULT_BACKEND` is validated
- **GPU Support**: CUDA validation for GPU-enabled backends
- **Installation Completeness**: Checks for required files and libraries

### Setup Scripts

#### Unified Setup Script

The main setup script installs dependencies for a selected backend:

```bash
# Setup any backend
./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
```

For supported backend IDs, use the values registered in
`cmake/BackendRegistry.cmake`. Some backends still require backend-specific
installation steps even when they are valid CMake `DEFAULT_BACKEND` values.

#### Individual Backend Scripts

All backends can be set up using the unified script:

```bash
./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
```

Use the same `<BACKEND_NAME>` values accepted by `DEFAULT_BACKEND`.

### GGUF-native backends

The converged multimodal branch adds two GGUF-oriented backends:

- `CACTUS`: Cactus runtime integration for prompt-in / generated-text-out inference.
  **ARM64 only** — the library uses ARM NEON intrinsics unconditionally and cannot be
  compiled on x86_64. Tested targets: Jetson Orin, Raspberry Pi 5.
- `LLAMACPP`: llama.cpp integration for GGUF LLM and multimodal inference.

Both backends are wired through the same neuriplo backend-selection path as the
other optional backends. They are installed through `scripts/setup_dependencies.sh`
and validated through the same CMake dependency validation entry points:

```bash
./scripts/setup_dependencies.sh --backend CACTUS   # ARM64 host required
./scripts/setup_dependencies.sh --backend LLAMACPP
```

### MIGraphX model support

In neuriplo, the MIGraphX backend accepts **ONNX** model files only.

- Models are loaded through MIGraphX's ONNX parser in `backends/migraphx/src/MIGraphXInfer.cpp`
- PyTorch models must be exported to ONNX before use
- Native PyTorch/TorchScript model loading is not supported by this integration

This is a neuriplo backend constraint, not a general statement about every ROCm or MIGraphX integration.

### MIGraphX Docker workflow

Build the MIGraphX test image:

```bash
docker build --rm -t neuriplo:migraphx -f docker/Dockerfile.migraphx .
```

Run the MIGraphX backend test container on a ROCm-capable host:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video neuriplo:migraphx
```

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
   
   ```

### Configuration Options

#### CMake Variables

- `DEFAULT_BACKEND`: Choose one backend registered in `cmake/BackendRegistry.cmake`
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
export CACTUS_DIR="$HOME/dependencies/cactus"
export LLAMACPP_DIR="$HOME/dependencies/llamacpp"
export LD_LIBRARY_PATH="$ONNX_RUNTIME_DIR/lib:$TENSORRT_DIR/lib:$LIBTORCH_DIR/lib:$OPENVINO_DIR/lib:$CACTUS_DIR/lib:$LLAMACPP_DIR/lib:$LD_LIBRARY_PATH"
```

**Note**: TensorFlow environment variables are set by the individual TensorFlow setup scripts.

## Supported Platforms

### Linux (Ubuntu/Debian) — x86_64
- Support for all inference backends except CACTUS (ARM64 only)
- Automatic system dependency installation via apt-get
- OpenCV and glog installation
- TensorFlow C++ library support

### Linux (Ubuntu/Debian) — ARM64 (aarch64)
- Full support for all inference backends including CACTUS
- Tested targets: Jetson Orin, Raspberry Pi 5

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

**ExecuTorch**:

ExecuTorch has no pre-built binary release — it must be built from source. The
recommended path is the project's Docker image, which handles all Python build
dependencies and installs the C++ runtime to `/opt/executorch`:

```bash
docker build --rm -t neuriplo:executorch -f docker/Dockerfile.executorch .
```

For a native (non-Docker) build, follow the steps in
`docker/Dockerfile.executorch` stages `base_dependencies` and
`executorch_install`. The key cmake flags are:

```bash
cmake -S /tmp/executorch -B /tmp/executorch/cmake-out \
    -DCMAKE_INSTALL_PREFIX=/opt/executorch \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_TESTS=OFF
cmake --build /tmp/executorch/cmake-out -j$(nproc)
cmake --install /tmp/executorch/cmake-out
# Do NOT delete the cmake-out directory — ExecuTorchTargets.cmake references
# build-tree paths that must remain accessible at neuriplo configure time.
```

Then build neuriplo:

```bash
cmake -B build -DDEFAULT_BACKEND=EXECUTORCH -DEXECUTORCH_DIR=/opt/executorch
cmake --build build
```

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
5. For neuriplo-specific build and usage details, see [TVM Backend Build Guide](TVM_BUILD_GUIDE.md)
6. For upstream installation details, see [TVM Installation Guide](https://tvm.apache.org/docs/install/from_source.html)

## Testing Integration

### Local CI simulation

Before pushing Dockerfile or workflow changes you can replay any CI job locally
using [nektos/act](https://github.com/nektos/act). See [LOCAL_CI.md](LOCAL_CI.md)
for installation, per-job commands, and YAML debugging flags.

### Automated Testing

The testing framework supports all backends:

```bash
# Test specific backend
./scripts/test_backends.sh --backend <BACKEND_NAME>

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

When adding new inference backends, use
[Adding an Inference Backend](ADDING_BACKEND.md) as the checklist. Keep the
backend ID and CMake metadata in `cmake/BackendRegistry.cmake`; this document
should only include backend-specific operational notes that users need at setup
time.

## Available Scripts

Scripts in `scripts/` follow a consistent naming convention: `setup_<backend>.sh`
for individual backends, with `setup_dependencies.sh` as the unified dispatcher.

### Dependency setup

| Script | Backend / purpose |
|---|---|
| `setup_dependencies.sh` | Unified dispatcher — delegates to the script below for the chosen `--backend` |
| `setup_onnx_runtime.sh` | ONNX Runtime |
| `setup_tensorrt.sh` | NVIDIA TensorRT (manual download required) |
| `setup_libtorch.sh` | PyTorch LibTorch |
| `setup_openvino.sh` | Intel OpenVINO |
| `setup_libtensorflow.sh` | TensorFlow C++ library |
| `setup_tensorflow_pip.sh` | TensorFlow via pip (alternative) |
| `setup_ggml.sh` | GGML |
| `setup_tvm.sh` | Apache TVM |
| `setup_executorch.sh` | ExecuTorch C++ runtime (builds from source) |
| `setup_cactus.sh` | Cactus (ARM64 only — fails fast on x86_64) |
| `setup_llamacpp.sh` | llama.cpp |
| `setup_migraphx.sh` | AMD MIGraphX (installs from ROCm apt repo) |
| `build_cactus.sh` | Builds the Cactus Docker image (ARM64 only) |

### Testing

| Script | Purpose |
|---|---|
| `test_backends.sh` | Build and run tests for one or all backends |
| `run_complete_tests.sh` | Full test suite with result aggregation |
| `validate_test_system.sh` | Validate test system setup |
| `setup_test_models.sh` | Download / generate test models for all backends |
| `model_downloader.py` | Download individual test models |
| `analyze_test_results.sh` | Parse and summarise test result XML files |

### Model conversion

| Script | Purpose |
|---|---|
| `convert_to_ggml.sh` | Convert a model to GGML format |
| `convert_onnx_to_ggml.py` | Convert an ONNX model to GGML |
| `convert_resnet18_to_ggml.py` | Convert ResNet-18 to GGML (test helper) |
