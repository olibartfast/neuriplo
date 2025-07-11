# Dependency Management for InferenceEngines

This document describes dependency management system for the InferenceEngines library, which provides a unified approach to managing inference backend dependencies.

## Architecture

### Version Management

All inference backend versions are centrally managed in `cmake/versions.cmake`:

```cmake
# Inference Backend Versions
set(ONNX_RUNTIME_VERSION "1.19.2" CACHE STRING "ONNX Runtime version")
set(TENSORRT_VERSION "10.7.0.23" CACHE STRING "TensorRT version")
set(LIBTORCH_VERSION "2.0.0" CACHE STRING "LibTorch version")
set(OPENVINO_VERSION "2023.1.0" CACHE STRING "OpenVINO version")
set(TENSORFLOW_VERSION "2.13.0" CACHE STRING "TensorFlow version")
```

### Dependency Validation

The `cmake/DependencyValidation.cmake` module provides validation:

- **System Dependencies**: OpenCV, glog, CMake version
- **Inference Backends**: ONNX Runtime, TensorRT, LibTorch, OpenVINO
- **GPU Support**: CUDA validation for GPU-enabled backends
- **Installation Completeness**: Checks for required files and libraries

### Setup Scripts

#### Unified Setup Script

The main setup script `scripts/setup_dependencies.sh` supports all inference backends:

```bash
# Setup ONNX Runtime
./scripts/setup_dependencies.sh --backend ONNX_RUNTIME

# Setup TensorRT
./scripts/setup_dependencies.sh --backend TENSORRT

# Setup LibTorch
./scripts/setup_dependencies.sh --backend LIBTORCH

# Setup OpenVINO
./scripts/setup_dependencies.sh --backend OPENVINO
```

#### Individual Backend Scripts

Individual scripts are available:

- `scripts/setup_onnx_runtime.sh`
- `scripts/setup_tensorrt.sh`
- `scripts/setup_libtorch.sh`
- `scripts/setup_openvino.sh`

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
   # For ONNX Runtime
   ./scripts/setup_dependencies.sh --backend ONNX_RUNTIME
   
   # For TensorRT (requires manual download)
   ./scripts/setup_dependencies.sh --backend TENSORRT
   ```

2. **Set environment variables**:
   ```bash
   source $HOME/dependencies/setup_env.sh
   ```

3. **Build the library**:
   ```bash
   mkdir build && cd build
   cmake .. -DDEFAULT_BACKEND=ONNX_RUNTIME
   make
   ```

### Configuration Options

#### CMake Variables

- `DEFAULT_BACKEND`: Choose the inference backend (ONNX_RUNTIME, TENSORRT, LIBTORCH, OPENVINO)
- `DEPENDENCY_ROOT`: Set custom dependency installation root
- `ONNX_RUNTIME_VERSION`: Override ONNX Runtime version
- `TENSORRT_VERSION`: Override TensorRT version
- `LIBTORCH_VERSION`: Override LibTorch version
- `OPENVINO_VERSION`: Override OpenVINO version

#### Environment Variables

The setup script creates environment variables for each backend:

```bash
export ONNX_RUNTIME_DIR="$HOME/dependencies/onnxruntime-linux-x64-gpu-1.19.2"
export TENSORRT_DIR="$HOME/dependencies/TensorRT-10.7.0.23"
export LIBTORCH_DIR="$HOME/dependencies/libtorch"
export OPENVINO_DIR="$HOME/dependencies/openvino-2023.1.0"
```

## Supported Platforms

### Linux (Ubuntu/Debian)
- Full support for all inference backends
- Automatic system dependency installation
- GStreamer support for video processing

### Linux (CentOS/RHEL/Fedora)
- Not tested on these distros

### Windows
- Not tested/not installed


## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   # Reinstall with force flag
   ./scripts/setup_dependencies.sh --backend ONNX_RUNTIME --force
   ```

2. **Version Conflicts**:
   ```bash
   # Override version in CMake
   cmake .. -DONNX_RUNTIME_VERSION=1.18.0
   ```

3. **CUDA Issues**:
   ```bash
   # Check CUDA installation
   nvcc --version
   nvidia-smi
   ```

### Validation Errors

The validation system provides detailed error messages:

```
[ERROR] ONNX Runtime not found at /home/user/dependencies/onnxruntime-linux-x64-gpu-1.19.2
Please ensure the inference backend is properly installed or run the setup script.
```

### Manual Installation

For backends requiring manual installation (TensorRT, OpenVINO):

1. Download from official websites
2. Extract to the dependency root directory
3. Run validation: `./scripts/setup_dependencies.sh --backend TENSORRT`

## Integration with Main Project

The InferenceEngines library is designed to be integrated into the main object-detection-inference project:

1. **FetchContent Integration**: The main project fetches this library using CMake FetchContent
2. **Version Synchronization**: Backend versions are managed here, not in the main project
3. **Setup Script Coordination**: Main project setup scripts call this library's setup scripts

## Contributing

When adding new inference backends:

1. **Update versions.cmake**: Add version variables
2. **Update DependencyValidation.cmake**: Add validation functions
3. **Update setup_dependencies.sh**: Add installation logic
4. **Create individual script**: Add backward compatibility script
5. **Update documentation**: Document the new backend
