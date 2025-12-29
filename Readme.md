
# Neuriplo

![Alt text](data/neuriplo.png)

## Overview

* Neuriplo is a C++ library designed for seamless integration of various backend engines for inference tasks. 
* It supports multiple frameworks and libraries such as OpenCV DNN, TensorFlow, PyTorch (LibTorch), ONNX Runtime, TensorRT, OpenVINO, TVM and GGML.
* The project aims to provide a unified interface for performing inference using these backends, allowing flexibility in choosing the most suitable backend based on performance or compatibility requirements.
* The library is currently mainly used as component of the [Object Detection Inference Project](https://github.com/olibartfast/object-detection-inference)

## Dependencies 
- C++17
- OpenCV
- glog

### Supported Backends (Inside [versions.env](versions.env) file, versions tested in this project):
* OpenCV DNN module
* ONNX Runtime 
* Pytorch (Libtorch) 
* TensorRT 
* OpenVINO 
* Tensorflow (LibTensorFlow C++ library) - inference on saved models, not graph
* GGML - Efficient tensor library for machine learning
* TVM - Open deep learning compiler stack

### Optional
* CUDA (if you want to use GPU)

## Quick Start
### Automated Setup and Testing

#### Setup Dependencies for a Specific Backend

```bash
./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
```

Supported `<BACKEND_NAME>` values:

* `OPENCV_DNN`
* `ONNX_RUNTIME`
* `LIBTORCH`
* `TENSORRT`
* `LIBTENSORFLOW`
* `OPENVINO`
* `GGML`
* `TVM`

#### Test All Backends
```bash
./scripts/test_backends.sh
````

#### Test a Specific Backend

```bash
./scripts/test_backends.sh --backend <BACKEND_NAME>
```


### Manual Build Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/neuriplo.git
   cd neuriplo
   ```

2. Create a build directory and navigate into it:

   ```bash
   mkdir build
   cd build
   ```

3. Configure the build with CMake:

   ```bash
   cmake ..
   ```

   Optionally, you can specify the default backend by setting `-DDEFAULT_BACKEND=your_backend` during configuration.
   - **Note**: If the backend package is not installed on your system, set the path manually in the backend's CMake module or use the automated setup scripts.

4. Build the project:

   ```bash
   cmake --build .
   ```

   This will compile the project along with the selected backend(s).

## Usage

To use the Neuriplo library in your project, link against it and include necessary headers ([check the example here](https://github.com/olibartfast/object-detection-inference/blob/master/app/CMakeLists.txt)):

```cmake
target_link_libraries(your_project PRIVATE neuriplo)
target_include_directories(your_project PRIVATE path_to/neuriplo/include)
```

Ensure you have initialized and set up the selected backend(s) appropriately in your code using the provided interface headers.

## Backend Configuration System

Neuriplo uses a centralized configuration system that makes it easy to add new backends. The system consists of:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Backend Configuration Flow                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. versions.env                                                │
│     ┌────────────────────────────────────────────────────┐      │
│     │ Defines version numbers:                           │      │
│     │ TVM_VERSION=0.18.0                                 │      │
│     │ ONNX_RUNTIME_VERSION=1.19.2                        │      │
│     │ PYTORCH_VERSION=2.3.0                              │      │
│     └────────────────────────────────────────────────────┘      │
│                           ↓                                      │
│  2. cmake/versions.cmake                                        │
│     ┌────────────────────────────────────────────────────┐      │
│     │ Reads versions.env and validates consistency       │      │
│     │ Maps backends to their version variables           │      │
│     │ Ensures all backends have version variables        │      │
│     └────────────────────────────────────────────────────┘      │
│                           ↓                                      │
│  3. scripts/*.sh                                                │
│     ┌────────────────────────────────────────────────────┐      │
│     │ Backend arrays and mappings defined directly       │      │
│     │ Sources versions.env for version numbers           │      │
│     └────────────────────────────────────────────────────┘      │
│                           ↓                                      │
│  ✓ All bash scripts and CMake automatically synchronized        │
└─────────────────────────────────────────────────────────────────┘
```

### Adding a New Backend

To add a new backend (e.g., NCNN), you need to edit **three files**:

1. **Add to `versions.env`**:
   ```bash
   NCNN_VERSION=1.0.34
   ```

2. **Add to `cmake/versions.cmake`**:
   ```cmake
   # Add cache variable after read_versions_from_env()
   set(NCNN_VERSION "${NCNN_VERSION}" CACHE STRING "NCNN version")
   
   # Add to BACKEND_VERSION_MAPPING
   set(BACKEND_VERSION_MAPPING
       ...
       "NCNN:NCNN_VERSION"
   )
   ```

3. **Add to relevant scripts** (e.g., `scripts/test_backends.sh`):
   ```bash
   # Add to BACKENDS array
   BACKENDS=(...  "NCNN")
   
   # Add to mapping arrays
   BACKEND_DIRS=(... ["NCNN"]="ncnn")
   BACKEND_TEST_EXES=(... ["NCNN"]="NCNNInferTest")
   ```

That's it! The validation system will automatically verify consistency, and all scripts will recognize the new backend.

### Validation

The system automatically validates that every backend has a corresponding version:

```bash
# CMake validation (runs automatically)
cmake ..
```

## Documentation

For detailed documentation, see the [docs/](docs/) directory:

- **[Dependency Management](docs/DEPENDENCY_MANAGEMENT.md)** - Complete setup guide for all backends
- **[TVM Build Guide](docs/TVM_BUILD_GUIDE.md)** - Detailed instructions for building and using TVM backend
