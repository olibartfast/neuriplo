
# Inference Engines

## Overview

* InferenceEngines is a C++ library designed for seamless integration of various backend engines for inference tasks. 
* It supports multiple frameworks and libraries such as OpenCV DNN, TensorFlow, PyTorch (LibTorch), ONNX Runtime, TensorRT, and OpenVINO.
* The project aims to provide a unified interface for performing inference using these backends, allowing flexibility in choosing the most suitable backend based on performance or compatibility requirements.
* The library is currently mainly used as component of the [Object Detection Inference Project](https://github.com/olibartfast/object-detection-inference)

## Dependencies 
- C++17
- OpenCV
- glog

### A backend between (In parentheses, version tested in this project):
* OpenCV DNN module (4.11.0) 
* ONNX Runtime (1.19.2 gpu package)
* LibTorch (2.0.1-cu118)
* TensorRT (10.0.7.23)
* OpenVino (2024.1)
* Libtensorflow (2.13) only inference on saved models, not graph

### Optional
* CUDA (if you want to use GPU)

## Build Instructions

### Build Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/inference_engines.git
   cd InferenceEngines
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
   - **Note**:  If the backend package is not installed on your system, set the path manually in the backend's CMake module (i.e. for Libtorch modify [Libtorch.cmake](cmake/LibTorch.cmake)  or pass the argument ``Torch_DIR``, for onnx-runtume modify [ONNXRuntime.cmake](cmake/ONNXRuntime.cmake) or pass the argument ``ORT_VERSION``, same apply to other backend local packages)

5. Build the project:

   ```bash
   cmake --build .
   ```

   This will compile the project along with the selected backend(s).

## Usage

To use the InferenceEngines library in your project, link against it and include necessary headers ( [check the example here](https://github.com/olibartfast/object-detection-inference/blob/master/CMakeLists.txt)) :

```cmake
target_link_libraries(your_project PRIVATE InferenceEngines)
target_include_directories(your_project PRIVATE path_to/InferenceEngines/include)
```

Ensure you have initialized and set up the selected backend(s) appropriately in your code using the provided interface headers.
