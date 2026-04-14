# GEMINI.md - Neuriplo Project Context

## Agent Guidance
**CRITICAL:** All agents MUST follow, attain, read, and update the **`AGENTS.md`** file. This file serves as the primary reference and source of truth for all agents operating in this repository. Ensure that any architectural changes, new conventions, or workflow updates are reflected in `AGENTS.md` to maintain consistency across all agent interactions.

## Project Overview
**Neuriplo** is a high-performance C++17 library that provides a unified abstraction layer for various AI inference engines. It allows developers to switch between different backends seamlessly without changing their application logic.

### Core Technologies
- **Language:** C++17
- **Build System:** CMake (modular architecture)
- **Primary Dependencies:** OpenCV (core & DNN), glog, GoogleTest
- **Supported Backends:**
  - **ONNX Runtime:** General purpose ONNX inference
  - **LibTorch:** PyTorch models (TorchScript)
  - **TensorRT:** NVIDIA GPU-optimized inference
  - **OpenVINO:** Intel CPU/GPU/NPU-optimized inference
  - **OpenCV DNN:** Lightweight, dependency-free inference
  - **LibTensorFlow:** TensorFlow SavedModel inference
  - **GGML:** Efficient tensor library (CPU-focused, quantization support)
  - **TVM:** Apache TVM compiler stack
  - **MIGraphX:** AMD ROCm-optimized inference

### Architecture
The library uses a factory pattern. The core abstraction is `InferenceInterface` (defined in `backends/src/InferenceInterface.hpp`). Backends are instantiated via `setup_inference_engine()` (defined in `include/InferenceBackendSetup.hpp`).

## Building and Running

### Linux (Primary Platform)
1. **Setup Dependencies:**
   ```bash
   ./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
   ```
2. **Configure & Build:**
   ```bash
   mkdir build && cd build
   cmake .. -DDEFAULT_BACKEND=<BACKEND_NAME> -DBUILD_INFERENCE_ENGINE_TESTS=ON
   make -j$(nproc)
   ```
3. **Run Tests:**
   ```bash
   ./scripts/test_backends.sh --backend <BACKEND_NAME>
   ```

### Windows (Native MSVC)
1. **Prerequisites:** Install `vcpkg`, `cmake`, and `Visual Studio 2022`.
2. **Install Base Deps:**
   ```powershell
   vcpkg install opencv4:x64-windows glog:x64-windows gtest:x64-windows
   ```
3. **Configure & Build:**
   ```powershell
   cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN `
         -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
   cmake --build build --config Release
   ```
4. **Run Tests:**
   ```powershell
   ctest --test-dir build -C Release
   ```

## Development Conventions

### 1. Centralized Versioning
Backend versions are managed in two places that **MUST** remain synchronized:
- `versions.env`: Source of truth for version strings (bash/script compatible).
- `cmake/versions.cmake`: Maps environment variables to CMake cache variables and validates them.

### 2. Adding a New Backend
To add a new backend (e.g., `MYBACKEND`):
1. Add `MYBACKEND_VERSION=x.y.z` to `versions.env`.
2. Add the version mapping and cache variable to `cmake/versions.cmake`.
3. Add `MYBACKEND` to `SUPPORTED_BACKENDS` in the root `CMakeLists.txt`.
4. Implement the backend class in `backends/mybackend/src/` inheriting from `InferenceInterface`.
5. Update `src/InferenceBackendSetup.cpp` to include the new backend in the factory function.
6. Add unit tests in `backends/mybackend/test/` and update `scripts/test_backends.sh`.

### 3. Error Handling
- Use the custom exceptions defined in `InferenceInterface.hpp`: `ModelLoadException`, `InferenceExecutionException`.
- Use `LOG(INFO)`, `LOG(ERROR)` from `glog` for logging.

### 4. Input/Output Format
- Inputs are typically `std::vector<std::vector<uint8_t>>` (raw byte buffers).
- Outputs are `std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>` (data and shapes).
- `TensorElement` is a `std::variant` supporting `float`, `int32_t`, `int64_t`, and `uint8_t`.

## Key Files
- `backends/src/InferenceInterface.hpp`: Base class for all backends.
- `include/InferenceBackendSetup.hpp`: Public API for backend instantiation.
- `versions.env`: Central version configuration.
- `cmake/versions.cmake`: CMake logic for version management.
- `scripts/setup_dependencies.sh`: Unified dependency installer for Linux.
- `scripts/test_backends.sh`: Unified test runner.
