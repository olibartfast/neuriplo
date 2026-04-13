# Agent Instructions

- Repo-local agent metadata lives in `REPO_META.yaml`.
- Use `REPO_META.yaml` as the local source of truth for build/test entrypoints, owned paths, and allowed automated change classes.
- `develop` is the integration branch for normal work.
- `master` is release-only.
- Prioritize correctness, backend compatibility, dependency safety, device placement assumptions, fallback behavior, and performance regressions.
- Best practice: commit intentional, scoped changes before branch handoff.
- Best practice: push the working branch before starting branch-closure or integration steps.
- Best practice: after merging a feature branch into `develop`, push local `develop` to `origin/develop`, remove the merged feature branch locally and remotely, and update related docs and `Readme.md` when behavior or workflow changes.


# Neuriplo –  Instructions

## Project Overview

Neuriplo is a C++17 shared library that provides a **single unified inference interface** over multiple ML backends (ONNX Runtime, LibTorch, TensorFlow, TensorRT, OpenVINO, OpenCV DNN, GGML, TVM). Only **one backend is compiled at a time**, selected via the `DEFAULT_BACKEND` CMake variable.

## Build

```bash
cmake -B build -DDEFAULT_BACKEND=OPENCV_DNN   # or ONNX_RUNTIME, LIBTORCH, etc.
cmake --build build
```

Supported `DEFAULT_BACKEND` values: `OPENCV_DNN`, `ONNX_RUNTIME`, `LIBTORCH`, `LIBTENSORFLOW`, `TENSORRT`, `OPENVINO`, `GGML`, `TVM`, `MIGRAPHX`.

Backend dependencies are expected under `~/dependencies/` by default (configured in `cmake/versions.cmake`). Use the setup script to install them:

```bash
./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
```

## Tests

Each backend has its own test executable. To build and run tests for one backend:

```bash
cmake -B build -DDEFAULT_BACKEND=ONNX_RUNTIME
cmake --build build
cd build && ctest           # or run the executable directly:
./build/backends/onnx-runtime/test/ONNXRuntimeInferTest
```

Test executable names follow the pattern `<BackendName>InferTest` (e.g. `OCVDNNInferTest`, `GGMLInferTest`, `TensorFlowInferTest`).

To test all backends via Docker (mirrors CI):

```bash
./scripts/test_backends.sh
./scripts/test_backends.sh --backend ONNX_RUNTIME   # single backend
```

## Architecture

```
include/
  InferenceBackendSetup.hpp   ← public header; declares setup_inference_engine()
  common.hpp                  ← common includes (OpenCV, glog, etc.)

backends/src/
  InferenceInterface.hpp/.cpp ← abstract base class all backends derive from
  InferenceMetadata.hpp/.cpp  ← input/output layer shape info
  MockInferenceInterface.hpp  ← Google Mock stub for unit tests
  BackendTestTemplate.hpp     ← CRTP template base for all backend test fixtures

backends/<backend>/src/       ← backend-specific implementation (e.g. ORTInfer)
backends/<backend>/test/      ← backend-specific tests + model generation scripts

src/
  InferenceBackendSetup.cpp   ← factory: setup_inference_engine() selects backend via #ifdef
```

**Factory function** (`setup_inference_engine`) uses preprocessor macros (`USE_ONNX_RUNTIME`, `USE_LIBTORCH`, etc.) set by CMake to instantiate the correct backend class. There is no runtime dispatch — backend selection is compile-time only.

**Core interface** (`InferenceInterface`) takes raw byte tensors as input:
```cpp
virtual std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) = 0;
```
`TensorElement` is `std::variant<float, int32_t, int64_t, uint8_t>`.

## Key Conventions

### Adding a new backend — touch exactly three files

1. **`versions.env`** — add `NEWBACKEND_VERSION=x.y.z`
2. **`cmake/versions.cmake`** — add a cache variable and an entry in `BACKEND_VERSION_MAPPING`
3. **Relevant scripts** — add to `BACKENDS`, `BACKEND_DIRS`, and `BACKEND_TEST_EXES` arrays in `scripts/test_backends.sh`

CMake validates that every backend has a version entry on every `cmake ..` run.

### Version management

All backend versions live in `versions.env`. CMake reads this file via `cmake/versions.cmake`. Shell scripts source it directly. Never hardcode version strings in scripts or CMake — always read from `versions.env`.

### Test structure

Backend tests use a **hybrid mock + integration** pattern via `BackendHybridTestBase<BackendClass>` (CRTP):

- **Mock unit tests** always run (no model needed) — use `MockInferenceInterface` + Google Mock.
- **Integration tests** are skipped (`GTEST_SKIP`) if no real model is found, falling back to a generated dummy model.
- Tests are named with prefixes `Unit_`, `Integration_`, `Performance_`, `Stress_` via macros in `BackendTestTemplate.hpp`.
- Each backend test directory contains `generate_model.sh` and a Python export script to produce test models.

### Dependency paths

Backend libraries are installed to `~/dependencies/<backend>-<version>/` and referenced via CMake cache variables like `ONNX_RUNTIME_DIR`, `LIBTORCH_DIR`, etc. Override on the CMake command line if installing elsewhere:

```bash
cmake -B build -DONNX_RUNTIME_DIR=/opt/onnxruntime-1.19.2 -DDEFAULT_BACKEND=ONNX_RUNTIME
```

### CI

CI runs each CPU backend in a dedicated Docker image (`docker/Dockerfile.<backend>`). GPU backends (TensorRT, LibTorch GPU, MIGraphX) require a self-hosted runner and are gated by `if: false` in `.github/workflows/ci.yml` until one is registered. MIGraphX requires `--device=/dev/kfd --device=/dev/dri --group-add video` for GPU passthrough.

