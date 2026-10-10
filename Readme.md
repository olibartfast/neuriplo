# Neuriplo

[![C++](https://img.shields.io/badge/language-C++17-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.10.0-blue.svg)](https://github.com/olibartfast/neuriplo/releases/tag/v0.10.0)

![Neuriplo](data/neuriplo.png)

One C++ interface over many inference runtimes: ONNX Runtime, TensorRT,
OpenVINO, LibTorch, ExecuTorch, LiteRT, llama.cpp and more, plus NVIDIA DALI for
GPU preprocessing. Neuriplo also ships its own runtime, NERT (Neuriplo Engine
Runtime), with no vendor SDK underneath.

Write your application once against `InferenceInterface`, then pick the runtime
that fits your model and hardware at build time, without rewriting code around
each vendor SDK. Neuriplo is the backend layer of
[neuriplo-infer](https://github.com/olibartfast/neuriplo-infer).

**Start here, then go deeper:** this page is the short path: build, test, link.
Per-backend setup, model formats and Docker workflows are in
**[docs/DEPENDENCY_MANAGEMENT.md](docs/DEPENDENCY_MANAGEMENT.md)**. The design
behind the interface is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Table of Contents
- [Quick Start](#quick-start): from a clean Ubuntu box to passing tests
- [Using the Library](#using-the-library): C++ API and the stable C ABI
- [Choosing a Backend](#choosing-a-backend)
- [Versions](#versions)
- [Common Build Options](#common-build-options)
- [Documentation](#documentation)
- [Testing](#testing)

---

## Quick Start

Four steps on Ubuntu 24.04, using the default backend, NERT (CPU: no GPU, no CUDA,
no vendor SDK).

### 1. Install the toolchain

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential pkg-config \
  libgoogle-glog-dev libgtest-dev libgmock-dev
```

OpenCV is needed only by the OpenCV DNN backend.

### 2. Build

NERT is the default backend, so no `-DDEFAULT_BACKEND` is needed:

```bash
git clone https://github.com/olibartfast/neuriplo.git
cd neuriplo
cmake -S . -B build
cmake --build build --parallel
```

### 3. Test

NERT's tests run a real ResNet-18 ONNX model, so provision it first. The script
needs `torch` and `torchvision` installed (it pins torchvision 0.27.0; torch is
not pinned):

```bash
python3 backends/nert/test/provision_parity_fixture.py fixtures
NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure
```

Without the model those tests fail on purpose rather than skip.

### 4. Switch backend

Every other backend has a setup script that installs the pinned dependency into
`~/dependencies`:

```bash
./scripts/setup_dependencies.sh --backend ONNX_RUNTIME
cmake -S . -B build -DDEFAULT_BACKEND=ONNX_RUNTIME
cmake --build build --parallel
```

OpenCV DNN needs `libopencv-dev` and builds with `-DDEFAULT_BACKEND=OPENCV_DNN`.

---

## Using the Library

Link against `neuriplo` and include the setup header
([full example](https://github.com/olibartfast/neuriplo-infer/blob/master/app/CMakeLists.txt)):

```cmake
target_link_libraries(your_project PRIVATE neuriplo)
target_include_directories(your_project PRIVATE path_to/neuriplo/include)
```

```cpp
#include "InferenceBackendSetup.hpp"

auto engine = setup_inference_engine("model.onnx", /*use_gpu=*/false);
if (!engine) {
    // the model failed to load; the backend logged why
}
auto metadata = engine->get_inference_metadata();          // input/output names and shapes
auto [outputs, shapes] = engine->get_infer_results(inputs); // inputs: one byte buffer per tensor
```

That C++ API passes STL types across the library boundary, so the application
must use the same toolchain as `libneuriplo`. Anything else (C, C++ built with
another compiler, Python, C#/Unity) uses the stable **C ABI**
(`neuriplo/neuriplo_c.h`) or its header-only C++ wrapper (`neuriplo/neuriplo.hpp`)
from an installed package:

```cmake
find_package(neuriplo CONFIG REQUIRED)
target_link_libraries(your_project PRIVATE neuriplo::neuriplo)
```

See **[docs/C_API.md](docs/C_API.md)** for installation, ownership, threading,
and examples.

---

## Choosing a Backend

`-DDEFAULT_BACKEND=<ID>` selects the runtime compiled into the library, and
`./scripts/setup_dependencies.sh --backend <ID>` installs it.

| ID | Runtime | Model format | Best for |
|----|---------|--------------|----------|
| `OPENCV_DNN` | OpenCV DNN | `.onnx`, Darknet | Zero-setup CPU inference |
| `NERT` (default) | NERT (Neuriplo Engine Runtime, first-party) | `.onnx` (opset 18) | Self-contained CPU inference with no vendor SDK; in development |
| `ONNX_RUNTIME` | ONNX Runtime | `.onnx` | General CPU/GPU inference, many execution providers |
| `TENSORRT` | TensorRT | `.engine` | Production on NVIDIA GPUs |
| `OPENVINO` | OpenVINO | IR (`.xml` + `.bin`) | Intel CPUs and GPUs |
| `LIBTORCH` | LibTorch | TorchScript `.pt`, AOTInductor `.pt2` | PyTorch models on server/desktop |
| `EXECUTORCH` | ExecuTorch | `.pte` | PyTorch models on edge devices |
| `LITERT` | LiteRT (TensorFlow Lite) | `.tflite` | Mobile and edge |
| `LIBTENSORFLOW` | TensorFlow C++ | SavedModel | Existing TensorFlow models |
| `MIGRAPHX` | MIGraphX | `.onnx` | AMD GPUs (ROCm) |
| `LLAMACPP` | llama.cpp | GGUF | LLM and multimodal generation |
| `CACTUS` | Cactus | GGUF | Text generation on ARM64 |
| `DALI` | NVIDIA DALI | `.dali` pipeline | GPU preprocessing in front of a model backend |
| `GGML` | GGML | GGML | Experimental: returns placeholder output |
| `TVM` | Apache TVM | compiled module | Experimental: returns placeholder output |

> **NERT status:** CPU only. Loads opset-18 ONNX models (convert others with
> `onnx.version_converter`). Outputs match `ONNX_RUNTIME` on YOLO (v5–v10, v12
> detection; YOLO11 segmentation and pose; YOLO26 detection, segmentation and
> pose) and on ViT, ViTPose, Depth Anything V2, VideoMAE, ViViT and TimeSformer.
> DETR-family, RAFT and open-vocabulary coverage in progress. See
> [engine/README.md](engine/README.md) and [the roadmap](specs/roadmap.md#nert-track).

Several backends can be compiled into one library (`-DNEURIPLO_BACKENDS`), or
built as runtime-loaded plugins (`-DNEURIPLO_PLUGIN_BACKENDS`); see
[docs/PLUGIN_BACKENDS.md](docs/PLUGIN_BACKENDS.md). Per-backend setup notes,
architecture and GPU support, and model sources are in
[docs/DEPENDENCY_MANAGEMENT.md](docs/DEPENDENCY_MANAGEMENT.md).

---

## Versions

> **Every dependency version is pinned once, in [`versions.env`](versions.env).**
> CMake reads it through `cmake/versions.cmake` and the setup scripts source it
> directly, so a bump is a one-line edit. The tested version of each backend is
> listed in the generated
> [backend table](docs/DEPENDENCY_MANAGEMENT.md#supported-backends).

| Component | Requirement |
|-----------|-------------|
| C++ compiler | C++17 (C++20 when LibTorch is enabled) |
| CMake | 3.10+ |
| glog | any recent release |
| CUDA | only for GPU backends; version pinned in `versions.env` |

---

## Common Build Options

| Option | Default | Effect |
|--------|---------|--------|
| `-DDEFAULT_BACKEND=<ID>` | `NERT` | Backend compiled in and used by `setup_inference_engine` |
| `-DNEURIPLO_BACKENDS="A;B"` | `DEFAULT_BACKEND` | Compile several backends into one library |
| `-DNEURIPLO_PLUGIN_BACKENDS="A;B"` | empty | Build backends as runtime-loaded plugins |
| `-DBUILD_INFERENCE_ENGINE_TESTS=ON/OFF` | `ON` | Build the test suites |
| `-DNEURIPLO_INSTALL=ON/OFF` | `ON` when top-level | Generate install rules and the CMake package |
| `-DWERROR=ON/OFF` | `OFF` | Treat compiler warnings as errors (what CI does) |
| `-DSANITIZERS=ON/OFF` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer |
| `-DORT_ENABLE_<EP>_EP=ON/OFF` | `OFF` | ONNX Runtime execution providers; see [docs/ORT_EXECUTION_PROVIDERS.md](docs/ORT_EXECUTION_PROVIDERS.md) |

---

## Documentation

| Document | Covers |
|----------|--------|
| **[docs/DEPENDENCY_MANAGEMENT.md](docs/DEPENDENCY_MANAGEMENT.md)** | Per-backend setup, model formats, Docker workflows, version validation |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Adapter, Bridge, Abstract Factory, Decorator and State design of the backend layer |
| **[engine/README.md](engine/README.md)** | NERT engine: layout, build and parity tests |
| **[docs/C_API.md](docs/C_API.md)** | Stable C ABI and C++ wrapper for third-party applications |
| **[docs/ADDING_BACKEND.md](docs/ADDING_BACKEND.md)** | Implementing and registering a new backend |
| **[docs/PLUGIN_BACKENDS.md](docs/PLUGIN_BACKENDS.md)** | Multi-backend builds and the plugin ABI |
| **[docs/ORT_EXECUTION_PROVIDERS.md](docs/ORT_EXECUTION_PROVIDERS.md)** | ONNX Runtime execution provider selection |
| **[docs/EXECUTORCH_DELEGATES.md](docs/EXECUTORCH_DELEGATES.md)** | ExecuTorch delegate selection |
| **[docs/TVM_BUILD_GUIDE.md](docs/TVM_BUILD_GUIDE.md)** | Building TVM from source |
| **[docs/CODE_QUALITY.md](docs/CODE_QUALITY.md)** | Formatting, static analysis, sanitizers, git hooks |
| **[docs/LOCAL_CI.md](docs/LOCAL_CI.md)** | Running the CI workflows locally with `act` |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Known build and CI failures and their fixes |
| **[docs/Versioning.md](docs/Versioning.md)** | Release and versioning policy |
| **[specs/](specs/)** | Project [mission](specs/mission.md), [tech stack](specs/tech-stack.md) and [roadmap](specs/roadmap.md) |

---

## Testing

```bash
ctest --test-dir build --output-on-failure           # the backend you built
./scripts/test_backends.sh --backend <ID>            # setup + build + test one backend
./scripts/test_backends.sh                           # every backend
```

Local quality gate and git hooks:

```bash
./scripts/quality/setup_hooks.sh   # pre-commit + pre-push hooks
./scripts/quality/run.sh           # format + cppcheck
./scripts/quality/sanitizers.sh    # ASan + UBSan build and ctest
```

CI runs on pushes and PRs to `master`, `develop`, `release/**` and `feature/**`:

| Workflow | What it does |
|----------|-------------|
| `ci.yml` | Format, backend-docs sync, cppcheck, clang-tidy, `-DWERROR=ON` builds, sanitizers, C ABI checks, per-backend Docker build and test |
| `windows-build.yml` | Windows builds, when a Windows-relevant path or `versions.env` changes |

Run them locally with `act`: see [docs/LOCAL_CI.md](docs/LOCAL_CI.md).

---

## License

[MIT](LICENSE)
