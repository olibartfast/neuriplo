# Dependency Management for neuriplo

This document describes the dependency management system for the neuriplo library.
Backend metadata is maintained in [`docs/backends.yaml`](backends.yaml).
Run `python3 scripts/gen_backend_docs.py` to regenerate all auto-generated sections.

## Supported backends

<!-- GEN:backend-overview -->
| Backend ID | Name | Version | Arch | GPU |
|---|---|---|---|---|
| `OPENCV_DNN` | OpenCV DNN | `4.6.0` | x86_64, ARM64 | no |
| `ONNX_RUNTIME` | ONNX Runtime | `1.19.2` | x86_64, ARM64 | no |
| `LIBTORCH` | LibTorch | `2.3.0` | x86_64, ARM64 | no |
| `LIBTENSORFLOW` | TensorFlow C++ | `2.19.0` | x86_64, ARM64 | no |
| `TENSORRT` | TensorRT | `10.14.1.48` | x86_64 only | yes |
| `OPENVINO` | OpenVINO | `2025.2.0` | x86_64, ARM64 | no |
| `GGML` | GGML | `v0.11.0` | x86_64, ARM64 | no |
| `TVM` | Apache TVM | `0.22.0` | x86_64, ARM64 | no |
| `MIGRAPHX` | MIGraphX | `2.12.0` | x86_64 only | yes |
| `CACTUS` | Cactus | `v1.14` | ARM64 only | no |
| `LLAMACPP` | llama.cpp | `b9049` | x86_64, ARM64 | no |
| `EXECUTORCH` | ExecuTorch | `v1.2.0` | x86_64, ARM64 | no |
<!-- /GEN:backend-overview -->

## Architecture

### Backend Registry and Version Management

The CMake backend registry is the single source of truth for all supported
backend IDs and their CMake metadata:

- `cmake/BackendRegistry.cmake` — supported `DEFAULT_BACKEND` values, CMake
  modules, test directories, and version-variable mapping.
- `versions.env` — pinned dependency versions for all backends.
- `cmake/versions.cmake` — reads `versions.env` and validates that every
  registered backend has a version entry.

### Dependency Validation

`cmake/DependencyValidation.cmake` validates at configure time:

- **System dependencies**: OpenCV, glog, minimum CMake version
- **Selected backend only**: the `DEFAULT_BACKEND` is validated; others are
  ignored.
- **GPU support**: CUDA presence checked for GPU-enabled backends.
- **Installation completeness**: required headers and libraries must exist.

### Setup Scripts

#### Unified dispatcher

```bash
./scripts/setup_dependencies.sh --backend <BACKEND_NAME>
```

`<BACKEND_NAME>` is one of the IDs in the [Supported backends](#supported-backends)
table above.  The dispatcher sources `versions.env` and delegates to the
corresponding `scripts/setup_<backend>.sh`.

#### Individual backend scripts

Each backend has a dedicated script that can also be called directly:

```bash
./scripts/setup_onnx_runtime.sh
./scripts/setup_libtorch.sh
./scripts/setup_openvino.sh
./scripts/setup_libtensorflow.sh
./scripts/setup_ggml.sh
./scripts/setup_tvm.sh
./scripts/setup_executorch.sh          # builds from source
./scripts/setup_cactus.sh              # ARM64 host required
./scripts/setup_llamacpp.sh
./scripts/setup_migraphx.sh            # requires ROCm
# TensorRT requires a manual download first — see Manual Installation below
```

### GGUF-native backends

Two GGUF-oriented backends provide LLM and multimodal inference:

- `CACTUS`: Cactus runtime — prompt bytes in, generated text out.
  **ARM64 only** — the library uses ARM NEON intrinsics unconditionally and
  cannot be compiled on x86_64. Tested targets: Jetson Orin, Raspberry Pi 5.
- `LLAMACPP`: llama.cpp — GGUF LLM and multimodal inference.

```bash
./scripts/setup_cactus.sh              # ARM64 host required
./scripts/setup_llamacpp.sh
```

### MIGraphX

MIGraphX ships as part of ROCm — there is no separate source build.

- neuriplo's MIGraphX backend accepts **ONNX** model files only.
- PyTorch models must be exported to ONNX before use.
- Requires an AMD GPU with ROCm support.

```bash
./scripts/setup_migraphx.sh            # installs migraphx + migraphx-dev from apt
```

Docker test image on a ROCm-capable host:

```bash
docker build --rm -t neuriplo:migraphx -f docker/Dockerfile.migrachx .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video neuriplo:migraphx
```

## Usage

### Quick start

```bash
# 1. Install backend dependencies
./scripts/setup_dependencies.sh --backend ONNX_RUNTIME

# 2. Export environment variables written by the setup script
source $HOME/dependencies/setup_env.sh

# 3. Configure and build
cmake -B build -DDEFAULT_BACKEND=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build

# 4. Run tests
ctest --test-dir build --output-on-failure
```

### Configuration Options

#### CMake variables

Static variables:

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_BACKEND` | — | Backend to compile (required) |
| `BUILD_INFERENCE_ENGINE_TESTS` | `OFF` | Build GTest executables |
| `DEPENDENCY_ROOT` | `$HOME/dependencies` | Root for installed backends |

Per-backend install-path overrides:

<!-- GEN:cmake-dir-variables -->
| Variable | Default path |
|---|---|
| `ONNX_RUNTIME_DIR` | `$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-<ver>` |
| `LIBTORCH_DIR` | `$DEPENDENCY_ROOT/libtorch` |
| `TENSORFLOW_DIR` | `$DEPENDENCY_ROOT/tensorflow` |
| `TENSORRT_DIR` | `$DEPENDENCY_ROOT/TensorRT-<ver>` |
| `OPENVINO_DIR` | `$DEPENDENCY_ROOT/openvino_<ver>` |
| `GGML_DIR` | `$DEPENDENCY_ROOT/ggml` |
| `TVM_DIR` | `$DEPENDENCY_ROOT/tvm` |
| `MIGRAPHX_ROOT` | `/opt/rocm` |
| `CACTUS_DIR` | `$DEPENDENCY_ROOT/cactus` |
| `LLAMACPP_DIR` | `$DEPENDENCY_ROOT/llamacpp` |
| `EXECUTORCH_DIR` | `$HOME/dependencies/executorch` |
<!-- /GEN:cmake-dir-variables -->

Per-backend version overrides (default from `versions.env`):

<!-- GEN:cmake-version-variables -->
| Variable | Current value in `versions.env` |
|---|---|
| `OPENCV_VERSION` | `4.6.0` |
| `ONNX_RUNTIME_VERSION` | `1.19.2` |
| `PYTORCH_VERSION` | `2.3.0` |
| `TENSORFLOW_VERSION` | `2.19.0` |
| `TENSORRT_VERSION` | `10.14.1.48` |
| `OPENVINO_VERSION` | `2025.2.0` |
| `GGML_VERSION` | `v0.11.0` |
| `TVM_VERSION` | `0.22.0` |
| `MIGRAPHX_VERSION` | `2.12.0` |
| `CACTUS_VERSION` | `v1.14` |
| `LLAMACPP_VERSION` | `b9049` |
| `EXECUTORCH_VERSION` | `v1.2.0` |
<!-- /GEN:cmake-version-variables -->

#### Environment variables written by setup scripts

After running any `setup_*.sh` script, source `$HOME/dependencies/setup_env.sh`
to export:

<!-- GEN:env-variables -->
```bash
export DEPENDENCY_ROOT="$HOME/dependencies"
export ONNX_RUNTIME_DIR="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-1.19.2"
export LIBTORCH_DIR="$DEPENDENCY_ROOT/libtorch"
export TENSORFLOW_DIR="$DEPENDENCY_ROOT/tensorflow"
export TENSORRT_DIR="$DEPENDENCY_ROOT/TensorRT-10.14.1.48"
export OPENVINO_DIR="$DEPENDENCY_ROOT/openvino_2025.2.0"
export GGML_DIR="$DEPENDENCY_ROOT/ggml"
export TVM_DIR="$DEPENDENCY_ROOT/tvm"
export MIGRAPHX_ROOT="/opt/rocm"
export CACTUS_DIR="$DEPENDENCY_ROOT/cactus"
export LLAMACPP_DIR="$DEPENDENCY_ROOT/llamacpp"
export EXECUTORCH_DIR="$HOME/dependencies/executorch"
export LD_LIBRARY_PATH="\
$ONNX_RUNTIME_DIR/lib:\
$LIBTORCH_DIR/lib:\
$TENSORFLOW_DIR/lib:\
$TENSORRT_DIR/lib:\
$OPENVINO_DIR/runtime/lib/intel64:\
$GGML_DIR/lib:\
$TVM_DIR/build:\
$MIGRAPHX_ROOT/lib:\
$CACTUS_DIR/lib:\
$LLAMACPP_DIR/lib:\
$EXECUTORCH_DIR/lib:\
$LD_LIBRARY_PATH"
```
<!-- /GEN:env-variables -->

```bash
export PATH="$OPENVINO_DIR/bin:$TVM_DIR/bin:$PATH"
export PYTHONPATH="$TVM_DIR/python:$PYTHONPATH"
```

## Supported Platforms

### Linux (Ubuntu/Debian) — x86_64
- All backends except CACTUS (ARM NEON intrinsics — ARM64 only).
- Recommended Ubuntu version: 24.04 (required for ExecuTorch).

### Linux (Ubuntu/Debian) — ARM64 (aarch64)
- All backends including CACTUS.
- Tested targets: Jetson Orin, Raspberry Pi 5.

### Linux (CentOS/RHEL/Fedora)
- Basic support; uses yum/dnf. May require additional configuration.

### Windows
- Not supported.

## Manual Installation

Use the setup scripts when possible. The entries below cover cases that require
extra steps or have no automated installer.

**OpenCV DNN** — system package, no setup script needed:

```bash
sudo apt-get install -y libopencv-dev libopencv-contrib-dev
```

**ONNX Runtime**:
```bash
./scripts/setup_onnx_runtime.sh
```

**LibTorch**:
```bash
./scripts/setup_libtorch.sh
```

**GGML**:
```bash
./scripts/setup_ggml.sh
```

**TensorRT** — manual download required (NVIDIA login):
1. Download from [NVIDIA Developer](https://developer.nvidia.com/tensorrt).
2. Extract to `$HOME/dependencies/TensorRT-<VERSION>`.
3. Ensure CUDA is installed.
4. Run: `./scripts/setup_tensorrt.sh`

**OpenVINO**:
1. Download from [Intel Developer Zone](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html).
2. Extract to `$HOME/dependencies/openvino_<VERSION>`.
3. Run: `./scripts/setup_openvino.sh`

**TensorFlow**:
1. Run: `./scripts/setup_libtensorflow.sh`
2. Alternative pip path: `./scripts/setup_tensorflow_pip.sh`

**ExecuTorch** — no pre-built binaries; built from source:

```bash
./scripts/setup_executorch.sh [--install-dir <path>]
```

Or use the Docker image (recommended — handles all Python build deps):

```bash
docker build --rm -t neuriplo:executorch -f docker/Dockerfile.executorch .
```

> **Note**: do **not** delete the `cmake-out` build directory after installing.
> `ExecuTorchTargets.cmake` references build-tree paths that must remain
> accessible when neuriplo is configured.

**Cactus** (ARM64 only):

```bash
./scripts/setup_cactus.sh [--install-dir <path>]
# or build the Docker image:
./scripts/build_cactus.sh
```

**llama.cpp**:

```bash
./scripts/setup_llamacpp.sh [--install-dir <path>]
```

**MIGraphX** — requires ROCm:

```bash
./scripts/setup_migraphx.sh
```

**TVM** — built from source:
```bash
./scripts/setup_tvm.sh
```
For full build options see [TVM Backend Build Guide](TVM_BUILD_GUIDE.md) and
the [upstream installation guide](https://tvm.apache.org/docs/install/from_source.html).

## Troubleshooting

**Missing dependency after setup**:
```bash
./scripts/setup_dependencies.sh --backend <BACKEND_NAME> --force
```

**Version conflict** — override in CMake:
```bash
cmake -B build -DDEFAULT_BACKEND=LIBTENSORFLOW -DTENSORFLOW_VERSION=2.18.0
```

**CUDA not found**:
```bash
nvcc --version
nvidia-smi
```

**ExecuTorch cmake error** (`extension_evalue_util` not found):
The `cmake-out` directory from the ExecuTorch source build was deleted.
Re-run `setup_executorch.sh` or rebuild the Docker image — and do not clean
the build directory afterwards.

**Cactus fails on x86_64**:
Expected — Cactus requires an ARM64 host. Use a Jetson Orin or Raspberry Pi 5,
or run the Docker image on an ARM64 machine.

**Validation error message**:
```
[ERROR] TensorFlow not found at /home/user/dependencies/tensorflow
Please ensure the inference backend is properly installed or run the setup script.
```

## Testing Integration

### Local CI simulation

Replay any CI job locally before pushing:

```bash
act push --job build-executorch --dryrun   # inspect resolved steps
act push --job build-executorch --verbose  # full run
```

See [LOCAL_CI.md](LOCAL_CI.md) for installation and per-job examples.

### Automated testing

```bash
./scripts/test_backends.sh --backend <BACKEND_NAME>   # single backend
./scripts/test_backends.sh                             # all backends
./scripts/run_complete_tests.sh                        # full suite + report
./scripts/validate_test_system.sh                      # preflight check
```

### Test models per backend

<!-- GEN:test-models-table -->
| Backend | Model format | How it is obtained |
|---|---|---|
| OpenCV DNN | ONNX, Darknet | `scripts/setup_test_models.sh` |
| ONNX Runtime | ONNX | `scripts/model_downloader.py` |
| LibTorch | TorchScript (.pt) | `backends/libtorch/test/generate_model.sh` |
| TensorFlow C++ | SavedModel | auto-generated at test runtime (Keras ResNet-50) |
| TensorRT | TensorRT engine (.engine) | converted from ONNX at test time |
| OpenVINO | IR (.xml / .bin) | `backends/openvino/test/generate_model.sh` |
| GGML | quantized GGML | `scripts/convert_to_ggml.sh` |
| Apache TVM | compiled TVM module | `backends/tvm/test/generate_model.sh` |
| MIGraphX | ONNX only | `backends/migraphx/test/generate_model.sh` |
| Cactus | GGUF | downloaded by Dockerfile or mock fallback |
| llama.cpp | GGUF | downloaded by Dockerfile or mock fallback |
| ExecuTorch | .pte | `backends/executorch/test/export_executorch_classifier.py` |
<!-- /GEN:test-models-table -->

## Contributing

When adding new inference backends, use
[Adding an Inference Backend](ADDING_BACKEND.md) as the checklist. Keep the
backend ID and CMake metadata in `cmake/BackendRegistry.cmake`; this document
should only include backend-specific operational notes that users need at setup
time.

## Available Scripts

### Dependency setup

<!-- GEN:setup-scripts-table -->
| Script | Backend / purpose |
|---|---|
| `setup_dependencies.sh` | Unified dispatcher — delegates to the script below for the chosen `--backend` |
| _(none)_ | OpenCV DNN — system package — no script needed |
| `setup_onnx_runtime.sh` | ONNX Runtime |
| `setup_libtorch.sh` | LibTorch |
| `setup_libtensorflow.sh` | TensorFlow C++ — alternative pip path via `setup_tensorflow_pip.sh` |
| `setup_tensorrt.sh` | TensorRT — manual NVIDIA download required first |
| `setup_openvino.sh` | OpenVINO |
| `setup_ggml.sh` | GGML |
| `setup_tvm.sh` | Apache TVM — builds from source — see TVM_BUILD_GUIDE.md |
| `setup_migraphx.sh` | MIGraphX — installs from ROCm apt repo — requires ROCm |
| `setup_cactus.sh` | Cactus — **ARM64 only** — fails fast on x86_64 |
| `setup_llamacpp.sh` | llama.cpp |
| `setup_executorch.sh` | ExecuTorch — builds from source — do not delete cmake-out after install |
| `build_cactus.sh` | Build the Cactus Docker image (ARM64 only) |
<!-- /GEN:setup-scripts-table -->

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
