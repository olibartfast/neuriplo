# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.6.0] - 2026-06-12

### Added
- Multi-backend builds: the `NEURIPLO_BACKENDS` CMake list compiles several
  backends into one library, with runtime lookup through
  `BackendRuntimeRegistry` (`get_registered_backends`,
  `find_backend_registration`) and a new `EngineOptions` overload of
  `setup_inference_engine` for explicit backend selection. Single-backend
  `DEFAULT_BACKEND` builds are unchanged.
- dlopen backend plugins behind a stable C ABI
  (`include/neuriplo/plugin_abi.h`): per-backend
  `libneuriplo_backend_<id>.so` targets via `NEURIPLO_PLUGIN_BACKENDS`, a
  generic plugin shim over existing backend factories, and an
  `RTLD_LOCAL` host loader with ABI version checks and
  `NEURIPLO_PLUGIN_DIR` discovery. Plugin dependency conflicts (e.g.
  llama.cpp + GGML) are isolated per plugin.
- Raw typed-buffer output API: `RawOutputTensor{dtype, bytes, shape}` and
  `InferenceInterface::get_infer_results_raw`, letting consumers receive
  outputs as typed contiguous bytes instead of per-element `TensorElement`
  variants. ONNX Runtime and OpenCV-DNN override the raw path; the default
  implementation adapts `get_infer_results()` so all other backends keep
  working unchanged.
- Optional ccache support to speed up non-release builds.
- Library roadmap (`ROADMAP.md`) and the ORT execution-provider plan
  (`docs/plans/ort-execution-providers.md`).

### Fixed
- GGML backend frees its backend handle on constructor failure.
- Hardened backend failure paths and plugin builds when consumed as a
  subdirectory.
- CI reliability: reclaim runner disk before backend image builds, retry
  Docker Buildx bootstrap on registry flakes, and suppress an ONNX Runtime
  internal leak in LeakSanitizer runs.

## [0.5.0] - 2026-06-07

### Added
- Design-pattern-driven backend architecture: Abstract Factory per backend
  (`IBackendRuntimeFactory` plus a `*RuntimeFactory` for each of the 13 backends),
  a `BackendRuntimeRegistry` for runtime factory lookup, and a `ModelRunner` bridge
  over `InferenceInterface`.
- Backend decorators (`CachingBackend`, `LoggingBackend`, `ProfilingBackend`,
  `QuantizedBackend`) layered on a shared `BackendDecorator` base.
- Explicit backend lifecycle/state model (`BackendState`) wired across all backends,
  with lifecycle hooks added to `InferenceInterface`.
- Tensor-conversion abstractions (`ITensorConverter`, `HostTensorConverter`,
  `IAllocator`) and a dedicated patterns test suite (`PatternsTest.cpp`).
- Local code-quality tooling: clang-format, clang-tidy, cppcheck, and sanitizer
  scripts under `scripts/quality/`, pre-commit/pre-push git hooks, and
  `docs/CODE_QUALITY.md` plus `docs/REFACTOR_DESIGN_PATTERNS.md`.

### Changed
- `setup_inference_engine` now constructs backends through the Abstract Factory
  while preserving its existing signature and `unique_ptr<InferenceInterface>`
  return type (cross-repo contract with neuriplo-infer unchanged).
- Documentation now references the renamed sibling repositories
  (`vision-inference` → `neuriplo-infer`, `vision-core` → `neuriplo-tasks`) in
  `Readme.md` and `docs/REFACTOR_DESIGN_PATTERNS.md`.

### Fixed
- Backend load failures now set a `Failed` state and throw `ModelLoadException`
  instead of calling `std::exit(1)`, making failures observable to callers.
- Registered `TVMRuntimeFactory` in `setup_inference_engine` and corrected
  `ModelRunner` Failed-state handling and decorator cache keys.

## [0.4.0] - 2026-05-28

### Added
- LiteRT backend integration for `.tflite` FlatBuffer models, including CMake
  registration, setup script, Docker/CI coverage, and GTest smoke coverage.

### Changed
- Backend test orchestration, reports, validation scripts, and dependency docs
  now include ExecuTorch and LiteRT in the supported backend matrix.

### Fixed
- LiteRT backend now transposes NCHW input data to NHWC before model inference,
  matching the pattern used by the TensorFlow backend. Without this, vision
  models silently produce garbage because channel and spatial dimensions are
  swapped but byte counts match.

## [0.3.0] - 2026-05-21

### Added
- Configurable ONNX Runtime execution providers via `NEURIPLO_ORT_EP`, with CMake
  build gates for TensorRT, OpenVINO, MIGraphX, QNN, XNNPACK, CANN, and Vitis AI EPs
- ExecuTorch delegate selection for `xnnpack` and `portable`, including matching
  `.pte` export flow and delegate documentation
- ExecuTorch v1.2.0 backend for PyTorch edge inference
- llama.cpp multimodal VLM support via libmtmd
- MIGraphX AMD ROCm graph inference backend
- Cactus GGUF-native text generation backend (ARM64 / Jetson support)
- Auto-generated backend list sections from `backends.yaml`
- Setup scripts for Cactus, llama.cpp, and MIGraphX (`scripts/setup_*.sh`)
- `TROUBLESHOOTING.md` with CI/inference debugging patterns
- Pre-commit act hook and pre-push clang-format / docs-sync checks

### Changed
- Centralized backend registry metadata in `cmake/BackendRegistry.cmake`
- Complete rewrite of `docs/DEPENDENCY_MANAGEMENT.md`

### Fixed
- llama.cpp backend lifecycle, template loading, and test stability
- llama.cpp chat-template API migration to b9049 / b9085
- `llama_kv_cache_clear` replaced with `llama_memory_clear` after upstream removal
- `-march=native` removed / guarded by architecture to support ARM/aarch64 CI
- ExecuTorch cmake `configure_file` using `COPYONLY` for `generate_model.sh`
- Cactus x86_64 guard and ARM arch-detection build fixes
- Pinned `GGML_VERSION=v0.11.0`, `LLAMACPP_VERSION=b9049`, `CACTUS_VERSION=v1.14`
- libmtmd linking in LLAMACPP cmake and Dockerfile validation
- CI disk-space failures in LibTensorFlow and LibTorch Docker builds

## [0.2.0] - 2026-03-31

### Changed
- Expanded CI coverage across CPU backends and TensorRT-related lint/build checks
- Added canonical `VERSION` and `CHANGELOG.md` release metadata on `develop`

### Fixed
- Sanitizer and `-Werror` build issues across backend test and mock code
- OpenVINO AddressSanitizer failures caused by `libtbbbind` deep-bind conflicts
- TensorRT Docker build handling for non-GPU environments

## [0.1.0] - 2026-03-02

### Added
- Unified inference backend abstraction (`InferenceInterface`)
- Backend implementations: OpenCV DNN, ONNX Runtime, LibTorch, TensorRT, OpenVINO, LibTensorFlow, GGML, TVM
- Centralized backend version management via `versions.env`
- CMake-based backend selection and linking (`SelectBackend`, `LinkBackend`)
- Dependency validation framework (`DependencyValidation.cmake`)
- FindTensorFlow CMake module
- Docker CI with matrix strategy for CPU backends
- Docker build files for all backends
- GTest-based test suite
- Git-flow branch policy enforcement via GitHub Actions

[Unreleased]: https://github.com/olibartfast/neuriplo/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/olibartfast/neuriplo/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/olibartfast/neuriplo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/olibartfast/neuriplo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/olibartfast/neuriplo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/neuriplo/releases/tag/v0.1.0
