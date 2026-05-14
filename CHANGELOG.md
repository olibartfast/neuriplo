# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/olibartfast/neuriplo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/olibartfast/neuriplo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/neuriplo/releases/tag/v0.1.0
