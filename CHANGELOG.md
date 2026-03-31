# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
