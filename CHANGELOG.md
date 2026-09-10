# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- `TensorRT`: device buffers are keyed by tensor name instead of assumed
  all-inputs-first positional indexing. TensorRT does not guarantee that the
  engine's I/O enumeration groups inputs before outputs, while every buffer
  read, address assignment, and importantly the new `get_infer_results_raw()`
  copy did. Interleaved-enumeration engines copied from another tensor's
  device buffer; these engines now read from their own allocations. Engines
  with conventional input-then-output enumeration are unaffected, and all six
  TensorRT GPU tests pass unchanged.
- `DALI`: advertise external inputs as Int8/Bool where applicable rather than
  silently falling back to Float32, and reject input types with no metadata
  representation instead of advertising them as Float32.
- `DALI`: unnamed outputs are addressed positionally (`output0`, ...) in
  metadata instead of fabricating `preprocessed`/`IMAGE_SHAPE` names for any
  pipeline. Use `|outnames=` for semantic names. Consumers that matched
  `preprocessed`/`IMAGE_SHAPE` in unspecified-pipeline metadata must switch to
  `outnames=` or positional names.
- `DALI`: reject rank-1 dynamic inputs whose byte count is not a multiple of
  the element type's size, and reject declared shapes that overflow a signed
  element count before DALI receives them.
- `setup_dependencies.sh`: the DALI install honors `-r/--root` by forwarding
  the dependency root into `setup_dali.sh`; generated env blocks add `$DALI_DIR`
  to the shared-library search path, where `setup_dali.sh` actually places
  `libdali.so`, rather than a nonexistent `lib/` subdirectory.
- `Readme.md`: DALI output 1 documents `(H, W)` matching the generator and
  pipeline implementation, and the pipeline-generation command uses the
  NVIDIA image rather than an impossible-to-import extracted wheel layout.

## [0.9.0] - 2026-09-10

### Added
- Spec-driven project constitution under `specs/`: mission, technical
  boundaries, a status-bearing roadmap, and the three-document feature-packet
  workflow for new or actively changing phases.
- `DALI` backend: hosts a serialized NVIDIA DALI pipeline in-process through
  the DALI C API, pure C++ at inference time. Not an inference engine -- it
  fills the same `InferenceInterface` slot so a serving pipeline can chain
  GPU preprocessing (nvJPEG decode, letterbox, normalize) ahead of a model
  such as a TensorRT engine. Feed it encoded image bytes; it returns the
  preprocessed tensor plus an `IMAGE_SHAPE` output carrying the source
  dimensions, which downstream postprocessing needs to map results back onto
  the original frame. Enable with `-DDEFAULT_BACKEND=DALI` (or add `DALI` to
  `NEURIPLO_BACKENDS`) and `-DDALI_DIR=<dir>`; pipelines are authored offline
  with `export/dali/generate_yolo_pipeline.py`, which reproduces the
  neuriplo-tasks YOLO letterbox exactly (centered padding, zero fill,
  `antialias=False`). Both `libdali.so` and `libdali_operators.so` are linked,
  and `daliInitOperators()` is called at first use -- without either, every
  pipeline fails at run time with `No schema found for operator
  "decoders__Image"`.
- `DALI` backend: postprocessing pipelines. The backend now discovers whatever
  external sources a pipeline declares and feeds them in order, so a
  postprocessing stage that consumes model outputs works the same way as
  preprocessing. GPU preprocessing, inference, and postprocessing can be chained,
  but the current byte-buffer interface still copies outputs to host memory
  between stages; this is not a zero-copy device-to-device interface.
  `model_path` carries optional suffixes: `|plugin=<lib.so>` for custom DALI
  operator plugins, `|out=CxHxW` for the declared output shape, and
  `|outnames=A,B,...` for named positional outputs. Pipelines are serialized
  inside the NVIDIA container (`export/dali/generate_pipelines.sh`) rather
  than a host virtualenv, since a serialized pipeline and its operator plugin
  are tied to one DALI version. The source-dimensions tensor is now INT64
  (height, width).
- Windows/MSVC CI coverage: an OPENCV_DNN build job, an ONNX_RUNTIME build job
  with `ctest`, and a standalone public-headers project built with glog only,
  where OpenCV is never installed, proving the public headers compile without
  OpenCV and without relying on it for transitive `std` headers.

### Changed
- `TensorRT` backend: implements `get_infer_results_raw()` instead of
  inheriting the default, which routed through `get_infer_results()` and
  materialised one 16-byte `std::variant` per output scalar before
  re-serialising it. For a YOLO26m-seg engine that was ~2.6M scalars, roughly
  42 MB of variant vector built and walked twice per inference. Outputs are now
  copied device-to-host straight into the destination byte buffer. Measured on
  an RTX 3060 Laptop through neuriplo-kserve-runtime, server-side, same engine
  and same frames: an all-CPU pre/post ensemble went 148.0 -> 123.5 ms, GPU
  preprocessing 123.6 -> 78.7 ms, and full GPU pre+post 70.9 -> 26.8 ms
  (2.65x). Detections are unchanged -- a 50-frame agreement run reproduced the
  recorded baseline exactly (701 reference, 739 candidate, 641 matched).
- `TensorRT` backend: the per-inference CUDA stream is RAII-owned and
  explicitly synchronised after `enqueueV3` rather than leaking on the binding
  error paths and relying on the legacy default stream to synchronise for it.
- `TensorRT` backend: reported tensor shapes keep the batch dimension, for
  inputs and outputs alike. The backend previously dropped it, so a model
  served through TensorRT advertised `[3,H,W]` where the ONNX Runtime backend
  advertised `[1,3,H,W]`, and any client valid against one backend was
  rejected with HTTP 400 "invalid shape for input" against the other. The
  engine's own leading dimension is authoritative; when it is dynamic, the
  configured batch size is reported. Consumers must stop prepending an extra
  batch dimension to reported TensorRT shapes and update rank assumptions;
  constructor `input_sizes` still exclude the batch dimension.
- Public headers no longer include OpenCV, even transitively. OpenCV belongs
  to the `OPENCV_DNN` backend alone: it is linked only from that backend's
  CMake branch, the test suite builds its inputs without OpenCV, and the
  includes the headers used to lean on OpenCV for are provided explicitly.
  Consumers must now include and link OpenCV directly when they use it.
- `OPENCV_DNN` backend: builds against OpenCV 4.6+ and 5.x. OpenCV 5 removes
  legacy importers including Darknet, Caffe, and Torch; use OpenCV 4.x or
  convert affected models to ONNX. Its default DNN engine is CPU-only, so
  CUDA-enabled OpenCV 5 builds reject GPU requests rather than silently
  ignoring them; pass `use_gpu=false` or use OpenCV 4.x for CUDA inference.
- Dependency validation checks installed-version metadata where available
  rather than trusting directory names, and warns about drift from the
  `versions.env` pin. Source-built backends use a `neuriplo-version.txt`
  stamp written only after a successful build and install, so bumping a pin
  no longer silently reuses a stale installation. GGML, llama.cpp, Cactus,
  and ExecuTorch setup stage new installs before replacing the old ones;
  this is not a transactional-install guarantee for every setup script.
- LibTorch setup picks the CUDA or CPU build deliberately: it keeps the family
  of an existing installation, otherwise selects the newest CUDA build
  published for the pinned PyTorch release that the local driver supports,
  when CUDA is wanted. With no existing variant and no driver capability
  reported by `nvidia-smi`, it selects CPU even if a CUDA toolkit is installed.
  An existing CUDA installation with no detected driver instead requires an
  explicit decision; failed CUDA download probes do not silently select CPU.
  Override with `LIBTORCH_VARIANT` (a CPU-only build runs on CPU regardless
  of `use_gpu`).

### Fixed
- `DALI`: reject serialized pipelines with maximum batch sizes other than one,
  decode INT32/INT64 legacy results without loss, and report every output's
  actual datatype and batch-inclusive shape. Output-name counts are checked.
  Older pipelines without datatype declarations require a successful inference
  before metadata retrieval; regenerate metadata-first deployments with explicit
  `output_dtype` declarations. The preprocessing generator supplies them.
- `DALI` integration test: expect the current preprocessing generator's INT64
  `(height, width)` output rather than the obsolete three-element INT32 shape.
- `OPENCV_DNN` integration test: pass the fixture's input dimensions when
  constructing the backend, so a valid model does not fall back to mock
  inference because the dimensions were omitted.
- `DALI` backend: reject an external input whose declared shape needs more
  bytes than the supplied buffer holds. `daliSetExternalInput` takes no
  destination length, so this previously became an out-of-bounds read inside
  the library with no diagnostic. The output copy is likewise redzone-checked,
  because `daliOutputCopy` has the same no-length signature and a
  `daliTensorSize` that under-reported would corrupt the heap silently.
- `ONNX_RUNTIME`: an installed runtime older than the build headers now raises
  a named `ModelLoadException` instead of faulting on the first ONNX Runtime
  call past every catch clause (the C++ API table pointer is a null deref
  before then). The Windows runtime DLLs are also staged where the loader
  looks first, so version mismatches are rejected rather than depending on
  PATH order.
- `TensorRT`: the logger outlives the runtime, engine and execution context;
  `createInferRuntime()` stores only a reference, and a function-local logger
  died with the function that created it, so the next TensorRT diagnostic
  called through a freed vtable. `TRTInfer` is also non-copyable: its
  destructor frees every device buffer, and the implicit copy would have
  freed them twice.
- Windows: a test executable that fails before it can report (missing DLL,
  fault in static initialisation) no longer hangs the CI job for hours --
  error dialogs are suppressed and the test step runs under a bounded
  timeout.

## [0.8.0] - 2026-06-14

### Added
- LiteRT now registers custom TensorFlow Lite kernels required by models that
  depend on non-builtin TFLite operators.

### Fixed
- OpenVINO and ExecuTorch backends handle ecdet dual-input models without
  hard-coded model-name assumptions, keeping backend setup model-agnostic.

## [0.7.0] - 2026-06-13

### Added
- Tensor datatype metadata: `TensorDtype` enum and datatype fields on
  `InferenceMetadata`; ONNX Runtime and TensorRT backends report real tensor
  datatypes from model metadata instead of assuming float32.
- Plugin metadata ABI v2: `neuriplo_layer_info_t` now carries
  `element_type`, so dlopen plugins preserve non-FP32 tensor datatypes across
  the host boundary (`NEURIPLO_PLUGIN_ABI_VERSION` bumped to 2).

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
- Library roadmap (`specs/roadmap.md`) and the ORT execution-provider guide
  (`docs/ORT_EXECUTION_PROVIDERS.md`).

### Changed
- `setup_inference_engine` no longer lets vendor exceptions (e.g.
  `cv::Exception` from an unreadable or unparseable model) propagate to the
  caller: every load failure is logged and surfaces as a `nullptr` return,
  matching the contract already used for `InferenceException`. Consumers that
  caught vendor exception types around engine setup must switch to checking
  the returned pointer (neuriplo-infer adapted in its v0.6.1).

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
  `docs/CODE_QUALITY.md` plus `docs/ARCHITECTURE.md`.

### Changed
- `setup_inference_engine` now constructs backends through the Abstract Factory
  while preserving its existing signature and `unique_ptr<InferenceInterface>`
  return type (cross-repo contract with neuriplo-infer unchanged).
- Documentation now references the renamed sibling repositories
  (`vision-inference` → `neuriplo-infer`, `vision-core` → `neuriplo-tasks`) in
  `Readme.md` and `docs/ARCHITECTURE.md`.

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

[Unreleased]: https://github.com/olibartfast/neuriplo/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/olibartfast/neuriplo/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/olibartfast/neuriplo/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/olibartfast/neuriplo/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/olibartfast/neuriplo/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/olibartfast/neuriplo/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/olibartfast/neuriplo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/olibartfast/neuriplo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/olibartfast/neuriplo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/neuriplo/releases/tag/v0.1.0
