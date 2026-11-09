# TensorFlow GPU execution - requirements

**Date:** 2026-11-09
**Branch:** `feature/tensorflow-gpu-remote-tpu` from `develop`
**Status:** Implemented; pending GPU evidence
**Cross-repo packet:** `2026-11-09-tensorflow-gpu-remote-tpu` in
`neuriplo`, `neuriplo-infer`, `neuriplo-tasks`, and `neuriplo-platform`

## Goal

Make the `LIBTENSORFLOW` backend honor the existing `EngineOptions::use_gpu`
contract. A GPU request must execute TensorFlow work on an NVIDIA GPU and must
fail clearly when no usable GPU is present. CPU remains the default and the
existing CPU behavior must not change.

TPU execution is deliberately out of scope for this repository in this packet.
The TPU path is an external KServe V2 service consumed by `neuriplo-infer`; it
does not add a PJRT backend, an XLA dependency, or a new device field here.

## In Scope

- `LIBTENSORFLOW` device selection in `backends/libtensorflow/src/TFDetectionAPI.cpp`.
- Session configuration, actual device inspection, failure behavior, and logging.
- Unit and optional live GPU coverage in `backends/libtensorflow/test/`.
- An opt-in GPU build path: setup script validation, GPU Docker image, and CI
  build coverage. The CPU image and CPU default remain unchanged.
- Backend metadata and generated documentation for optional GPU support.

## Out of Scope

- In-process TPU/PJRT execution, StableHLO model format, `libtpu`, and XLA.
  TPU is remote serving only in the cross-repo packet.
- Replacing `bool use_gpu` with a device/provider enum. The device and fallback
  contract remains `neuriplo` roadmap Phase 3 work.
- Multi-GPU selection, device index flags, MIG, CUDA graphs, or stream tuning.
- Changes to public C++ headers, the consumer C ABI, or the plugin ABI.
- Changes to model input/output layout, datatype, batching, timing, or task
  semantics.
- A TensorFlow version bump. The packet uses the existing `TENSORFLOW_VERSION`
  pin unless the GPU probe proves the pinned wheel cannot expose a GPU device.

## Decisions

- [D-1] Keep the existing `use_gpu` boolean for this packet. The public
  construction contract already carries it; the defect is that the TensorFlow
  backend ignores it.
- [D-2] For `use_gpu=false`, set `ConfigProto.device_count["GPU"] = 0` on the
  session so TensorFlow cannot silently place work on a GPU.
- [D-3] For `use_gpu=true`, enable GPU memory growth and soft placement, then
  inspect `Session::ListDevices()` after load. A request that cannot produce a
  GPU device is a load failure, not a CPU fallback.
- [D-4] Keep the CPU Docker image as the default. Add an opt-in GPU image or
  build argument; do not make every CPU CI job pull CUDA layers.
- [D-5] Report the requested and actual device in logs. Do not use
  `TF_CPP_MIN_LOG_LEVEL` to hide placement failures.

## Assumptions

- [A-1] The pinned `tensorflow==2.19.0` wheel contains GPU kernels and its
  `libtensorflow_cc.so.2` can register GPU devices. Confirm with
  `Session::ListDevices()` before finalizing the image design.
- [A-2] A supported NVIDIA base image provides the CUDA and cuDNN versions that
  the pinned TensorFlow wheel expects.
- [A-3] GPU-free CI can prove the negative path (`use_gpu=true` without a GPU)
  without a hardware runner.

## Open Questions

- [Q-1] Does the pinned pip wheel expose a GPU device through the C++
  `Session` API, or does it only expose GPU through the Python runtime? This
  blocks the image design and must be probed first.
- [Q-2] Is `tensorflow-cpu` preferable in the CPU image to avoid shipping unused
  CUDA code, or does that diverge from the supported setup path? Default: keep
  the current `tensorflow` wheel until image size or policy requires a split.

## Requirements

- [R-1] With `use_gpu=true` and an available GPU, `LIBTENSORFLOW` must execute
  the SavedModel with GPU devices visible to the TensorFlow session.
- [R-2] With `use_gpu=true` and no usable GPU, model load must fail with an
  actionable error and `setup_inference_engine` must return `nullptr`; it must
  not run on CPU.
- [R-3] With `use_gpu=false`, the session must have no visible GPU devices and
  inference must remain CPU-only.
- [R-4] `is_gpu_available()` must reflect the actual session device list, not
  the request flag.
- [R-5] CPU execution, model format, metadata, input/output layout, datatype
  handling, batching, timing, and existing tests must remain unchanged.
- [R-6] The public C++ API, consumer C ABI, plugin ABI, exported symbols, and
  struct layouts must not change.
- [R-7] The GPU build path must be opt-in and reproducible: pinned base image,
  documented CUDA/cuDNN versions, `ldd` evidence, and a CI build job that does
  not require a GPU runner.
- [R-8] `docs/backends.yaml`, generated backend documentation, and dependency
  setup docs must state optional GPU support without claiming TPU support.
- [R-9] Tests must cover the CPU path, the GPU-request-without-GPU path, and an
  optional live GPU parity path. A missing GPU must skip only the live check.
- [R-10] A live GPU gate must compare the same SavedModel on CPU and GPU and
  record top-5 classes, scores, device name, driver, CUDA, cuDNN, and commands.

## Compatibility Contract

The legacy overload

```cpp
setup_inference_engine(model_path, use_gpu, batch_size, input_sizes)
```

keeps its current meaning. `EngineOptions::use_gpu` keeps its current meaning.
No caller has to opt into a new type or a new field for this packet. A caller
that already passes `use_gpu=true` to `LIBTENSORFLOW` gets GPU execution after
this change, which is the intended bug fix.
