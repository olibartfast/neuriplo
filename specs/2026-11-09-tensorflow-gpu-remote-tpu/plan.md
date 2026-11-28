# TensorFlow GPU execution - plan

**Packet:** `2026-11-09-tensorflow-gpu-remote-tpu`
**Branch:** `feature/tensorflow-gpu-remote-tpu`

Each group ends in an observable result. Groups 0 and 1 are prerequisites for
any GPU image work.

## Group 0 - Probe the pinned runtime

- [T-1] Build a minimal TensorFlow C++ probe against the pinned
  `TENSORFLOW_DIR` and call `Session::ListDevices()` with default options.
- [T-2] Repeat with a GPU host and with `ConfigProto.device_count["GPU"] = 0`.
  Record device names, TensorFlow version, driver, CUDA, and cuDNN.
- [T-3] Decide whether the pinned pip wheel is sufficient or whether the GPU
  image must come from a different TensorFlow distribution. Record the decision
  in `validation.md` before changing the Dockerfile.

## Group 1 - Backend behavior

- [T-4] Replace the unused `use_gpu` parameter in
  `backends/libtensorflow/src/TFDetectionAPI.cpp` with explicit session setup:
  disable GPU for `false`, enable growth and soft placement for `true`.
- [T-5] After `LoadSavedModel`, inspect `Session::ListDevices()`, set the
  backend's GPU availability from the actual list, and log requested versus
  actual placement.
- [T-6] If `use_gpu=true` yields no GPU device, throw `ModelLoadException` with
  a message naming the request and the missing device. Do not fall back.
- [T-7] Keep the metadata, input conversion, output extraction, timing, and
  lifecycle code paths unchanged.

## Group 2 - Tests

- [T-8] Add a GPU-negative test: `use_gpu=true` on a GPU-free process fails
  with the expected error and no CPU output is produced.
- [T-9] Add a CPU device test: `use_gpu=false` exposes no GPU devices even when
  the host has one.
- [T-10] Add an optional live GPU test, skipped when no GPU is available, that
  compares CPU and GPU top-5 results for the same SavedModel.
- [T-11] Keep all existing TensorFlow tests green without a GPU.

## Group 3 - Build and documentation

- [T-12] Add an opt-in GPU image or build argument. The CPU
  `Dockerfile.libtensorflow` remains the default and must still build.
- [T-13] Validate the GPU image's CUDA/cuDNN libraries with `ldd`; document
  `--gpus all` and the host driver requirement.
- [T-14] Update `scripts/setup_tensorflow_pip.sh` only as required by the probe:
  validate the GPU runtime libraries, or document the host/image prerequisite.
- [T-15] Update `docs/backends.yaml` only through fields the generator supports:
  keep `gpu: false` if it means "GPU required", and add the optional-GPU note
  where the generated backend table can carry it. Regenerate
  `docs/DEPENDENCY_MANAGEMENT.md` and update the TensorFlow setup/Docker
  documentation. Do not claim TPU support.
- [T-16] Add a GPU image build job to CI. The job builds only; a live run needs
  a GPU runner or manual evidence.

## Group 4 - Downstream handoff

- [T-17] Publish the GPU image tag, build command, and required driver/CUDA
  combination to the `neuriplo-infer` packet.
- [T-18] Record the CPU/GPU parity command and expected top-5 values for the
  TensorFlow classifier handoff.

## Acceptance

The packet is done when a CPU-only build and full test suite pass, the
GPU-negative path is executable, the GPU image builds, and a recorded live GPU
run shows the same model executing on a GPU with no CPU fallback. If no GPU
runner is available, the live run stays explicitly unattested; it is not a pass
by inference.
