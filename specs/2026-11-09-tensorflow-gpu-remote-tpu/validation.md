# TensorFlow GPU execution - validation

Defined before implementation. A check that cannot run is recorded as not
executed, not as a pass.

## Group 0 - runtime probe

- [x] [T-1/T-2] `/tmp/opencode/tf_device_probe` was built against the pinned
      `TENSORFLOW_DIR` and run on the GPU-free host. `ListDevices()` returned
      only `CPU` in all three configs. CUDA registration failed at `cuInit`
      with `UNKNOWN ERROR (303)`, so no GPU device existed to expose.
- [x] [T-3] Decision: the pinned CPU runtime cannot demonstrate GPU execution on
      this host. The GPU image is opt-in and builds from the pinned
      `nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04` base (CUDA 12.6.1,
      cuDNN 9.3.0), staying within TensorFlow 2.19's tested CUDA 12.5 /
      cuDNN 9.3 line. This matrix is a review item for M-1, not a verified live
      GPU result.

## Automated

- [x] [V-1] -> [R-5], [R-6]: CPU-only configure, build, and full CTest suite
      pass with no new warnings:
      `cmake -S . -B build-tf-cpu -DNEURIPLO_BACKENDS=LIBTENSORFLOW -DBUILD_INFERENCE_ENGINE_TESTS=ON`
      then `cmake --build build-tf-cpu --parallel "$(nproc)"` and
      `ctest --test-dir build-tf-cpu --output-on-failure`.
- [x] [V-2] -> [R-2], [R-9]: On a GPU-free process, `use_gpu=true` returns
      `nullptr` from `setup_inference_engine` and logs an error that names the
      GPU request. No output tensor is produced.
- [ ] [V-3] -> [R-3], [R-4]: On a GPU-capable host, `use_gpu=false` lists no
      GPU devices in the session and `is_gpu_available()` is false.
- [ ] [V-4] -> [R-1], [R-4]: On a GPU-capable host, `use_gpu=true` lists at
      least one GPU device, `is_gpu_available()` is true, and the log records
      the device name.
- [ ] [V-5] -> [R-10]: Live GPU parity for the TensorFlow classifier: same
      SavedModel, same input, CPU and GPU top-5 classes agree and scores agree
      within a recorded tolerance. Record exact commands and output.
- [ ] [V-6] -> [R-7]: GPU image builds from the pinned base; `ldd` on the
      binary resolves the TensorFlow and CUDA/cuDNN libraries. CPU image still
      builds and runs its existing test.
- [x] [V-7] -> [R-8]: `python3 scripts/gen_backend_docs.py --check` passes after
      regenerating the docs, and documentation links resolve.
- [x] [V-8] -> [R-6]: Consumer C ABI symbol/layout checks and plugin ABI tests
      pass unchanged; no exported symbol or struct layout diff.
- [x] [V-9a] -> [R-5]: Format and lint checks pass (`git diff --check`, shell
      syntax, and YAML parse of `ci.yml` / `docs/backends.yaml`).
- [ ] [V-9b] -> [R-9]: The optional live TensorFlow GPU test was not executed;
      it needs a GPU host and reports `GTEST_SKIP` when no device is present.

## Manual

- [ ] [M-1] Review the GPU image's CUDA/cuDNN/driver matrix against the pinned
      TensorFlow wheel's tested configuration. Decision under review:
      `nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04` (CUDA 12.6.1, cuDNN 9.3.0)
      against TensorFlow 2.19's tested CUDA 12.5 / cuDNN 9.3.
- [ ] [M-2] Inspect the CPU and GPU rendered outputs for the classifier run;
      record the output paths and top-5 values.
- [ ] [M-3] Confirm no TPU/PJRT dependency or claim leaked into this packet.

## Acceptance Commands

```bash
cmake -S . -B build-tf-cpu \
  -DNEURIPLO_BACKENDS=LIBTENSORFLOW \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build-tf-cpu --parallel "$(nproc)"
ctest --test-dir build-tf-cpu -R TensorFlow --output-on-failure

# GPU host only
nvidia-smi
ctest --test-dir build-tf-cpu -R TensorFlowGpu --output-on-failure
```

The GPU image build command is recorded here once Group 0 selects the base
image. It must not require a GPU at build time.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | CPU build + tests | pass | 2026-11-09 | 4/4 TensorFlow CTest passed |
| V-2 | GPU-negative | pass | 2026-11-09 | Covered by the 4/4 TensorFlow CTest run: `use_gpu=true` without a GPU throws `ModelLoadException` naming the request, no CPU output |
| V-3 | CPU hides GPU | not executed | - | needs GPU host |
| V-4 | GPU device visible | not executed | - | needs GPU host |
| V-5 | CPU/GPU parity | not executed | - | needs GPU host and model |
| V-6 | GPU image build | not executed | - | full `--target final` image build is now the CI job; pending CI execution on a GPU-free runner, base image `nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04`, M-1 review |
| V-7 | Docs check | pass | 2026-11-09 | `python3 scripts/gen_backend_docs.py --check` exit 0 after regeneration |
| V-8 | ABI checks | pass | 2026-11-09 | consumer C ABI symbol/layout and plugin ABI tests pass unchanged |
| V-9a | Format/lint | pass | 2026-11-09 | `git diff --check`, `bash -n` on the setup script, YAML parse of `ci.yml` and `docs/backends.yaml` |
| V-9b | Optional live GPU test | not executed | - | needs a GPU host; absent device reports `GTEST_SKIP` rather than failing |

## Deviations

None. The live GPU result (V-3/V-4/V-5) is explicitly unattested: no GPU host
was available, so it is recorded as not executed rather than inferred as a pass.
