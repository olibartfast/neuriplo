# Validation — NERT backend GPU selection

Defined before implementation.

## Checks

- [V-A1] `NEURIPLO_BACKENDS=NERT` CPU-only build; `NertInferTest`
  green (migration changes nothing observable).
- [V-B1] `NEURIPLO_NERT_CUDA=ON` configures (needs nvcc) and builds;
  flag OFF builds exactly as before (no CUDA in link closure).
- [V-C1] With the flag: factory GPU → Ready; EngineOptions + legacy
  GPU → non-null Ready; typed_io GPU-vs-CPU parity (typed + raw).
- [V-C2] Without the flag: RejectsGpuRequests / null tests unchanged.
- [V-C3] Non-CUDA graph on GPU (ResNet w/o cuDNN rows... Conv absent)
  surfaces `ModelLoadException`, not silent CPU.
- [V-D] Docs updated; diff audit = packet paths only.

## Acceptance

```bash
cmake -S . -B build-nert-cpu -DNEURIPLO_BACKENDS=NERT -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-nert-cpu --parallel "$(nproc)" && ctest --test-dir build-nert-cpu -R NertInfer --output-on-failure
cmake -S . -B build-nert-cuda -DNEURIPLO_BACKENDS=NERT -DBUILD_INFERENCE_ENGINE_TESTS=ON -DNEURIPLO_NERT_CUDA=ON -DFETCHCONTENT_SOURCE_DIR_NERT=/home/oli/repos/nert && cmake --build build-nert-cuda --parallel "$(nproc)" && ctest --test-dir build-nert-cuda -R NertInfer --output-on-failure
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | pending | — | — |

## Deviations

- `implementer` dispatch unavailable (harness); planner implements directly.
