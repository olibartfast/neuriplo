# Plan — NERT backend GPU selection

Work in the neuriplo tree. `implementer` dispatch unavailable (harness);
planner implements directly.

## Phase A — pin + migration (CPU behavior preserved)

- `versions.env`: `NERT_VERSION=b4c284d38cb4510f95f733ccf10a8663d8d4a955`.
- `backends/nert/src`: `engine/` → `nert/` includes, `engine::` →
  `nert::` (NertInfer.hpp/.cpp). No logic change; full CPU test leg
  green proves the migration.

## Phase B — CUDA flag + device selection

- `cmake/Nert.cmake`: `option(NEURIPLO_NERT_CUDA ... OFF)`; when ON,
  `NERT_ENABLE_CUDA=ON` (CACHE FORCE pattern like `NERT_BUILD_TESTS`)
  + `add_compile_definitions(NEURIPLO_NERT_CUDA)`.
- `NertInfer` ctor: `#ifdef NEURIPLO_NERT_CUDA` device pick
  (`use_gpu ? CudaDevice() : CpuDevice()`), else keep the CPU-only
  throw. Factory drops its throw and forwards `use_gpu`.

## Phase C — tests + docs

- `NertInferTest`: `#ifdef NEURIPLO_NERT_CUDA` GPU variants —
  factory GPU constructs Ready; EngineOptions/legacy GPU return
  non-null Ready; typed_io GPU-vs-CPU parity (raw + typed). `#else`
  keeps the current reject/null tests.
- `docs/backends.yaml` + `Readme.md` NERT GPU note.

## Validation

Per-check evidence in
`specs/2026-10-18-nert-gpu-select/validation.md`. Acceptance:
(1) CPU-only configure+build, `NertInferTest` green;
(2) `NEURIPLO_NERT_CUDA=ON` + local NERT source dir, `NertInferTest`
green on the RTX 3060 (GPU tests execute, ResNet stays CPU);
(3) format/lint per repo convention if any; (4) diff audit.
