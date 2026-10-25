# Requirements — NERT backend GPU selection

Date: 2026-10-18. Neuriplo's NERT adapter hardcodes `CpuDevice()` and
rejects every GPU request, while NERT itself ships 70 CUDA rows. This
packet makes `use_gpu=true` with `backend_id=NERT` construct a CUDA
NERT backend instead of throwing/returning null.

## In scope

- NERT pin `81b6b1f` → `b4c284d` (first commit with the `nert::`
  namespace + full CUDA table incl. Einsum) + adapter migration
  `engine::` → `nert::` (`engine/` → `nert/` includes). API maps 1:1
  (Graph, Model, HostTensorMap, FoldConstants, RunTyped).
- `NEURIPLO_NERT_CUDA` CMake option (default OFF): sets
  `NERT_ENABLE_CUDA=ON` for the FetchContent build and defines
  `NEURIPLO_NERT_CUDA` for the adapter + tests. OFF = byte-identical
  behavior (CPU-only; `CudaDevice` symbol absent).
- Device selection in `NertInfer`: `use_gpu` picks `nert::CudaDevice()`
  (ifdef) else `CpuDevice()`; `FoldConstants` + `Model` use the same
  device. HostAllocator/HostTensorConverter unchanged (NERT stages
  transfers internally). Factory forwards `use_gpu`; registry stays
  `force_gpu=false` (ORT pattern). Without the build flag, GPU
  requests keep throwing CPU-only (symbol absent, not silent CPU).
- Non-CUDA-runnable graphs on GPU: NERT rejects at load
  (`ModelLoadException`, translated as today) — no per-op fallback,
  surfacing through the existing error contract.
- Tests (`NertInferTest`): under `NEURIPLO_NERT_CUDA`, GPU requests
  construct Ready backends; typed_io (Add/Greater/Cast — cuDNN-free
  CUDA rows) GPU-vs-CPU parity; without the flag, existing
  reject/null expectations unchanged.
- Docs: `docs/backends.yaml` NERT `gpu:` flag + `Readme.md` NERT note.

## Out of scope

- cuDNN-gated rows (Conv/ConvTranspose) in Neuriplo CI (no cuDNN on
  runners); ResNet CUDA parity (needs cuDNN; NERT-side gate stands).
- Grounding-DINO / gate runs (separate packets), HIP/OpenCL selection
  (no NERT rows selected here; future `use_gpu` routing).
- NERT closure slice (GridSample/InstanceNorm/ConvTranspose, still
  uncommitted in the NERT tree) — pin stays on pushed `b4c284d`.

## Constraints

- No behavior change with the flag OFF (existing tests byte-identical).
- No `nert::` symbol may be referenced without `NEURIPLO_NERT_CUDA`
  where the symbol is build-conditional (`CudaDevice`); everything
  else in `nert::` is unconditional post-rename.
- Exception translation boundary unchanged: no engine-typed exception
  escapes the adapter.
