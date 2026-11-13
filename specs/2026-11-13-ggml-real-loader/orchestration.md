# GGML real loader — orchestration

Pyramid for `feature/ggml-real-loader`. Roles, boundaries, acceptance.

## Roles

- [O-1] Orchestrator (this session): owns spec IDs, brev lifecycle, merge readiness.
  Only role that creates/deletes brev instances.
- [O-2] Loader worker: `backends/ggml/src/GGMLInfer.cpp/.hpp` only. No CMake
  registry, no other backend, no public headers.
- [O-3] Converter worker: `scripts/convert_onnx_to_ggml.py`,
  `docker/Dockerfile.ggml` model stage, `docs/backends.yaml` text only.
- [O-4] Test worker: `backends/ggml/test/GGMLInferTest.cpp` + docs regen evidence.
- [O-5] Reviewer (read-only): R↔V traceability, silent-fallback ban, ABI boundary,
  scope leak (generic GGUF / LLAMACPP / GPU).

## Permission boundaries

- Workers never touch `versions.env`, `cmake/BackendRegistry.cmake`,
  public `include/neuriplo/*`, other `backends/*/`, or CI workflows.
- No pin move in this packet; neuriplo-infer pin move is a separate follow-up.
- Brev: one instance max (`ggml-real-loader`, `cpu-e2.4vcpu-16gb`); delete +
  `brev ls` verification mandatory before merge consideration.

## Handoff packet (each worker gets)

Spec dir + exact files owned + single acceptance command + what to report
(pass/fail + verbatim output, no inference).

## Run ledger

| Date | Role | Command | Result |
|------|------|---------|--------|
| 2026-11-13 | Orchestrator | spec packet written; validation before implementation | done |
| 2026-11-13 | Reviewer (read-only) | NOT merge-ready: F1 (converter destroyed by clang-format-on-Python), F2 (transpose inverted), F3 (spec/code scope mismatch) + S1-S6 | done |
| 2026-11-13 | Orchestrator | F1: rewrote converter (py_compile OK, byte-identical round trip, clean exits); F2: fixed orientation + empirical probe (logit[3]=10.0 exact vs numpy); F3: spec amended head-only; S1 cap + S2 dead-tensor removal + S3/S6 test hardening | done, local gates green |
| 2026-11-13 | neuriplo-infer wiring | `LinkBackend.cmake` + test mirror + help string; `-DDEFAULT_BACKEND=GGML` configure+build link OK | done (branch `feature/ggml-matrix-coverage`) |
| 2026-11-13 | Brev gate V-11..V-12 | `ggml-real-loader` box: apt+GTest+SDK(BLAS) built, configure+build+ctest green, real resnet18-v1-7.onnx→NGGL (102 tensors), golden 9/1 on real weights, top-5 (986/1.213…) + infer<1ms recorded | done |
| 2026-11-13 | Brev teardown V-13/V-15 | `ggml-resnet-full` DELETING→gone; final `brev ls`: "No instances in org" | done, org clean |
| 2026-11-13 | Spec close-out | V-14/V-15 ticked; Q-2 adopted (≤1e-3, measured ~6e-06); gradient case closes layout hole; residuals 2/3/4/6 recorded | done |
| - | Reviewer re-pass | all prior findings CLOSED; new notes closed/recorded (see validation deviations) | done |
| - | Loader worker | not started | - |
| - | Converter worker | not started | - |
| - | Test worker | not started | - |
| - | Reviewer re-pass | not started (needs V-12/V-13 + conv-path decision) | - |
