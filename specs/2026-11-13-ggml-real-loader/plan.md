# GGML real loader — plan

Derived from `requirements.md`. Thin phases, each ending runnable. Pyramid:
orchestrator → workers (loader impl, converter/model-stage, tests/docs) →
reviewer. Brev box `cpu-e2.4vcpu-16gb`, deleted after testing.

## Group 0 — Probe (no behavior change)

- [T-1] Compile probe: confirm pinned `v0.11.0` headers expose the symbols the
  loader needs; record `GGML_DIR` layout used.
- [T-2] Fixture probe: dump the current `resnet18.ggml` bytes (header-only?) and
  the ONNX source shapes; fix the byte layout decision [Q-1] in writing before
  coding the loader.

## Group 1 — Loader (core fix)

- [T-3] `GGMLInfer::load_model`: read + validate real weight file (magic, dims,
  weight count); failure names path + reason. Keep `ctx_`/`backend_` lifetime;
  retain `ggml_backend_buffer_t` handle for release.
- [T-4] Graph from weights: block-average-pool the 150528 input elements to 512
  features (CPU, documented preprocessing) + the real FC (`W*x+b`, weight
  transposed to ggml ne order) through ggml. Head-only by design (see
  requirements scope note); conv/BN do not execute.
- [T-5] `get_infer_results`: execute graph, extract the real output tensor
  (dtype-aware; dequant if the file is quantized), return real shape; delete
  the zeros-vector path.
- [T-6] `use_gpu=true` throws actionable error; buffer-alloc failure throws
  (closes warn-and-continue).

## Group 2 — Converter + model stage

- [T-7] Fix `scripts/convert_onnx_to_ggml.py` (or its replacement) to write the
  layout [T-3] reads; Dockerfile model stage copies the artifact where
  `GGMLInferTest` expects it (`build/backends/ggml/test` + `model_path.txt`).
- [T-8] `docs/backends.yaml` model_format/source text matches the artifact;
  regenerate docs (`gen_backend_docs.py`).

## Group 3 — Tests + gates

- [T-9] `GGMLInferTest.cpp`: real-model golden (non-zero/finite/deterministic,
  tolerance recorded per [Q-2]), missing-file, bad-size, multi-input negatives.
  Mock-only green does not close [R-7].
- [T-10] Full gates: CPU configure/build/ctest, ABI checks, `gen_backend_docs
  --check`, `git diff --check`, clang-format/cppcheck on touched C++.
- [T-11] Brev live gate (`cpu-e2.4vcpu-16gb`): configure
  `-DDEFAULT_BACKEND=GGML -DBUILD_INFERENCE_ENGINE_TESTS=ON`, build,
  `ctest -R GGMLInferTest`, single-shape run with top-5 recorded; then
  `brev delete` + `brev ls` verified empty.

## Group 5 — Phase 2: full ResNet18 graph + ORT parity (scope: goal)

- [T-14] Layout + BN-fold helpers in `GGMLInfer.cpp` (no graph change yet):
  NCHW blob → `[W,H,C,N]` input permute at inference; `[O,I,H,W]` → `[KW,KH,IC,OC]`
  kernel permute at load; per-channel BN fold (`s=w/sqrt(var+1e-5)`,
  `beta=b-mean*s`); role lookup (conv/BN/FC tensors by shape order).
- [T-15] Full topology builder replacing the head-only graph: stem
  (conv7x7/s2/p3 → bias → relu → maxpool3x3/s2/p1), 8 basic blocks with skips
  (3 with 1x1 downsample), global-avg-pool-7x7, FC. All 20 convs + 20 folded
  BNs + 8 adds; graph inputs: permuted image tensor; output: real logits.
- [T-16] ORT reference harness (evidence only, not shipped): `/tmp/ortenv`
  onnxruntime on `resnet18-v1-7.onnx`, fixed inputs (gray-128, zeros, one
  natural image if available), saves logits + top-5.
- [T-17] Parity golden in `GGMLInferTest.cpp`: same inputs through GGML,
  top-5 classes agree exactly, max-abs-logit-diff ≤ adopted tolerance;
  record measured diff. Old head-only assertions stay as regression history.
- [T-18] Brev re-gate on a fresh instance (old one STOPPED): configure +
  build + test + converter (real ONNX) + parity golden; then delete +
  verify.
- [T-19] Reviewer re-pass (full-graph numerics focus: conv orientation,
  padding/stride, BN fold, pool coverage, skip wiring) + merge.

## Group 6 — Review + integrate (Phase 2 gate)

- [T-20] Read-only reviewer: every `R-` (incl. Phase 2) has a `V-`, every `V-`
  names its `R-`; no silent fallback, no ABI drift, no scope leak.
- [T-21] Merge `feature/ggml-real-loader` → `develop` (merge commit), then
  follow-ups: pin move in neuriplo-infer `versions.env` + BackendCoverage cell
  promotion (ResNet18 classification PASS) in a separate packet.
