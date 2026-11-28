# GGML real loader — validation

Written before implementation; execute from here. A check that cannot run is
recorded as not executed, never as a pass.

## Automated

- [x] [V-1] → [R-1], [R-4]: CPU configure/build/test green, no new warnings:
  `cmake -S . -B build-ggml -DNEURIPLO_BACKENDS=GGML -DBUILD_INFERENCE_ENGINE_TESTS=ON`
  then `cmake --build build-ggml --parallel "$(nproc)"` and
  `ctest --test-dir build-ggml --output-on-failure`.
- [x] [V-2] → [R-1]: missing file fails with path in message; header-only /
  truncated `.ggml` fails instead of "loading".
- [x] [V-3] → [R-2], [R-3]: real-model inference returns 1000 finite values,
  not all zeros, deterministic across two runs, and input-sensitive (gray vs
  zeros differ). Head-only scope per requirements note: no ORT parity claimed.
  Record `max_abs_diff_vs_zeros` (measured 37.1 on synthetic weights) and the
  F2 orientation probe (asymmetric W[3,7]=5, controlled input → logit[3]=10
  exact vs numpy).
- [x] [V-4] → [R-4]: wrong byte size and multi-input both throw with
  expected-vs-got.
- [x] [V-5] → [R-5]: `use_gpu=true` throws naming the GPU request; no CPU output.
- [ ] [V-6] → [R-6]: backend-buffer failure path throws (code review + fault
  injection or log evidence; warn-and-continue gone).
- [x] [V-7] → [R-7]: `ctest -R GGMLInferTest` passes with the real artifact
  present (no mock fallback); negatives above included in the same binary.
- [x] [V-8] → [R-8]: `python3 scripts/gen_backend_docs.py --check` passes;
  Dockerfile model stage still produces the artifact [T-3] reads.
- [x] [V-9] → [R-9]: consumer C ABI symbol/layout + plugin ABI tests unchanged.
- [x] [V-10] → [R-5], [R-6]: format/lint clean (`git diff --check`,
  clang-format, cppcheck on touched C++).

## Manual (brev live gate — evidence, not aspiration)

- [x] [V-11] → [R-10]: `brev create ggml-real-loader --type cpu-e2.4vcpu-16gb`;
  record `nproc/free/uname -m`.
- [x] [V-12] → [R-10]: on brev, configure + build + `ctest -R GGMLInferTest`
  + single-shape run; record commands, exit codes, top-5 classes/scores.
- [ ] [V-13] → [R-10]: `brev delete ggml-real-loader`; `brev ls` verified empty.
  Required even if V-11/V-12 fail — no orphan instances.

## Phase 2 (full ResNet18 + ORT parity)

- [x] [V-14] → [R-2], [R-3]: parity golden on the real `resnet18.ggml`:
  identical inputs through GGML and ORT (`/tmp/ortenv`, `resnet18-v1-7.onnx`);
  top-5 classes agree exactly; max-abs-logit-diff ≤ adopted tolerance
  (proposal ≤1e-3). Record commands, tolerance, measured diff, top-5.
  — pass 2026-11-13: top-5 exact on gray ([623,473,902,512,783]) and zeros
  ([623,473,783,892,844]); diff 6.38e-06 / 4.59e-06 vs ≤1e-3.
  — pass 2026-11-13 (gradient): nonuniform NCHW pattern closes the layout
  coverage hole (review finding 1); top-5 exact [858,538,807,904,905],
  diff 5.72e-06. New findings 5 (docstring) + 8a (input label) fixed in-branch;
  2/3/4/6 recorded as residuals.
- [x] [V-15] → [R-10]: fresh brev instance re-gate (configure + build + test +
  converter on real ONNX + parity golden); then `brev delete` + `brev ls`
  verified empty.
  — pass 2026-11-13 (`ggml-resnet-full`): SDK rebuilt with BLAS, configure +
  build green, real resnet18-v1-7.onnx → 102-tensor NGGL, `GGMLInferTest`
  10 passed / 1 correctly skipped; parity gray 4.05e-06, zeros 6.26e-06,
  gradient 7.63e-06, top-5 exact on all three.

## Acceptance Commands

```bash
cmake -S . -B build-ggml -DNEURIPLO_BACKENDS=GGML -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build-ggml --parallel "$(nproc)"
ctest --test-dir build-ggml -R GGMLInferTest --output-on-failure
python3 scripts/gen_backend_docs.py --check
```

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | CPU build + tests | pass | 2026-11-13 | `build-ggml` full `ctest`: 77/77 pass |
| V-2 | missing/corrupt file | pass | 2026-11-13 | `MissingFileThrowsWithPath` names path; `TruncatedCorruptFileThrows` names path; huge-count capped vs remaining bytes (S1) |
| V-3 | real-model golden | pass (synthetic) | 2026-11-13 | 1000 finite, non-zero, deterministic, input-sensitive (gray vs zeros differ); `max_abs_diff_vs_zeros=37.1`; F2 probe: asymmetric W[3,7]=5 + controlled input → logit[3]=10.0 exact vs numpy. No ORT parity (head-only, Q-2 resolved) |
| V-4 | bad-size / multi-input | pass | 2026-11-13 | both throw with expected-vs-got / single-input text |
| V-5 | use_gpu=true error | pass | 2026-11-13 | names GPU request, no CPU fallback; no artifact needed |
| V-6 | buffer-fail throws | pass (review) | 2026-11-13 | `fail_model` path; warn-and-continue removed. Fault injection not run |
| V-7 | GGMLInferTest real | pass (synthetic) | 2026-11-13 | 9 passed / 1 skipped (MockUnit, correctly skipped with artifact); 5 skipped without artifact |
| V-8 | docs check | pass | 2026-11-13 | `gen_backend_docs.py --check` exit 0; Dockerfile CLI args unchanged |
| V-9 | ABI checks | pass | 2026-11-13 | full ctest incl. PluginAbiContractTest, CApiContractTest/CApiWrapperTest; only private members added |
| V-10 | format/lint | pass | 2026-11-13 | `git diff --check` clean; clang-format dry-run clean (C++ only — never run on .py) |
| V-11 | brev create | pass | 2026-11-13 | `ggml-real-loader` (cpu-e2.4vcpu-16gb, 4 vCPU/15 GB, x86_64); SSH needed retries, then `nproc/free/uname` OK |
| V-12 | brev live run | pass | 2026-11-13 | Box: pkg-config+gfortran added, SDK FORCE-rebuilt with BLAS, GTest built from /usr/src, configure+build green, `ctest -R GGMLInferTest` green. Real resnet18-v1-7.onnx (46.8 MB, ONNX Zoo) → NGGL 102 tensors/46.8 MB. Golden 9 passed/1 skipped on real weights. Top-5 gray input: 986/1.213, 351/1.152, 92/1.144, 142/1.033, 95/1.028; infer <1 ms. Head-only scope: no ORT parity claimed. |
| V-13 | brev teardown (both boxes) | pass | 2026-11-13 | `ggml-real-loader` STOPPED record vanished; `ggml-resnet-full` went DELETING→gone; final `brev ls`: "No instances in org" |
| V-15 | fresh-box re-gate | pass | 2026-11-13 | `ggml-resnet-full`: SDK+BLAS rebuilt, configure+build green, real ONNX→102-tensor NGGL, 10 passed/1 skipped; parity gray 4.05e-06, zeros 6.26e-06, gradient 7.63e-06, top-5 exact ×3 |

## Deviations

- Reviewer pass (read-only subagent) verdict NOT merge-ready → fixed in-branch:
  F1 converter was destroyed by running clang-format on Python (shebang/imports
  crushed); rewrote `scripts/convert_onnx_to_ggml.py` cleanly, `py_compile` OK,
  synthetic-ONNX conversion byte-identical round trip, clean exit codes.
  Never run clang-format on .py again.
- F2 transpose inverted in loader; fixed + proven empirically (asymmetric
  weights + controlled input → exact numpy match, argmax=3, 10.0).
- F3 scope mismatch resolved by amending spec (head-only: pool + real FC;
  conv/BN do not run; no ORT parity claimed) instead of implementing ResNet conv.
- S1 (allocation cap vs remaining bytes + dim-product overflow guard) done.
- S2 (dead `input_tensor_` removed; shape contract is a validated constant) done.
- S3 (input-sensitivity assertion) + S6 (path assertion on corrupt file;
  use_gpu test needs no artifact) done.
- S4 residual: stale writers (`scripts/convert_to_ggml.sh`,
  `scripts/convert_resnet18_to_ggml.py`, `backends/ggml/test/create_test_model.py`,
  `test/generate_model.sh`, `scripts/test_backends.sh` placeholder echo) still
  emit artifacts the strict loader rejects. Left for a follow-up; CI paths
  using them were already mock-degraded. Do not cite mock-green as evidence.
- S5 residual: loader accepts `[512,1000]`-ordered files too (transposed into
  place); converter only writes `[1000,512]`. Harmless superset, documented.
- Reviewer new low/info (closed in-branch): converter skips 0-d/>4-d float
  initializers with a note (reader only accepts 1D..4D); `--input-shape` /
  `--batch-size` non-defaults now exit 2 instead of silent ignore; read-back
  verification catches `struct.error` with exit 1. Accepted residual: ambiguous
  duplicate head shapes pick first (single head for ResNet18); batch>1 output
  framing unverified (fixed batch-1 scope).
- Teardown state (V-13): `brev delete ggml-real-loader` (and by ID `ab91q7kcw`)
  each exit 0; the instance transitioned RUNNING→UNHEALTHY→STOPPED, so no
  compute is burning, but the STOPPED record still lists in `brev ls`
  (~25 min after the first delete). Different from last session's lingering
  RUNNING: this one is halted. If the record persists, remove it from the
  brev web console; do not present the bill as open either way. `video-vc-test`
  (hyperstack_L40) is someone else's — never touched.
- `$0` with no artifact: 5 passed / 5 skipped is the honest no-model signal;
  only artifact-backed runs count (V-3/V-7).
