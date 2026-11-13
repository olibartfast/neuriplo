# GGML ResNet-50 — validation

Written before implementation; execute from here.

## Automated

- [x] [V-1] → [R-1], [R-2]: configure/build green; `GGMLInferTest` with real
  `resnet50.ggml` loads and runs (finite/deterministic/input-sensitive).
- [x] [V-2] → [R-3]: parity golden vs ORT on gray/zeros/gradient: top-5 exact,
  max-abs-diff ≤ 1e-3. Record measured diffs.
- [x] [V-3] → [R-4]: ResNet-18 golden + parity still green (no regression).
- [x] [V-4] → [R-5]: truncated/unknown layout file fails with path + shapes.
- [x] [V-5] → gates: full ctest, `gen_backend_docs --check`, `git diff --check`,
  clang-format clean (C++ only).

## Manual (brev)

- [ ] [V-6] fresh CPU instance: configure + build + test + converter (real
  resnet50) + parity golden; then `brev delete` + `brev ls` empty.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | resnet50 load+run | pass | 2026-11-13 | 299-tensor NGGL loads; finite/deterministic/input-sensitive (via parity runs) |
| V-2 | resnet50 parity | pass | 2026-11-13 | top-5 exact gray/zeros/gradient ([902,623,892,584,677] etc.); diff 2.98e-06/2.77e-06/3.81e-06 vs ≤1e-3 |
| V-3 | resnet18 no-regress | pass | 2026-11-13 | ParityVsOrt still exact (6.38e-06/4.59e-06/5.72e-06); full ctest 77/77 |
| V-4 | bad layout throws | pass (existing) | 2026-11-13 | unknown file → path + shapes seen (R-5 path, covered by corrupt/negative tests) |
| V-5 | gates | pass | 2026-11-13 | ctest 77/77, gen_backend_docs --check, diff-check, clang-format clean |
| V-6 | brev re-gate | not executed | - | |

## Deviations

None yet.
