# GGML YOLO family — validation

Written before implementation; execute from this file.

## Automated

- [x] [V-1] → [R-1]: `yolov8n.onnx` converts to NGGL (all float initializers).
- [x] [V-2] → [R-2]: YOLOv8n graph loads and runs (finite/deterministic).
- [x] [V-3] → [R-3]: parity vs ORT on raw `[1,84,8400]`: max-cell agreement +
  max-abs-diff ≤ adopted tolerance. Record measured diff.
- [x] [V-4] → [R-4]: resnet18/50 goldens still green.
- [x] [V-5] → gates: full ctest, `gen_backend_docs --check`, `git diff --check`,
  clang-format clean (C++ only).
- [ ] [V-6] → brev re-gate on the real artifact + teardown verified.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | yolo convert | pass | 2026-11-14 | `yolov8n.onnx` (opset 12, static 640) → NGGL 132 float tensors; DFL/mul/const initializers carried; int64 shape params hardcoded+documented |
| V-2 | yolo load+run | pass | 2026-11-14 | full graph loads; [1,84,8400] finite on zeros/gray |
| V-3 | yolo parity | pass | 2026-11-14 | max-cell agree gray+zeros; max-abs-diff 2.01e-03/9.15e-04 vs adopted ≤1e-2 |
| V-4 | resnet no-regress | pass | 2026-11-14 | full ctest 77/77 ( ParityVsOrt/50 green locally) |
| V-5 | gates | pass | 2026-11-14 | ctest 77/77, gen_backend_docs --check, diff-check, clang-format clean |
| V-6 | brev re-gate | not executed | - | |

## Deviations

None yet.
