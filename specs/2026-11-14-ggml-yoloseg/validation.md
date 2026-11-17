# GGML YOLOv8n-seg — validation

Written before implementation; evidence appended after the corresponding run.
All commands below run in `/home/oli/repos/neuriplo` on branch
`feature/ggml-yolo-family`.

## Automated

- [x] [V-1] → [R-1]: `yolov8n-seg.onnx` converts to NGGL with all float
  initializers, including cv4 and proto weights.
- [x] [V-2] → [R-2]: the segmentation graph loads and both outputs are
  finite/deterministic; unknown tensor/shape failures name path + tensor.
- [x] [V-3] → [R-3]: `output0` parity vs ORT on gray/zeros/gradient:
  max-cell agreement and max-abs-diff within the adopted tolerance. Record
  measured diffs.
- [x] [V-4] → [R-4]: `output1` parity vs ORT elementwise on the same inputs.
  Record measured diffs.
- [x] [V-5] → [R-5]: ResNet-18/50 and YOLOv8n detection goldens remain green
  (full `GGMLInferTest` + ctest).
- [x] [V-6] → [R-6]: detection returns exactly one `{1,84,8400}` output with
  unchanged shape; segmentation returns `[1,116,8400]` then
  `[1,32,160,160]`.
- [x] [V-7] → gates: build `build-ggml`, `ctest -R GGMLInferTest`, full
  ctest where practical, `./scripts/quality/format.sh --check` for touched
  C++ files, `git diff --check`.
- [x] [V-8] → [R-7]: no changes outside `backends/ggml`, `scripts/`, and
  this packet; no commit or push.

## Adopted tolerance

`output0`: max-abs-diff ≤ 1e-2 (same bar as detection); measured
7.63e-04 … 1.53e-03.
`output1`: max-abs-diff ≤ 1e-2 (F16 ConvTranspose); measured
1.09e-03 … 1.68e-03.

No tolerance loosening was needed.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `python3 scripts/convert_onnx_to_ggml.py --input /tmp/opencode/ggml-matrix/yolov8n-seg.onnx --output /tmp/yolov8n-seg.ggml --input-shape 3,640,640` | pass | 2026-11-14 | 158 float tensors, 13,793,302 bytes; int64 shape params skipped with notes; cv4 + proto + ConvTranspose weights carried. |
| V-2 | `GGMLInferTest --gtest_filter='*ParityVsYoloSeg*'` | pass | 2026-11-14 | Loads as `YOLOv8n-seg (CSP stem + C2f + SPPF + PAN + decoupled head + DFL + mask/proto, real weights)`; both outputs finite (test asserts `std::isfinite` on every element); segment head detected from `model.22.proto.cv1.conv.weight` + `model.22.cv4.0.0.conv.weight`. |
| V-3 | `GGMLInferTest --gtest_filter='*ParityVsYoloSeg*'` | pass | 2026-11-14 | `output0` max-cell agree=1 on gray/zeros/gradient; max-abs-diff 7.62939e-04 / 1.52588e-03 / 7.93457e-04 (≤1e-2). |
| V-4 | same run | pass | 2026-11-14 | `output1` max-abs-diff 1.60217e-03 / 1.08612e-03 / 1.67644e-03 (≤1e-2). |
| V-5 | full `GGMLInferTest` + `ctest --test-dir build-ggml` | pass | 2026-11-14 | 14 tests: 13 pass, 1 intentional mock skip; full ctest **77/77** pass. ResNet-18 parity 6.38e-06 / 4.59e-06 / 5.72e-06; ResNet-50 2.98e-06 / 2.77e-06 / 3.81e-06; YOLOv8n detection 2.01e-03 / 9.16e-04 / 6.71e-04. |
| V-6 | `ParityVsYoloSeg` shape assertions | pass | 2026-11-14 | `shapes[0] == {1,116,8400}`, `shapes[1] == {1,32,160,160}`; existing `ParityVsYolo` still asserts one `{1,84,8400}` output. |
| V-7 | build/format/diff | pass | 2026-11-14 | `cmake --build build-ggml` green; `./scripts/quality/format.sh --check` green; `git diff --check` clean. |
| V-8 | worktree scope | pass | 2026-11-14 | Changes only `backends/ggml/src/GGMLInfer.{hpp,cpp}`, `backends/ggml/test/GGMLInferTest.cpp`, `scripts/ggml_ort_reference.py`, and this packet. No commit/push. |

## Commands

```bash
# Build
cmake --build build-ggml -j"$(nproc)"

# Targeted seg parity
build-ggml/backends/ggml/test/GGMLInferTest --gtest_filter='*ParityVsYoloSeg*'

# Full backend suite
ctest --test-dir build-ggml -R GGMLInferTest --output-on-failure
ctest --test-dir build-ggml --output-on-failure

# Format / hygiene
./scripts/quality/format.sh --check
git diff --check
```

Reference generation (outside the repo):

```bash
/tmp/opencode/ggml-matrix/yoloenv/bin/python scripts/ggml_ort_reference.py \
  --model /tmp/opencode/ggml-matrix/yolov8n-seg.onnx \
  --img-size 640 --all-outputs \
  --gray-out /tmp/ort_yoloseg_gray.npy \
  --zeros-out /tmp/ort_yoloseg_zeros.npy \
  --gradient-out /tmp/ort_yoloseg_gradient.npy
```

## Deviations

- The installed `ggml` `CONV_TRANSPOSE_2D` CPU kernel requires an F16 weight
  and casts activations to F16. The loader transposes ONNX
  `[Cin,Cout,kH,kW]` to ggml `[kW,kH,Cout,Cin]` and fills an F16 constant;
  the measured proto diff stays well inside 1e-2, so no tolerance change was
  needed.
- The seg export numbers the box-decode constants differently from the
  detection export (`Constant_15/16/18` vs `Constant_12/13/15`). The loader
  selects by tensor presence instead of assuming one numbering.

## Review (2026-11-14)

- Read-only reviewer pass over both repos: **no blocking issues**. Addressed:
  corrected the `output0` channel map (4 box + 80 class + 32 mask), added an
  F32 output-type assertion before every float read, documented the
  ConvTranspose copy as an intentional reversed-dims coincidence, added a
  two-run determinism check to `ParityVsYoloSeg`, and added gate-harness
  negative-path tests in `neuriplo-infer`.

## Out of scope

- `yolo26` (attention/model.23 head) and all app-surface evidence live in
  the `neuriplo-infer` packet and E1/E19; this packet only proves the raw
  backend outputs.
