# GGML ONNX-subset frontend — validation

Commands run in `/home/oli/repos/neuriplo` on `feature/ggml-yolo-family`.

## Automated

- [x] [V-1] → [R-1]: multi-input Add matches the ONNX reference.
- [x] [V-2] → [R-2]: INT64 indices → host I32 cast → Gather matches.
- [x] [V-3] → [R-3]: attention block matches within 1e-5.
- [x] [V-4] → [R-4]: Conv+bias+MaxPool and depthwise Conv match within 1e-5.
- [x] [V-5] → [R-5]: shape/activation and ConstantOfShape fixtures match.
- [x] [V-6] → [R-6]: unsupported ops and dynamic shapes are rejected.
- [x] [V-7] → [R-7]: existing GGML tests and full ctest stay green.
- [x] [V-8] → [R-8]: Shape→Slice→Gather→Concat→Reshape folding preserves
  the input.
- [x] [V-9] → [R-9]: Slice reverse/step, Pad, ReduceMean/Sum/Max, Split,
  Resize-nearest, Expand/Tile match ONNX ReferenceEvaluator.
- [x] [V-10] → [R-10]: TopK values/indices and GatherElements match the
  supported axis contract.
- [x] [V-11] → [R-11]: Erf matches ONNX within 1e-6.
- [x] [V-12] → [R-12]: GridSample bilinear matches ONNX.
- [x] [V-13] → [R-13]: fixed-input `rtdetrv4` raw parity matches ORT.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `./GGMLInferTest --gtest_filter='GGMLOnnxTest.*'` | pass | 2026-11-15 | **29/29**. MultiInputAdd `[11.5..66.5]`; ScaledGemm `[1.3,0.175,2.8,1.175]`; BatchedMatMul `[2.2..33.4]`; RankThreeTranspose perm `[2,0,1]`; MatMulVector `[14,32]`. |
| V-2 | same run | pass | 2026-11-15 | I64 `[2,0]` → rows `[[5,6],[1,2]]`. |
| V-3 | same run | pass | 2026-11-15 | Gemm(transB)/Transpose/MatMul/Sqrt/Div/Softmax → `[3.3395233,2.3395233,3.651452,2.651452]`. |
| V-4 | same run | pass | 2026-11-15 | `conv_pool` `[1,3,2,2]`; `depthwise_conv` `[1,2,4,4]`. |
| V-5 | same run | pass | 2026-11-15 | `shape_ops` `[3,4]`; `constant_of_shape` 2x3 of 1.5. |
| V-6 | same run | pass | 2026-11-15 | `unsupported ONNX op 'Softplus'`; dynamic input shape rejected; Slice step 2, Pad begin, GatherElements rank 2 rejected. |
| V-7 | `ctest --test-dir build-ggml --output-on-failure` | pass | 2026-11-15 | **77/77**; backend binary **57 tests: 56 pass, 1 intentional mock skip**. |
| V-8 | same run | pass | 2026-11-15 | `shape_fold.onnx`: `[2,3,4]` input preserved through folded shape chain. |
| V-9 | same run | pass | 2026-11-15 | Slice reverse `[7,8,5,6,3,4,1,2]`; step `[2,3,4,5,7,8,9,10]`; Pad `[1,2,0,0,3,4,0,0,0,0,0,0]`; ReduceMean `[2.5..22.5]`, ReduceMax `[4..24]`, ReduceSum `[10..90]`; Split `[1..8]/[9..16]`; Resize 2x2→4x4; Expand/Tile `[1,1,2,2,3,3,1,1,2,2,3,3]`. |
| V-10 | same run | pass | 2026-11-15 | TopK top-2 values `{4,5}`, indices `{1,3}`; GatherElements axis 1 → `[3,2,7,6]`; OOB index returns a runtime error instead of aborting. |
| V-11 | same run | pass | 2026-11-15 | Erf `[-2,-1,0,1,2]` → `[-0.995322287,-0.842700779,0,0.842700779,0.995322287]` at 1e-6. |
| V-12 | same run | pass | 2026-11-15 | GridSample bilinear on a 2x2 map with grid `[-0.5,-0.5]..[0.5,0.5]` → `[1,2,3,4]`. |
| V-13 | ORT 1.32 vs `onnx_run` on `rtdetrv4_static1.onnx` | pass | 2026-11-15 | 300/300 detections matched by label+box; max matched box L1 **2.7e-3**, mean **4.9e-5**; max score diff **2.2e-5**. 290/300 labels match at the same index; the rest are near-tie swaps. |
| V-14 | `./scripts/quality/format.sh --check` (clang-format 21.1.8) | pass | 2026-11-15 | 127 files clean. |

## Raw parity: static-batch rtdetrv4

Inputs:

- `images` FP32 `[1,3,640,640]`, deterministic pattern in `[0,1]`
- `orig_target_sizes` INT64 `[1,2]` = `[640,640]`

Outputs:

- labels: `[1,300]` INT64 (GGML emits I32)
- boxes: `[1,300,4]` FP32
- scores: `[1,300]` FP32

Matching result:

- all 300 ORT detections have a GGML detection with the same label and a box
  within L1 2.7e-3
- mean matched box L1 4.9e-5
- max score difference 2.2e-5
- exact-position label agreement 290/300; the remaining 10 are adjacent
  near-tie swaps whose boxes and labels are swapped as a pair

## Known limitations

- `TopK` is still not sorted to ONNX `sorted=1` order; near-tie order can
  differ from ORT.
- `GatherElements` supports axis 1 only.
- `Pad` supports end padding only; non-zero begin padding is rejected.
- `Resize` supports nearest/asymmetric/floor only.
- `Erf`, `GridSample`, integer shape arithmetic, and I32→F32 casts are host
  callbacks, not portable ggml kernels.
- The app gate is not run yet; no matrix cell is promoted.

## Status

Shared frontend and raw parity are green. Next is T-14: wire the ONNX
frontend into `GGMLInfer`/the app and run the `rtdetr` app gate, then promote
the matrix cell only if the app gate passes.
