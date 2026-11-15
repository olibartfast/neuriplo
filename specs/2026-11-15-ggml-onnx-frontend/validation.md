# GGML ONNX-subset frontend — validation

Commands run in `/home/oli/repos/neuriplo` on `feature/ggml-yolo-family`.

## Automated

- [x] [V-1] → [R-1]: multi-input Add matches the ONNX reference.
- [x] [V-2] → [R-2]: INT64 indices → host I32 cast → Gather matches.
- [x] [V-3] → [R-3]: attention block matches within 1e-5.
- [x] [V-4] → [R-4]: Conv+bias+MaxPool and depthwise Conv match within 1e-5.
- [x] [V-5] → [R-5]: shape/activation and ConstantOfShape fixtures match.
- [x] [V-6] → [R-6]: GridSample is rejected by name; dynamic shape is rejected.
- [x] [V-7] → [R-7]: existing GGML tests and full ctest stay green.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `./GGMLInferTest --gtest_filter='GGMLOnnxTest.*'` | pass | 2026-11-15 | 10/10. MultiInputAdd: [11.5,22.5,33.5,44.5,55.5,66.5]; ScaledGemm alpha/beta: [1.3,0.175,2.8,1.175]. |
| V-2 | same run | pass | 2026-11-15 | I64 [2,0] → rows [[5,6],[1,2]]. |
| V-3 | same run | pass | 2026-11-15 | Gemm(transB)/Transpose/MatMul/Sqrt/Div/Softmax → [3.3395233,2.3395233,3.651452,2.651452]. |
| V-4 | same run | pass | 2026-11-15 | `conv_pool` [1,3,2,2] and `depthwise_conv` [1,2,4,4] match ONNX ReferenceEvaluator at 1e-5. |
| V-5 | same run | pass | 2026-11-15 | `shape_ops` [3,4]; `constant_of_shape` 2x3 of 1.5. |
| V-6 | same run | pass | 2026-11-15 | `unsupported ONNX op 'GridSample'`; `graph input 'x' has a dynamic shape`. |
| V-7 | `ctest --test-dir build-ggml --output-on-failure` | pass | 2026-11-15 | **77/77**; backend binary **38 tests: 37 pass, 1 intentional mock skip**. |
| V-8 | `./scripts/quality/format.sh --check` | pass | 2026-11-15 | 127 files clean. |

## Probe: static-batch rtdetrv4

`rtdetrv4.onnx` has a symbolic batch `N`. Setting both inputs to batch 1
(`/tmp/opencode/rtdetrv4_static1.onnx`) gets past the input-shape check and
Conv/ConstantOfShape, then stops at:

```text
unsupported ONNX op 'Slice (/model/backbone/stem/Slice)'
```

The first Slice is `start=-1, end=INT64_MIN, step=-1` on a `[4,2]` tensor,
i.e. an axis-0 reverse. The remaining unsupported set is
`Slice`, `ReduceMean`, `Pow`, `Abs`, `Log`, `Pad`, `Resize`, `Split`,
`Expand`, `Tile`, `ConstantOfShape` chains (shape arithmetic),
`TopK`, `GatherElements`, `GridSample`, and `Erf`.

## Status

The frontend foundation is green, but no `rtdetr` raw-parity or app gate is
claimed. `rtdetrv4` still needs T-7 through T-10 from the plan.
