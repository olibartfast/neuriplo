# GGML multi-input graph executor — validation

Commands run in `/home/oli/repos/neuriplo` on `feature/ggml-yolo-family`.

## Automated

- [x] [V-1] → [R-1]: multi-input FP32 graph executes; exact-size validation
  rejects missing/extra/wrong-size blobs.
- [x] [V-2] → [R-1]: INT64 input → host I64→I32 cast → row Gather works.
- [x] [V-3] → [R-2]: two marked outputs return independently.
- [x] [V-4] → [R-3]: tiny transformer block parity within 1e-5.
- [x] [V-5] → [R-4]: existing GGML model tests and full ctest stay green.
- [x] [V-6] → [R-5]: no RT-DETR/transformer cell promoted.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `GGMLInferTest --gtest_filter='GGMLGraphTest.*'` | pass | 2026-11-15 | 6/6: MultiInputAdd, Int64InputCastAndGather, MultipleOutputs, TinyTransformerBlock, RejectsBadContracts, ShapeAndActivationOps. |
| V-2 | same run | pass | 2026-11-15 | I64 indices [2,0] → I32 → Gather over `[D=2,N=3]` data returns `[5,6,1,2]`. |
| V-3 | same run | pass | 2026-11-15 | Add and Mul outputs returned in mark order: `[8,10]` and `[15,24]`. |
| V-4 | same run | pass | 2026-11-15 | Tiny block output `[1.6826661, 2.8413231, 3.6826661, 4.8413231]` within 1e-5. |
| V-5 | full `GGMLInferTest` + `ctest --test-dir build-ggml` | pass | 2026-11-15 | 38 tests: 37 pass, 1 intentional mock skip; full ctest **77/77**. The later ONNX-frontend packet adds 10 of those tests. |
| V-6 | matrix status | pass | 2026-11-15 | No app cell changed; `rtdetr` remains blocked on the ONNX frontend and deformable/dynamic ops. |

## Blocker after this packet

The executor is not an ONNX importer. `rtdetrv4.onnx` still needs:

- a frontend that maps ONNX nodes and static value shapes onto `GGMLGraph`;
- `GridSample` (deformable attention), `TopK`, `GatherElements`, `Resize`,
  `Pad`, `Reduce*`, `Clip`, `Erf`, `Gemm`-specific layouts;
- dynamic-shape handling for `TopK`/`GatherElements` outputs.

`yolo26` additionally needs the converter to accept its `model.23` head.
