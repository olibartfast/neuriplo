# GGML attention/executor ops — validation

Written before implementation; evidence appended after the corresponding run.
Commands run in `/home/oli/repos/neuriplo` on `feature/ggml-yolo-family`.

## Automated

- [x] [V-1] → [R-1]: `GGMLOps.{hpp,cpp}` compile into the GGML backend and
  are linked into `libneuriplo.so`.
- [x] [V-2] → [R-2]: op-level parity for MatMul/Linear, LayerNorm, GELU,
  scaled dot-product attention, row Gather, and TopK indices.
- [x] [V-3] → [R-2]: tiny transformer block (LayerNorm → Linear → GELU →
  Linear → residual) parity.
- [x] [V-4] → [R-3]: existing ResNet/YOLO/YOLO-seg tests and full ctest stay
  green.
- [x] [V-5] → [R-4]: RT-DETR raw-output parity is **not** claimed; the graph
  blocker and next packet are documented below.

## Adopted tolerance

Op-level and tiny-block float parity: `1e-5` (LayerNorm/attention `1e-4`
because the reference uses float64 math and ggml uses float32). No loosening
was needed.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `cmake --build build-ggml` | pass | 2026-11-14 | `GGMLOps.cpp` built into `libneuriplo.so`; header exported to the test target; configure/build green. |
| V-2 | `GGMLInferTest --gtest_filter='GGMLOpsParity.*'` | pass | 2026-11-14 | 7/7: MatMul/Linear, LayerNorm, GELU, attention, Gather/TopK, tiny transformer block, contract rejection. |
| V-3 | same run | pass | 2026-11-14 | Tiny block expected `[1.6826661, 2.8413231, 3.6826661, 4.8413231]`; got within 1e-5. |
| V-4 | full `GGMLInferTest` + `ctest --test-dir build-ggml` | pass | 2026-11-14 | 21 tests: 20 pass, 1 intentional mock skip; full ctest **77/77**. ResNet-18/50, YOLOv8n detection, and YOLOv8n-seg parity all still green. |
| V-5 | `./scripts/quality/format.sh --check`, `git diff --check` | pass | 2026-11-14 | 121 files formatted; whitespace clean. |

## RT-DETR blocker (recorded, not a PASS)

The original Stage 1 plan said “prove it on `rtdetr` raw outputs first.” That
is deferred because the available `rtdetrv4.onnx` is not a hand-writable
topology:

- opset 18, **1928 nodes**, 380 initializers;
- **2 inputs**: `images` `[N,3,640,640]` FP32 and `orig_target_sizes`
  `[N,2]` INT64; `GGMLInfer` still accepts exactly one FP32 input;
- **3 outputs**: `labels` (I64), `boxes`, `scores`, with dynamic leading
  dimensions;
- op mix includes `GridSample` (9), `TopK` (3), `GatherElements` (3),
  `Cast` (22), `ConstantOfShape` (6), `Expand` (4), `Tile` (4), `Shape` (72),
  `Gather` (81), `MatMul` (66), `Softmax` (11), plus `Reduce*`, `Erf`,
  `Pad`, `Flatten`, `Gemm`.

Required next packet: multi-input/INT64 plumbing plus a generic ONNX-subset
importer (or a purpose-built RT-DETR builder) covering those ops. Until then,
the `rtdetr` GGML matrix cell stays `U`; no app gate is attempted.

## Deviations

- The op layer is implemented and parity-proven independently of RT-DETR.
  This is the useful, reviewable slice of Stage 1; the full RT-DETR graph is
  a separate, larger compiler/importer task.
- `TopK` exposes indices only (ggml returns I32 indices; order is explicitly
  unspecified). Values require a layout-aware Gather; the test compares the
  index set rather than order.
- `Gather` intentionally rejects int64 indices. The future importer must cast
  ONNX int64 indices to I32; this avoids silently reinterpreting bytes.
