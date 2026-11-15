# GGML multi-input graph executor — requirements

**Date:** 2026-11-15
**Branch:** `feature/ggml-yolo-family` (working tree)
**Status:** Implemented and tested; ONNX frontend still separate
**Program:** neuriplo-infer `2026-11-13-ggml-matrix-coverage` Phase 2, Stage 1

## Goal

Provide the execution substrate for transformer models: a programmatic graph
with multiple inputs (FP32 and INT64), multiple outputs, and the attention
op set, so a future ONNX-subset importer can map a graph onto ggml without
changing `GGMLInfer`'s hand-written model paths.

## In Scope

- `backends/ggml/src/GGMLGraph.{hpp,cpp}`:
  - F32/I32/I64 tensors, multiple inputs, multiple marked outputs;
  - ops: Add, Mul, MatMul, Linear, scaled dot-product attention, Softmax,
    LayerNorm, GELU, Gather, I64→I32 cast;
  - exact-size input validation and loud contract errors.
- Host-side I64→I32 materialization because this ggml build's `cast`/`dup`
  kernels abort on I64→I32.
- `backends/ggml/test/GGMLGraphTest.cpp`: multi-input Add, INT64
  cast+gather, multiple outputs, tiny transformer block, bad contracts.
- CMake wiring into `libneuriplo.so` and the GGML test binary.

## Out of Scope

- The ONNX frontend/parser. The graph API is the target, not the importer.
- Full RT-DETR execution: `GridSample`, `TopK`, `GatherElements`, `Resize`,
  dynamic `TopK` shapes, and the `model.23`/attention head still need work.
- App-surface changes, pins, or other backends.

## Decisions

- [D-1] Shapes use ggml dimensions (`ne[0]` first); the future importer must
  reverse ONNX axes.
- [D-2] `CastI64ToI32` is executed on the host after input copies; `ggml_cast`
  is not used because the installed CPU kernel aborts on I64→I32.
- [D-3] Outputs are returned in mark order with their dtype/shape; the matrix
  and app gates remain separate decisions.

## Requirements

- [R-1] Multi-input FP32/INT64 graphs execute with exact-size validation.
- [R-2] Multiple outputs are returned independently.
- [R-3] Op/block parity matches the direct ggml op tests (1e-5).
- [R-4] Existing ResNet/YOLO/YOLO-seg model paths and tests stay green.
- [R-5] No RT-DETR/transformer PASS is claimed in this packet.
