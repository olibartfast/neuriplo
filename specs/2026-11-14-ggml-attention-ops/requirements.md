# GGML attention/executor ops — requirements

**Date:** 2026-11-14
**Branch:** `feature/ggml-yolo-family` (working tree; follow-up to the YOLO family work)
**Status:** Op primitives implemented and parity-tested; RT-DETR graph deferred
**Program:** neuriplo-infer `2026-11-13-ggml-matrix-coverage` Phase 2, Stage 1

## Goal

Add a reusable attention/executor op layer to the GGML backend and prove it
with op-level and tiny-block parity. This is the foundation for transformer
representatives (`rtdetr`, `depth_anything_v2`, `vitpose`, …), not a promise
that any full transformer graph runs yet.

## Scope and reality check

`rtdetrv4.onnx` (opset 18) is 1928 nodes with 2 inputs (`images` FP32,
`orig_target_sizes` INT64), 3 outputs, `GridSample`, `TopK`,
`GatherElements`, `Cast`, `ConstantOfShape`, `Expand`, `Tile`, dynamic
shape/control ops, and 66 `MatMul`/11 `Softmax` sites. The current
`GGMLInfer` is a hand-written topology loader, not a generic ONNX executor.
Running the full RT-DETR graph therefore requires, at minimum:

- multi-input support with an INT64 side input;
- a generic ONNX-subset graph importer (or a purpose-built RT-DETR topology
  builder) covering the ops above;
- `GridSample` / deformable-attention support;
- dynamic-shape handling for `TopK` / `GatherElements` outputs.

Those are deliberately **out of scope for this packet**. This packet lands
the op helpers and proves their semantics; RT-DETR raw parity stays blocked
until the graph importer/multi-input work is specified separately.

## In Scope

- `backends/ggml/src/GGMLOps.{hpp,cpp}` helpers:
  - LayerNorm over `ne[0]` with per-feature weight/bias;
  - erf-based GELU;
  - MatMul / Linear with the reversed-axis layout convention;
  - single-head scaled dot-product attention;
  - row Gather with I32 indices (explicitly rejecting int64);
  - TopK indices (order unspecified, matching ggml).
- `backends/ggml/test/GGMLOpsParityTest.cpp`:
  - op-level parity for MatMul/Linear, LayerNorm, GELU, attention, Gather,
    TopK;
  - a tiny transformer block (LayerNorm → Linear → GELU → Linear →
    residual) parity check;
  - contract-rejection checks (shape mismatch, non-I32 gather indices).
- CMake wiring so the helpers are part of `libneuriplo.so` and the test
  binary.

## Out of Scope

- Full RT-DETR / `depth_anything_v2` / `vitpose` graph execution.
- Multi-input or INT64 graph inputs, `GridSample`, dynamic shapes, generic
  ONNX importer.
- App-surface gates in `neuriplo-infer` (those require the model path to
  load and run first).
- Any change to existing ResNet/YOLO behavior, pins, or other backends.

## Decisions

- [D-1] The op layer uses the same reversed-axis storage convention as
  `GGMLInfer`: ONNX `[S,D]` is stored as ggml `[D,S]`, so `ne[0]` is the
  feature/reduction axis.
- [D-2] Attention transposes V from `[head_dim, seq_k]` to `[seq_k,
  head_dim]` before the second MatMul; contracting the wrong axis is a
  silent numeric error.
- [D-3] Gather requires I32 indices. ONNX int64 indices must be cast by the
  future importer; the helper fails loud instead of reinterpreting bytes.
- [D-4] TopK exposes indices only; values require a layout-aware Gather.
  ggml explicitly does not guarantee index order, so tests compare sets.

## Requirements

- [R-1] Helpers compile into the GGML backend and are covered by tests.
- [R-2] Op-level parity vs numpy references is exact within 1e-5 (1e-4 for
  LayerNorm/attention); the tiny block is within 1e-5.
- [R-3] Existing `GGMLInferTest` models (ResNet-18/50, YOLOv8n detection,
  YOLOv8n-seg) remain green; full ctest stays green.
- [R-4] No RT-DETR/transformer PASS claim is made in this packet; the
  blocker and next step are recorded in `validation.md`.

## Open Questions

- The next packet should choose between a generic ONNX-subset importer and a
  purpose-built RT-DETR builder. The op layer is useful either way.
