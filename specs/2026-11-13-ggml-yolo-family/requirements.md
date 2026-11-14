# GGML YOLO conv family — requirements

**Date:** 2026-11-13
**Branch:** `feature/ggml-yolo-family` from `develop`
**Status:** Specified, not implemented
**Follows:** `2026-11-13-ggml-real-loader` (#74), `2026-11-13-ggml-resnet50-bottleneck` (#75)
**Program:** neuriplo-infer `2026-11-13-ggml-matrix-coverage` Phase 2 (TEST-ALL-THE-MODELS), rank 2

## Goal

Run YOLOv8n end to end through GGML with ORT parity on the raw output
tensor (`[1,84,8400]`), the way the ORT backend serves it. Decoding stays in
`neuriplo-tasks` (unchanged contract: backend returns raw tensors).

## In Scope

- Executor op extensions in `GGMLInfer`: SiLU, Concat/Split (views),
  Resize-nearest (upsample), Softmax, Reshape/Transpose/Permute, Slice,
  elementwise Mul/Add, Sigmoid (if the head needs it; verify from the export).
- YOLOv8n topology builder (CSP stem, C2f blocks, SPPF, PAN neck, decoupled
  head, DFL as exported — in-graph if the export bakes it, exact CPU
  equivalent if it does not; record which).
- Converter: carry all float initializers (already does); no layout change
  unless the export proves one necessary.
- Parity golden vs ORT on fixed inputs (letterboxed gray + zeros + one natural
  image if available): raw-tensor agreement, top-1 class agreement on decoded
  boxes optional (decoding is tasks-owned).

## Out of Scope

- yoloseg mask head, yolo26, NMS (tasks-owned), dynamic shapes, GPU, other
  backends, ABI changes, pin moves.

## Decisions

- [D-1] Raw-tensor parity is the exit bar (backend contract), not mAP.
  Decoding correctness belongs to neuriplo-tasks and its existing gates.
- [D-2] Fixed `1x3x640x640` NCHW input (matches the gated preset shape).
- [D-3] DFL placement follows the export: if ultralytics bakes DFL into the
  graph, implement it in-graph; explicit CPU DFL only if the export leaves it
  out (verify by node dump before coding).

## Requirements

- [R-1] `yolov8n.onnx` (ultralytics export, opset 12, simplify, static 640)
  converts to NGGL with all float initializers.
- [R-2] Full YOLOv8n graph executes in GGML: stem/C2f/SPPF/neck/head with the
  extended op set; no placeholder tensors.
- [R-3] Parity vs ORT on the raw `[1,84,8400]` output: max-cell agreement +
  max-abs-diff ≤ 1e-2 (adopted 2026-11-14; measured 2.01e-03 gray /
  9.15e-04 zeros / gradient case added per review).
- [R-4] ResNet-18/50 goldens still green (no regression).
- [R-5] Unknown ops/shapes fail loud with path + node/seen context.
