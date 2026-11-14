# GGML YOLOv8n instance segmentation (`yolov8n-seg`) — requirements

**Date:** 2026-11-14
**Branch:** `feature/ggml-yolo-family` from `develop`
**Status:** Implemented; validation green (2026-11-14)
**Follows:** `2026-11-13-ggml-yolo-family` (YOLOv8n detection on GGML, #76)
**Program:** neuriplo-infer `2026-11-13-ggml-matrix-coverage` Phase 2 (TEST-ALL-THE-MODELS)

## Goal

Run YOLOv8n instance segmentation end to end through the GGML backend and
return the same raw output tensors as `ONNX_RUNTIME`, so `neuriplo-infer` can
later select `--type=yoloseg` through GGML. Mask/proto decoding and NMS remain
in `neuriplo-tasks`; the backend returns raw tensors only.

## Raw-output contract

The ONNX artifact (`/tmp/opencode/ggml-matrix/yolov8n-seg.onnx`, ultralytics
export, opset 12, static 640, simplified) declares:

- `output0` — `float32[1,116,8400]`, flat NCHW order
  (channels `0..3` box/DFL-decoded, `4..83` class scores, `84..115`
  mask coefficients).
- `output1` — `float32[1,32,160,160]`, flat NCHW order (prototype masks).

GGML must return both tensors, in that order, with exactly those shapes and
the same flat element order. ResNet keeps its existing single `{N,1000}`
output and YOLOv8n detection keeps its single `{1,84,8400}` output.

## In Scope

- Multi-output support internal to `GGMLInfer`: per-output shapes, metadata,
  graph expansion, and result return for every output tensor.
- Container detection from exact ultralytics tensor names
  (`model.22.proto.cv1.conv.weight`, `model.22.cv4.0.0.conv.weight`), not from
  the file name/extension.
- Segmentation head extensions reusing the detection backbone/neck/head:
  - per-scale mask-coefficient branches `model.22.cv4.{0,1,2}`
    (3x3 C->32 + SiLU, 3x3 32->32 + SiLU, 1x1 32->32, no activation),
    concatenated along the flat spatial dim to `32 x 8400`;
  - proto branch from P3: `model.22.proto.cv1` (3x3 64->64 + SiLU),
    `model.22.proto.upsample` (ConvTranspose 2x2 stride 2, 64->64 + bias),
    `model.22.proto.cv2` (3x3 64->64 + SiLU),
    `model.22.proto.cv3` (1x1 64->32 + SiLU) -> `[1,32,160,160]`.
- ORT reference generation for both outputs from stock onnxruntime 1.32.
- Parity test `ParityVsYoloSeg` on the fixed gray/zeros/gradient inputs.

## Out of Scope

- `yolo26` and any other family (next sibling step, not this packet).
- Mask/box decoding, NMS, and any pre/postprocessing in the backend.
- Batch > 1 for YOLO or YOLO-seg (fixed batch-1 contract).
- Dynamic shapes, GPU/GGML offload, other backends, CLI/API changes beyond
  the internal multi-output plumbing, pin moves, app-surface edits.

## Decisions

- [D-1] Raw-tensor parity is the exit bar. Backend output schema is
  `[output0, output1]` with the ONNX shapes; decoding stays tasks-owned.
- [D-2] Fixed `1x3x640x640` NCHW F32 input, batch 1.
- [D-3] Detection classification output stays sigmoided and the box output
  stays DFL-decoded exactly as the exported graph bakes them; `output0`
  mirrors the ONNX concat order box(64) + class(80) + mask(32).
- [D-4] ConvTranspose runs through `ggml_conv_transpose_2d_p0` with an F16
  kernel (the installed ggml CPU kernel asserts F16 weights); the ONNX
  `[Cin,Cout,kH,kW]` weight is transposed into ggml's
  `[kW,kH,Cout,Cin]` layout and the orientation is probed numerically before
  the full graph is trusted.

## Requirements

- [R-1] The segmentation export converts to NGGL with every float
  initializer (including the ConvTranspose kernel and proto branch) carried.
- [R-2] The segmentation graph loads and runs both outputs; unknown
  ops/tensors/shapes fail loud with the model path and tensor/node names.
- [R-3] `output0` parity vs ORT on fixed inputs: max-cell agreement on the
  full `116x8400` tensor and max-abs-diff within the adopted tolerance.
- [R-4] `output1` parity vs ORT on fixed inputs: elementwise max-abs-diff
  within the adopted tolerance from the same runs.
- [R-5] ResNet-18/50 and YOLOv8n detection outputs, shapes, and parity
  goldens remain unchanged and green.
- [R-6] `get_infer_results` returns both outputs in ONNX order with their
  declared shapes; single-output models return exactly one output as before.
- [R-7] No new dependency, no pin move, no other backend change, and no
  commit/push in this packet's worktree.
