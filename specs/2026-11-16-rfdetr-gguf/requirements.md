# Feature Requirements — RF-DETR GGUF support (detection + seg)

Branch: `feature/ggml-rfdetr` (stacked on `feature/ggml-yolo-family` @ strict-GGUF-only).
Date: 2026-11-16.

## Goal

Native GGUF support for the RF-DETR family in the GGML backend: detection
(`rfdetr-medium`, 560px) first, segmentation (`rfdetrseg`) second. Each slice
lands only with ORT parity plus an app gate; anything else stays `U`.

## Source facts (measured 2026-11-16, `rfdetr.onnx`)

- 1,974 nodes, 336 initializers; input `1x3x560x560`; outputs `dets[·,300,4]`,
  `labels[·,300,91]`.
- Heavy ops: Constant 656, Add 186, MatMul 135, Mul 126, Reshape 121,
  Transpose 85, Concat 71, LayerNormalization 37, Softmax 17,
  GridSample 3 (deformable attention), GatherElements 1, TopK 1,
  Where/Tile/Expand, Sin/Cos (positional encoding).

## Scope

- R-1 detection slice: backbone → encoder → 300-query decoder → `300x4` box
  + `300x91` label head as a hand-written GGUF topology in `GGMLInfer.cpp`
  (no generic importer; strict-GGUF-only stands). Reuse `GGMLOps`
  (LayerNorm, erf-GELU, SDPA) where parity holds; re-implement GridSample /
  GatherElements / TopK natively.
- R-2 seg slice: mask head on the same packet only after R-1 parity is green.
- R-3 exporter: extend `convert_onnx_to_gguf.py` head contract past the
  YOLO/classifier rule (RF-DETR has no `model.22.dfl.conv.weight` and no
  `[1000,K]` classifier); loader + exporter stay in lockstep.
- R-4 evidence: sibling `GGMLInferTest` parity (label agreement + box L1 vs
  ORT) and `neuriplo-infer` app gate per slice before any matrix cell moves.

## Out of scope

- Touching the shipped strict-GGUF ResNet/YOLO paths except to share helpers.
- Transformer models beyond RF-DETR (ViT, Depth, OWLv2, video).
- Re-introducing any ONNX-at-runtime frontend.
- Commits/pushes without an explicit ask (repo policy).

## Decisions

- Detection + seg in one packet, detection lands first (maintainer scope,
  2026-11-16).
- Strict-GGUF base committed first (neuriplo `842fb76`, neuriplo-infer
  `aa5b00b`, neuriplo-tasks `093f8cc`); this branch starts from it.
- `rfdetr-medium` @ 560px is the representative (matches the gate table).

## Constraints

- Fixed batch-1, fixed 560px geometry; `dets`/`labels` shapes above are the
  contract.
- Existing parity bar applies: max-cell agreement plus abs-diff tolerance
  set from measured drift (v1.0.0 conv drift amplified through the decoder
  is expected; no PASS on aspiration).
