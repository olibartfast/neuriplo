# Implementation Plan — RF-DETR GGUF support (detection + seg)

Branch: `feature/ggml-rfdetr`. Requirements:
`specs/2026-11-16-rfdetr-gguf/requirements.md`. Source facts below measured
2026-11-16 against `/home/oli/.cache/neuriplo-op-inventory/op18/rfdetr.onnx`
(1,974 nodes, 336 inits, `1x3x560x560` → `dets[·,300,4]` + `labels[·,300,91]`).

## Architecture (measured)

- Backbone = DINOv2-S ViT: patch-embed Conv `[384,3,14,14]` (560/14 = 40x40
  = 1600 tokens), ~12 transformer blocks (26 LayerNorms), embed 384.
- Neck encoder, nodes 856..1061 (1 LayerNorm).
- Query selection: TopK @1017 → GatherElements @1025 (encoder top queries,
  same pattern as the RT-DETR chain).
- Decoder: 3 deformable layers (GridSample @1413/@1653/@1892 in cross-attn),
  10 LayerNorms total.
- Box decode: anchor/offset split with Exp on wh (DINO form). Class head:
  decoder norm @ `onnx::MatMul_4447 [256,91]` + `class_embed.bias [91]`.
  Anonymous `onnx::` initializer names must survive export verbatim.
- 26.9M params → ~102 MB fp32 GGUF.

## Slices (each lands only with its exit bar)

- S-0 discovery (this packet): section boundaries, dims, GridSample sites,
  head wiring — done above. Open: exact per-block tensor names for the
  backbone builder (enumerate in S-1).
- S-1 backbone ViT → GGUF: patch-embed + 11 blocks (QKV MatMul, Softmax,
  LayerNorm, GELU; blocks 2/5/8 use global attention, the rest windowed) +
  projector C2f + exporter shape rule for `onnx::` names. Status 2026-11-16:
  scratch probe parity green (layer 1/4/7/10 max-abs 3.65e-05 / 8.82e-05 /
  4.32e-04 / 5.98e-03; projector 2.07e-03, max-cell agree everywhere).
  Backend port is landed in `GGMLInfer.cpp` behind the RF-DETR dispatch
  (app-facing outputs lead with `dets`/`labels`).
- S-2 neck + deformable decoder: native GridSample + GatherElements/TopK
  query selection; encoder-output then decoder-output parity. Status
  2026-11-16: all loader-green — encoder selection set match `1.43e-03`,
  GridSample L0/1/2 ≤ `4.77e-07`, layer-0 norm3 `1.48e-02` (TopK order
  pinned; natural order agrees 218-232/300 with drift-driven near-tie
  swaps, documented, order-sensitive slots).
- S-3 head + full parity: decoder layers 1-2 share one parametrized
  builder with layer 0 (MatMul 43xx/44xx tables); head is decoder-norm →
  bbox MLP + class linear with anchor-mirroring box decode. Exit met
  2026-11-16: L1/L2 norm3 `3.91e-02`/`2.57e-02`, dets `2.10e-02`,
  labels top-1 **300/300** (`GgufRfdetrDecoderLayer12Parity`,
  `GgufRfdetrHeadParity` green; full suite 29/8/0).
- S-4 seg slice: seg dispatch + 12-block/200-query/4-layer parametrization +
  mask branch (bilinear Resize, exact dwconv3x3, spatial proj + bias,
  query MLP, Einsum + bias). Exit met 2026-11-16: decoder L0-L3
  `1.91e-02`/`4.52e-02`/`7.51e-02`/`5.32e-02`, dets `1.19e-02`, labels
  top-1 **200/200**, masks `4.47e-01`/`2.59e-02` (`GgufRfdetrSeg*Parity`
  green; full suite 35/8/0).
- S-5 app gate + matrix: `neuriplo-infer --type=rfdetr` gate green
  2026-11-16 (V-21: exit 0, detections identical to ORT) and the GGML
  `rfdetr` cell moves `U` → `P`; `--type=rfdetrseg` gate green
  2026-11-16 (V-23: exit 0, mask IoU 1.0 vs ORT on every kept instance)
  and the GGML `rfdetrseg` cell moves `U` → `P` (V-24).

## Risks and mitigations

- Hand-written ~2k-node-equivalent topology (decoder unrolled 3x): keep one
  parametrized block builder, parity after each block.
- Deformable GridSample numerics: unit-parity the native op vs ORT first
  (same A/B harness as the RT-DETR GridSample probe, which was bit-exact).
- v1.0.0 conv drift amplified through Exp/box decode: per-layer drift log
  (S-2 exit), tolerance from measurement, never aspiration.
- 91-class postprocess + task routing (`rfdetr` type) lives in
  neuriplo-tasks / neuriplo-infer: gate will expose gaps; fix there, not in
  the loader.

## Validation

Per-slice exit bars above; `validation.md` written before S-1 code, updated
per slice. No matrix promotion without a recorded gate. No commits/pushes
without an explicit ask.
