# Validation — RF-DETR GGUF support (detection + seg)

Written before implementation; execute from this file. Pyramid scoring:
worker runs each acceptance command once per attempt and reports pass/fail;
planner decides what happens next.

## Method (shared by all slices)

- ORT reference: same preprocessed input through stock onnxruntime on the
  source ONNX; dump raw float outputs (`.npy`/`.bin`).
- GGUF candidate: same input through `GGMLInfer` on the converted `.gguf`;
  dump raw outputs the same way.
- Compare: max-cell agreement (argmax over all elements must match) and
  max-abs-diff. No PASS without both numbers recorded here.
- Drift budget comes from measurement (v1.0.0 conv drift ~1e-5/element,
  stride-amplified through Softmax/Exp/decode), never from aspiration.

## Pre-S-1 loader safety (2026-11-16)

Independent review of the uncommitted GGUF parser found a blocking P1:
tensor offset/end arithmetic and dimension multiplication could wrap and
permit out-of-bounds reads. Fixed in `backends/ggml/src/GGMLInfer.cpp`:

- [x] Checked metadata lengths and 32-byte data-base alignment.
- [x] Bounded element counts by file size before allocation; checked
      tensor offset plus size without wrapping.
- [x] Rejected duplicate rank entries; kept zero-sized ONNX initializers
      (the converter carries them and the topology ignores them).
- [x] Regression fixtures in `GGMLInferTest`:
      `GgufRejectsOverflowingTensorOffset`,
      `GgufRejectsOverflowingTensorDimensions`,
      `GgufAcceptsZeroSizedTensorDimension`,
      `GgufRejectsDuplicateRankEntry`, `GgufAcceptsLongTensorName`.
      All five pass (`GGMLInferTest --gtest_filter='GGMLInferTest.Gguf*'`).
- [x] Full `GGMLInferTest` still reports 21 passed / 5 skipped / 3 failed
      (the 21st pass is the new `GGMLOpsParity.TopKStableTieBreak`).
      The same three parity failures (`ParityVsYolo`, `ParityVsYoloSeg`,
      `ParityVsOrt50`) reproduce with the committed `HEAD` parser, so they
      predate the parser rewrite and are tracked separately from S-1.
      (2026-11-16, with S-2 `GridSampleRfdetr` landed: 22 passed / 5 skipped
      / 3 failed — the same three.)

## S-1 backbone ViT (entry: plan.md S-1)

Probe build (scratch `rfdetr_bb_proj.cpp`, ported into `GGMLInfer.cpp`
2026-11-16 behind the RF-DETR dispatch): `rfdetr.onnx` → `/tmp/rfdetr.gguf` with the updated
converter, fixed-batch input `/tmp/opencode/rfdetr_bb_input.npy`, ORT 1.32.0
reference from a 852-node prefix extraction. The probe now covers the full
11-block DINOv2 backbone (windowed attention plus global attention in blocks
2/5/8), the channel-LayerNorm at layers 1/4/7/10, and the projector.

- [x] Exporter carries all 336 initializers incl. anonymous `onnx::` 2D
      weights (all FLOAT, ranks 1..4 — verified 2026-11-16). It also carries
      the 16 FLOAT graph-Constant tensors the projector needs (`onnx::Mul_*`
      / `onnx::Add_*` channel-norm affine pairs), for 352 tensors / 107,591,104
      bytes.
- [x] Backbone-output parity vs ORT, max-cell agreement and max-abs-diff:
      layer 1 `3.65e-05` / agree; layer 4 `8.82e-05` / agree; layer 7
      `4.32e-04` / agree; layer 10 `5.98e-03` / agree.
- [x] Projector-output parity vs ORT: max-cell agree, max-abs-diff `2.07e-03`,
      mean-abs-diff `9.00e-05`.
- [x] `convert_onnx_to_gguf.py` RF-DETR head rule present (R-3), old
      YOLO/classifier rule untouched.
- [x] Port the validated backbone/projector topology from the probe into
      `GGMLInfer.cpp` (2026-11-16, `GgufRfdetrBackboneProjectorParity` green:
      projector max-abs `2.07e-03`, mean `9.00e-05`, max-cell agree — the
      loader number reproduces the probe exactly; S-1 budget is the measured
      `5e-3`, ~2.4x chained drift, per the drift-budget method above).
      Exposed as the documented S-1 intermediate contract
      (`output_shapes_ = {{1,256,40,40}}`); the full detection graph (encoder +
      decoder + head) still waits for S-2 so the backend never returns a
      partial RF-DETR graph. Recipes carried over verbatim: tokens via
      `get_rows` on dim 1 (`[384,1601]` straight in, no dim-0 permute),
      `Transpose_2/Reshape_5`-style real permutes, `[1,0,2,3]` sine order.

Probe artifacts: `/tmp/opencode/rfdetr_bb_proj.cpp` (source),
`/tmp/rfdetr.gguf` (352-tensor GGUF), `/tmp/opencode/rfdetr_proj_got.bin`
(projector output), `/tmp/opencode/ref__backbone_*` (ORT refs).

## S-2 neck + deformable decoder

Probe status (2026-11-16): the encoder tower and query selection are now in
the scratch probe and compared against a 1053-node ORT prefix extraction.

- [x] Source-token layout: projector `[W,H,C]` → `[C,1600]` matches ORT
      `transformer/Concat_1` (max `2.07e-03`, mean `9.00e-05`).
- [x] Encoder linear + LayerNorm (eps 1e-5): max `1.30e-03`.
- [x] Encoder class / box heads: class max `2.34e-03`, decoded boxes max
      `1.69e-03` (default boxes `[cx,cy,0.05,0.05]` at 40×40).
- [x] Top-300 query selection set parity: deterministic descending +
      index-ascending tie-break via `GGML_OP_CUSTOM` CPU kernel landed;
      `GGMLOpsParity` 8/8 incl. new `TopKStableTieBreak` `{5,3,5,1}` →
      `{0,2,1}`; backend probe rows `[5,3,5,1]`/`[2,2,9,9]` k=3 → `0 2 1` /
      `2 3 0`; residual 82/300 order swaps are numeric drift (score drift max
      `1.42e-03`, row diff `9.43e-04`, 300 distinct ORT values, zero exact
      ties, all adjacent gaps <2×drift), set match 300/300. Order is not
      claimed as 300/300.
- [x] Native GridSample / deformable-attention parity vs ORT
      (`GGMLOpsParity.GridSampleRfdetr`, `GGML_OP_CUSTOM` CPU kernel in
      `backends/ggml/src/GGMLOps.cpp`). Layout memo: the ONNX grid
      `(N=16,Hout=300,Wout=2,2)` becomes ggml `[2,Wout,Hout,N]` =
      `[2,2,300,16]`; the axis reversal only renames axes, so the raw
      row-major reference dumps stay in ggml flat order. Per-layer
      max-abs / mean-abs diff (tolerance 1e-5): L0
      `4.77e-07` / `6.68e-09`; L1 `2.38e-07` / `5.06e-09`; L2
      `2.38e-07` / `3.94e-09`. All three layers agree max-cell and stay
      under 1e-5. Filtered suite
      `GGMLOpsParity.*:GGMLInferTest.Gguf*` = 15/15 pass; full
      `GGMLInferTest` = 22 passed / 5 skipped / 3 failed, the three
      failures being the pre-existing
      `ParityVsYolo`/`ParityVsYoloSeg`/`ParityVsOrt50` noted above.
- [x] Decoder-output parity; per-layer drift log (2026-11-16, probe green).
- Decoder layer-0 mapping complete (2026-11-16, probe built and green):
  `Tile_1`/`Expand_4178` and refine `Expand_4196` are learned constants in
  GGUF (verified present with all layer-0 MatMuls/biases); refine
  `dets=xy+wh*emb_xy, wh*exp(emb_wh)` exact vs ORT (`1.19e-07` max when fed
  ORT `Concat_9`); sine-embed coord order `[1,0,2,3]` verified (`4.25e-07`
  vs `2.00` for `[0,1,2,3]`); self-attn is DETR `q=k=Add, v=Tile_1`
  (8x32); cross-attn grid/weight permutes worked out for
  `ggml_ops::grid_sample`; residual base of `Add_3` is `norm1`, not `Add_1`.
  ORT refs: 48 tensors in `/tmp/opencode/dec0/` + `/tmp/opencode/dec0x/`,
  plus `layers.0` FFN/norm3 + int64 TopK + encoder `Concat_8` from the
  `/tmp/opencode/rfdetr_dec0_ffn.onnx` extraction.
  Probe `/tmp/opencode/rfdetr_dec0.cpp` (scratch, loader untouched; run log
  `/tmp/opencode/rfdetr_dec0_run.log`, exit 0, `DEC0 PROBE ALL GREEN`):
  query row order uses `ggml_ops::top_k_indices` (set 300/300, order exact
  vs int64 ORT indices when fed ORT scores); `GatherElements` via
  `ggml_get_rows` exact `0.0`; `Expand_4178` bytes == `Tile_1` exact.
  Pass bar: max-cell agree everywhere AND (abs<=1e-5 OR rel<=2e-5);
  isolated-op F32 noise measures ~1e-6 rel, worst chained cell 1.33e-05 rel.
  Per-op max-abs / rel vs ORT (fed ORT prefix refs unless noted):
  sine `Concat_8` 6.04e-07; ref-point-head l0 1.14e-05 / 8.1e-07,
  l1 1.43e-06; self-attn q/k/v 5.2e-06 / 3.8e-06 / 4.8e-06,
  scores 1.71e-05 / 3.4e-07, probs 8.3e-07, context 3.24e-05,
  out-proj 3.36e-04 / 7.5e-07 (same fed ORT context: linear noise, not
  amplification); `Add_1` 3.36e-04, norm1 3.34e-06, `Add_2` 3.34e-06;
  value-proj 2.62e-06 (+`Reshape_2`/`Reshape_3` views same);
  sampling-offsets 1.22e-04 / 5.9e-07; grid `Add` 8.58e-06;
  `Gather_5`/`Transpose_1`/`Reshape_4` 1.72e-05;
  `grid_sample` 5.05e-05 / 1.33e-05 chained (4.77e-07 fed ORT inputs,
  kernel proven); attn-weights 8.3e-07, softmax 2.38e-07;
  `Mul_6` 2.80e-05, `ReduceSum`/`Transpose_3` 3.51e-05,
  output-proj 1.26e-05, `Add_3` 1.27e-05; FFN l1 3.55e-05,
  relu 2.92e-05, l2 2.31e-05, `Add_4` 2.61e-05,
  norm3 (layer-0 output) 1.20e-05 / 2.3e-06.
  Port inputs recorded: 5D/6D index plumbing stays in C++ loops (ggml is
  <=4D); softmax-over-last-dim needs a rows-helper or the eye/transpose
  pattern used here; `Transpose_3`/`Transpose_1`/value-transpose are real
  permutes (byte views documented per op); `Slice_2` is an identity view
  (single 40x40 level); `Concat_9` == `GatherElements` bytes,
  decoder `Slice` == `Concat_10` bytes, `A0` == `Tile_1` + ref-point-head-l1
  bytes (all 0.0). Backend port stays batched (no partial graph).
  Open: port validated layer-0 (+ layers 1-2, same builder) into
  `GGMLInfer.cpp`; S-3 tolerance starts from this drift log (rel 2e-5
  chained budget, LayerNorms reset drift: norm1 output 3.3e-06).

## S-3 head + full parity

- [ ] Full-model raw parity: 300/300 label agreement target, box L1
      recorded; tolerance set from the S-2 drift log.

## S-4 seg slice

- [ ] Same bar on `rfdetr_seg.onnx` mask outputs.

## S-5 app gate + matrix

- [ ] `neuriplo-infer` gates per slice (exit 0 + render + run_report),
      evidence JSON paths recorded here; matrix cell moves only then.

## Standing rules

- No commits/pushes without an explicit ask. Spec, code, changelog and
  roadmap tell the same story on the branch.
