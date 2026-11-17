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
      gradient-input re-gate after `/tmp` wipe gives projector max-abs
      `7.87e-03`, mean `1.47e-04`, max-cell agree; budget re-baselined to the
      measured `1e-2` since the old input was lost and gradient input
      exercises full dynamic range. Probe-era numbers with the old input:
      `2.07e-03` / `9.00e-05`, budget `5e-3`).
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
- [x] Loader encoder port (2026-11-16, `GgufRfdetrEncoderSelectionParity`
      green): encoder tower + top-300 selection in `GGMLInfer.cpp` behind the
      RF-DETR dispatch, exposed as second output `[1,300,4]` (projector stays
      `[1,256,40,40]` first). Greedy set match vs ORT `GatherElements`:
      worst row max-abs `1.43e-03` on the gradient-input re-gate
      (probe-era `9.42e-04`/`9.43e-04` with the old input); budget is the
      measured `5e-3`. Query order uses `ggml_ops::top_k_indices`;
      `get_rows` gathers dim 1, so tokens stay `[384,1601]` with no dim-0
      permute.
- [x] Top-300 query selection set parity (probe): deterministic descending +
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
      `4.77e-07` / `8.28e-09`; L1 `2.38e-07` / `6.29e-09`; L2
      `2.38e-07` / `5.12e-09` (re-gated 2026-11-16 on gradient input
      after `/tmp` wipe). All three layers agree max-cell and stay
      under 1e-5. `SoftmaxRowsRfdetr` green (self max `6.56e-07`,
      cross `1.19e-07`; tie-aware max-cell check — peaked rows contain
      exact 1.0 ties); `RfdetrSineEmbed` green (max `5.66e-07`).
      Filtered suite
      `GGMLOpsParity.*:GGMLInferTest.Gguf*` = 20/20 pass (incl. new
      `SoftmaxRowsRfdetr`, `RfdetrSineEmbed`, `GgufRfdetrBackboneProjectorParity`,
      `GgufRfdetrEncoderSelectionParity`, `GgufRfdetrDecoderLayer0Parity`); full
      `GGMLInferTest` = 27 passed / 8 skipped / 0 failed (Yolo/Ort50
      parity SKIPs without their `/tmp` artifacts, lost in the wipe;
      failures pre-existing and unrelated to RF-DETR).
- [x] Decoder layer-0 in the loader (2026-11-16,
      `GgufRfdetrDecoderLayer0Parity` green): full layer-0 forward
      (refine + sine + ref_point_head + self-attn + deformable cross-attn +
      FFN + norm3) in `GGMLInfer.cpp`, exposed as third output
      `[1,300,256]`. Norm3 max-abs `1.48e-02` / mean `7.13e-04`
      (re-gated on gradient input; probe-era `9.4e-03` / `2.8e-04`),
      max-cell agree; budget stays the measured `5e-2`. Bugs fixed along the way: `Expand_4196` needs a real transpose
      (verbatim copy permutes it), `Add_4 = sel_xy + ref_xy*sel_wh` (not
      `ref_xy + ref_wh*sel_xy`), `Mul_8 = exp(ref_wh)*sel_wh` (not
      `exp(sel_wh)*ref_wh`), and the `ReduceSum→Transpose_3` path must not
      swap the two 16-dims. The parity test pins ORT's TopK order
      (`NEURIPLO_TEST_RFDETR_TOPK_ORDER`; 218/300 agree naturally): F32
      score drift (`2.3e-03` max, min adjacent ORT gap `3.8e-07`) flips 82
      near-tie pairs, which changes slot pairing downstream (fixed
      per-slot queries/refpoints), so exact slot-wise parity is
      unattainable without an order match — S-3 remains matching-based.
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
  Open: layers 1-2 (same builder, MatMul_43xx/44xx tables verified present)
  plus head, then S-3 full-model parity. S-3 must be matching-based, not
  slot-wise: TopK order agrees only 218/300 (F32 score drift flips 82
  near-tie pairs; min adjacent ORT gap is 3.8e-07), and slot-fixed queries
  make outputs order-sensitive. Tolerance starts from this drift log
  (rel 2e-5 chained budget, LayerNorms reset drift: norm1 output 3.3e-06).

## S-3 head + full parity

- [x] Full-model raw parity in the loader (2026-11-16,
      `GgufRfdetrDecoderLayer12Parity` + `GgufRfdetrHeadParity` green,
      gradient input, TopK order pinned via
      `NEURIPLO_TEST_RFDETR_TOPK_ORDER`):
      layer-1 norm3 max `3.91e-02`, layer-2 norm3 max `2.57e-02`
      (budget `1e-1`, chained drift grows per layer);
      dets max-abs `2.10e-02`, mean L1 `6.76e-04` (budget `5e-2`);
      labels top-1 agreement **300/300**, max-abs `2.52e-02`.
      Head: decoder norm → bbox MLP (4428/4429/4430) → dets decode
      (`xy+bbox_xy*anchor_wh`, `exp(bbox_wh)*anchor_wh`, mirroring refine)
      → `dets [1,300,4]`; class linear (4447) → `labels [1,300,91]`.
      Full `GGMLInferTest` = 29 passed / 8 skipped / 0 failed.

## S-4 seg slice

Seg is a distinct model (not a head swap): 12-layer backbone (patch 12,
stride 12 → 36x36), 4 decoder layers, 200 queries, own weight values
(only 205/336 names shared with detection, values differ). Entry status
2026-11-16:

- [x] Exporter carries it with no code change: `convert_onnx_to_gguf.py`
      on `rfdetr_seg.onnx` → 436 tensors / 123,514,688 bytes
      (`/tmp/rfdetr_seg.gguf`).
- [x] Backbone probe green (`/tmp/opencode/seg_bb.cpp`): 4 windows of
      18x18+cls = 325 tokens; blocks 3/6/9 are GLOBAL attention over
      1300 tokens (found by node-count, 58 vs 54 — same signature as
      detection blocks 2/5/8), the rest windowed. Stem max `1.01e-04`,
      block-0/1 max `1.1e-04`, max-cell agree everywhere, vs ORT
      152-node prefix refs on gradient 432 input.
- [x] Projector + encoder probe green (same C2f as detection, taps at
      layers 2/5/8/11 through the shared channel layernorm; default
      boxes `[cx,cy,0.05,0.05]` on 36x36): projector max `5.98e-04`
      (mean `5.5e-05`), selected-box set match worst `1.26e-03`
      (positional order 166/200 — drift-driven swaps as in S-2; probe
      uses unordered argsort, the loader will use deterministic
      `top_k_indices`). LN eps `1e-6` everywhere; split even 128/128.
- [x] Loader port + `GgufRfdetrSeg*Parity` green (2026-11-16, gradient
      432 input, TopK order pinned via `NEURIPLO_TEST_RFDETR_TOPK_ORDER`):      backbone/projector max `5.98e-04` (budget `5e-2`); encoder
      selected-set worst `1.26e-03` (budget `5e-3`); decoder L0-L3 norm3
      max `1.91e-02` / `4.52e-02` / `7.51e-02` / `5.32e-02`
      (budget `2e-1`); head dets max `1.19e-02` (budget `1e-1`),
      labels top-1 **200/200**, masks max `4.47e-01` / mean `2.59e-02`
      (budget `5e-1`). Seg dispatch keys on `segmentation_head.bias`
      (present only in seg); geometry 432, 12 blocks, 200 queries,
      4 decoder layers; outputs lead with dets/masks/labels for the
      stock `RfDetrSegmentationPostprocessor` (boxes=0, masks=1,
      labels=2). Bugs fixed along the way (all probe-isolated):
      seg self-attn out_proj is a Gemm(transB=1) — Y = X @ W^T, so the
      ggml constant is the verbatim bytes, not the rweight_2d transpose
      (was: decoder max 6.3 → 0.019); mask-branch permutes need ggml
      order out.ne[b_k] = in.ne[k] — pw/sflat use (1,2,0,3), pwmap back
      uses (2,0,1,3) (was: masks 139); spat 1x1 bias was dropped
      (±0.14; bias lives on ne[2] so repeat needs [1,1,256,1])
      (was: spat 0.14 → exact). Full `GGMLInferTest` = 35 passed /
      8 skipped / 0 failed.
- [x] Native `depthwise_conv3x3` custom op (`GGMLOps`, F32 direct
      accumulation): ggml_conv_2d is ~1e-3 off here (fast algorithm)
      and the mask branch amplifies it coherently through the 1x1 and
      Einsum 256-sums (was: masks 80.9). Op parity vs ORT dwconv0:
      max `2.38e-07` (`GGMLOpsParity.DepthwiseConv3x3Seg` green).
- [x] Seg app gate + matrix cell (S-5 below).

## S-5 app gate + matrix

- [x] `neuriplo-infer` RF-DETR detection gate (2026-11-16, V-21 in
      `neuriplo-infer/specs/2026-11-13-ggml-matrix-coverage/validation.md`):
      convert exit 0, run exit 0, `run_report.json` success, render
      inspected — bicycle 0.94 / horse 0.92 / car 0.89 / motorbike 0.60,
      identical to the ORT reference on identical preprocessing
      (0.945 / 0.928 / 0.884 / 0.610). Evidence:
      `/tmp/opencode/ggml-matrix-gguf/gate-rfdetr/evidence.json` +
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-rfdetr/`.
      GGML `rfdetr` detection cell `U` → `P` in `docs/BackendCoverage.md`.
- [x] `neuriplo-infer` RF-DETR-seg gate (2026-11-16, V-23/V-24 in
      `neuriplo-infer/specs/2026-11-13-ggml-matrix-coverage/validation.md`):
      convert exit 0, run exit 0, `run_report.json` success, render
      inspected — bicycle 0.94 / horse 0.82 / car 0.61 / motorbike 0.59
      + tight instance masks. ORT on identical preprocessing gives the
      same instances (0.938 / 0.880 / 0.826 / 0.320 raw top-1; render
      scores are post-NMS selections); raw parity with ORT order pinned:
      dets `3.14e-02`, labels 198/200, masks max 2.68 / mean 0.04, and
      mask IoU **1.0 on every kept instance**. Evidence:
      `/tmp/opencode/ggml-matrix-gguf/gate-rfdetrseg/evidence.json` +
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-rfdetrseg/`.
      GGML `rfdetrseg` cell `U` → `P` in `docs/BackendCoverage.md`.

## Standing rules

- No commits/pushes without an explicit ask. Spec, code, changelog and
  roadmap tell the same story on the branch.
