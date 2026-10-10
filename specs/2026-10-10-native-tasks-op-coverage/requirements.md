# Requirements — neuriplo-tasks Operator Coverage on the NERT

Maintainer decision (2026-10-10): `feature/native-engine-cpu-spine` merges into
`develop` only once `NERT` implements every operator needed by these
neuriplo-tasks families (recorded as the merge gate in `specs/roadmap.md`,
Phase N0):

- YOLO detection, every supported version, including the SiLU, HardSwish,
  LeakyRelu, Clip and pooling variants; YOLO segmentation and pose heads.
- ViT classification; ViTPose; Depth Anything V2; VideoMAE, ViViT and
  TimeSformer.
- RT-DETR, D-FINE, DEIM, DEIMv2, RF-DETR and RF-DETR-Seg.
- RAFT optical flow.
- OWLv2 and Grounding DINO.

Gaussian splatting (LGM, GRM) and VLM (llama.cpp) are outside the gate.

## Inventory (2026-10-10)

26 ONNX exports were surveyed: the existing `~/model_repository` exports, the
neuriplo-tasks `export/` recipes run as written (Grounding DINO with the
`-tiny` checkpoint; same op set as `-base`), and Ultralytics YOLO11/YOLO26
seg and pose. Engine op set at survey time: 31 kernels plus folded `Constant`.

40 operators were missing, grouped so each tier completes whole families:

| Tier | Operators | Completes |
| --- | --- | --- |
| 1 | ConvTranspose, Pow, Tile, AveragePool, Identity, Floor | YOLOv5/v6/v7/v9/v10, YOLO11/26 seg |
| 2 | Erf, LayerNormalization, Sqrt, Squeeze, Tanh, IsNaN | ViT, ViTPose, Depth Anything V2, VideoMAE, ViViT, TimeSformer |
| 3 | GridSample, ReduceSum, Range, Clip, Log, Pad, Abs, Neg, Exp, Cos, Sin, Einsum, And, Not, Greater, Less | RT-DETR v4, D-FINE/DEIM, DEIMv2, RF-DETR, RF-DETR-Seg |
| 4 | InstanceNormalization | RAFT |
| 5 | ArgMax, Elu, LessOrEqual, ReduceL2, CumSum, EyeLike, GreaterOrEqual, Max, Min, Or, ScatterND | OWLv2, Grounding DINO |

Gaps that are not operators, found by the same survey:

- **Opset.** The loader accepts ONNX-domain opset 18 only; 23 of the 26
  exports declare 12–17.
- **Integer graph inputs.** DETR-family exports take `orig_target_sizes`
  (int64); OWLv2 and Grounding DINO take int64 `input_ids` and
  `attention_mask`. Graph I/O is float32-only.
- **Integer compute.** Range, CumSum, ScatterND and the text branches compute
  on int64 data that is not a load-time constant.
- **Dynamic spatial dims.** RAFT exports `height`/`width` as symbolic; most
  others have only a symbolic batch, which `input_sizes` fixes at load.
- **Attribute policies.** Several admitted ops accept only the configuration
  the first fixtures used (for example Resize: nearest/asymmetric/floor). A
  model can name only admitted ops and still be rejected.
- **Adapter contract.** `NertInfer` takes full shapes in `input_sizes`;
  `ORTInfer` takes per-input dims without the batch axis plus `batch_size`.

## Scope

In:

- [R-1] Every operator in tiers 1–5 has a CPU reference kernel, shape
  inference, a loader allowlist entry with its opset-18 attribute policy, and
  hand-computed unit tests.
- [R-2] Every gated family loads and runs on `NERT` from its export,
  converted to opset 18 where needed ([D-2]).
- [R-3] Outputs match `ONNX_RUNTIME` on identical input: every element within
  `|native - ort| <= atol * (1 + |ort|)`, `atol = 1e-3` unless the packet
  records a family-specific bound; end-to-end detection exports compare
  confident rows only, on a real image ([D-3]).
- [R-4] The non-operator gaps above that block a gated family are closed or
  carry a recorded decision.

Out:

- Performance; the CPU kernels stay unoptimized oracles (Phase N1 owns speed).
- Families outside the gate; ONNX ops outside the surveyed exports.

## Decisions

- [D-1] Tiers land in order 1→5; each tier ends with its families passing
  [V-3] before the next starts.
- [D-2] Interim opset policy: exports are converted to opset 18 with
  `onnx.version_converter` for validation. Whether the loader accepts other
  opsets natively is [Q-1].
- [D-3] End-to-end detection exports (NMS or TopK inside the graph) compare
  confident rows on a real image, as packet `2026-10-01-yolo26-native`
  restated on 2026-10-10: near-tied background scores select
  backend-noise-dependent rows on random input.
- [D-4] Load-time constant folding: a node whose inputs are all initializers
  runs once at load with the CPU kernels and becomes an initializer. Older
  exports compute Resize sizes and anchor grids this way; folding reuses
  InferShapes on a one-node graph instead of growing the static evaluator.
- [D-5] DOUBLE constants, initializers and value-info entries load as float32.
  Compute is float32-only, and the surveyed exports use float64 only for
  constant subgraphs that end in a Cast to FLOAT (YOLOv6 anchor grids).
- [D-6] An identical repeated value-info entry is accepted; a conflicting one
  is rejected. torch 2.12's exporter writes every entry twice, which made the
  pinned YOLO26 fixture fail to load.
- [D-7] Resize accepts a `sizes` input as well as `scales`, never both.
- [D-8] A named or unknown dimension (`dim_param`, or no value) loads as -1;
  the caller fixes it per load through `input_sizes` (full shapes for
  `NERT`), and `NertInfer` names the input when it is missing. An
  explicit non-positive `dim_value` is still rejected. This resolves [Q-3]
  for symbolic batch dims; RAFT's spatial dims follow the same rule.
- [D-9] Resize implements `nearest` and `linear` with the four plain
  coordinate transforms and all four nearest roundings, with the ONNX
  defaults (`half_pixel`, `round_prefer_floor`) when an attribute is absent.
  Before this, an absent attribute was treated as `asymmetric`/`floor`;
  every fixture that relied on that sets both explicitly. Cubic,
  `tf_crop_and_resize`, antialiasing and `axes` are rejected at load.
- [D-10] BOOL initializers and Constants load (one byte per element), so
  attention masks and boolean constant folding work.
- [D-11] Reshape keeps its data dtype (float32, int64, or bool); it was
  float32-only, which broke int64 shape tensors reshaped in-graph.
- [D-12] Conv covers one, two, or three spatial axes (video tubelet
  embeddings use 3-D Conv); 1-D and 2-D run as 3-D with unit leading axes.
- [D-13] The load-time shape evaluator also follows Concat, 1-D Slice, Cast
  to int64, and the layout-only ops (Identity, Squeeze, Unsqueeze, Reshape,
  Flatten), and Shape honours `start`/`end` (opset 15).

- [D-14] Graph inputs and outputs widen to int64 and bool (maintainer,
  2026-10-10): the DETR exports take `orig_target_sizes`, and OWLv2 and
  Grounding DINO take int64 token ids and masks. Packet P3e.

- [D-15] Float reductions in the reference kernels (MatMul, Gemm, Conv,
  Einsum, ReduceSum, ReduceMean, Softmax) accumulate in double and round to
  float32 once per output. A correctness oracle should add as little rounding
  of its own as possible; tensor dtypes do not change. Introduced by P3f;
  accepted after the YOLO and tier 2 regression.
- [D-16] DETR-family exports (query outputs `[1,Q,4]` boxes in cxcywh and
  `[1,Q,C]` class logits, optional `[1,Q,H,W]` mask logits) compare
  confident queries on a real image, extending [D-3]: queries whose ONNX
  Runtime score (sigmoid of the max logit) exceeds 0.5, matched one-to-one
  by class with box IoU >= 0.99 and score difference <= 1e-3; matched masks,
  binarized at 0, reach IoU >= 0.98; the confident-query counts are equal.
  The query-selection TopK in these decoders breaks near-ties (1e-6) by
  rounding, so elementwise comparison of every query is not meaningful
  (P3f, 2026-10-10). Planner decision; the maintainer may overrule.

## Open questions

- [Q-1] Accept opsets 12–17 natively (per-op attribute-vs-input variants), or
  require opset-18 exports and document the conversion? Recommendation:
  require 18 and ship a conversion note; owner: maintainer.
- [Q-2] Resolved 2026-10-10 as [D-14].
- [Q-3] Dynamic spatial dims for RAFT: fix them through `input_sizes` (static
  per load), or wait for the Phase N2 plan cache? Recommendation: static per
  load. Owner: maintainer.
- [Q-4] The model zoo is large local exports, not CI fixtures. Which subset,
  if any, is provisioned in CI? Owner: maintainer.
