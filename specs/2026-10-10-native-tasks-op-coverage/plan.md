# Plan — neuriplo-tasks Operator Coverage on the NERT

One tier per phase; each ends with its families loading and matching
`ONNX_RUNTIME` through the model-zoo parity test ([V-3]). Branch:
`feature/nert`.

## Tier 1 — YOLO completion

- [T-1] Kernels, shape rules, loader entries and unit tests: Floor, Identity,
  Pow, Tile, AveragePool, ConvTranspose.
- [T-2] Load-time constant folding pass ([D-4]), called by `NertInfer`
  between load and planning.
- [T-3] DOUBLE narrowing ([D-5]); identical duplicate value-info ([D-6]);
  Resize `sizes` ([D-7]).
- [T-4] Model-zoo parity test `nert_model_zoo_parity_test`, driven by a
  manifest (`NEURIPLO_MODEL_ZOO_MANIFEST`); modes `parity`, `e2e`, `load`.

## Tier 2 — Transformer encoders

- [T-5] Erf, LayerNormalization, Sqrt, Squeeze, Tanh, IsNaN.
- [T-5a] Non-operator gaps the survey surfaced: symbolic dims ([D-8]), linear
  Resize ([D-9]), BOOL constants ([D-10]), dtype-preserving Reshape
  ([D-11]), 3-D Conv ([D-12]), evaluator coverage ([D-13]), Expand on bool.

## Tier 3 — DETR family

- [T-6] Resolve [Q-2] (int64 graph inputs) first.
- [T-7] GridSample, ReduceSum, Range, Clip, Log, Pad, Abs, Neg, Exp, Cos, Sin,
  Einsum, And, Not, Greater, Less; int64 compute where the exports need it.

## Tier 4 — RAFT

- [T-8] InstanceNormalization; resolve [Q-3].

## Tier 5 — Open vocabulary

- [T-9] ArgMax, Elu, LessOrEqual, ReduceL2, CumSum, EyeLike, GreaterOrEqual,
  Max, Min, Or, ScatterND; int64 text inputs per [Q-2].

## Close-out

- [T-10] Resolve [Q-1] and [Q-4]; update the roadmap gate, CHANGELOG and the
  PR description; lift the merge hold only on the maintainer's call.
