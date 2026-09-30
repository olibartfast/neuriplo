# Requirements — YOLO26 Detection on the Native Engine

Maintainer decisions (2026-10-01): variant **YOLO26 nano**, export **NMS-embedded**
(`nms=True`), success bar **raw-output parity** (NATIVE vs ONNX_RUNTIME
elementwise on identical input). Merge into `develop` stays held until parity
passes (see the merge-hold note in `specs/2026-09-17-native-engine-cpu-spine/plan.md`).

Reference fixture: Ultralytics `yolo26n.pt` exported at opset 18, imgsz 640,
NMS-embedded. Surveyed 2026-10-01 (ultralytics 8.4.113, torch 2.12, onnx 1.22):

- 485 nodes, opset 18. Op histogram (excluding `Constant`, which folds):
  `Conv` 102, `Mul` 94, `Sigmoid` 88, `Concat` 28, `Add` 22, `Reshape` 15,
  `Split` 12, `Unsqueeze` 11, `Expand` 8, `Transpose` 6, `MatMul` 4,
  `MaxPool` 3, `GatherElements` 3, `Cast` 3, `Softmax` 2, `Resize` 2, `Div` 2,
  `Slice` 2, `TopK` 2, `ConstantOfShape` 2, `Equal` 2, `Where` 2, `Shape` 1,
  `Gather` 1, `Sub` 1, `ReduceMax` 1, `Flatten` 1, `Mod` 1.
- I/O: `images` `[1,3,640,640]` → `output0` `[1,300,6]` (TopK-based end-to-end
  NMS, K=300 fixed — no `NonMaxSuppression` op, no data-dependent dims).
- 204 float32 initializers; 64 `Constant` nodes (53 int64, 11 float);
  `value_info` empty. `Flatten` axis=1, `Softmax` axis=2, `TopK` axis=2,
  `Concat`/`Split`/`Gather`/`GatherElements` carry explicit axes.

## Scope

In:

- [R1] Loader: fold `Constant` nodes (float32 + int64) into initializers;
  admit `Flatten`; allowlist + opset-18 attribute policies for the 23 new ops.
- [R2] IR: intermediate tensors may be float32, int64, or bool (`Shape`,
  `ConstantOfShape`, `Gather`/`TopK` indices, `Equal` masks). Graph
  inputs/outputs stay float32-only. Arena planner handles multi-dtype buffers.
- [R3] Kernels: naive CPU reference implementations for the 23 new ops, with
  hand-computed unit tests (same bar as [V-4]/[V-4a] of the spine phase).
- [R4] Static shapes throughout: every node output shape resolves at load for
  the fixed `[1,3,640,640]` input. No dynamic-shape machinery in this packet.
- [R5] Parity: the same NMS-embedded file through NATIVE and ONNX_RUNTIME,
  elementwise max abs diff within 1e-4 (budget carried from [A-2]; observed
  figure recorded). Missing fixture fails, never skips.

Out (with pointers):

- Other YOLO variants/sizes, other input resolutions → follow-up packets.
- `NonMaxSuppression`-op exports, dynamic batch/HxW, per-shape plan cache →
  roadmap Phase N2 proper.
- Detection mAP evaluation, DALI postprocessing changes → out of engine scope;
  DALI pipelines (`export/dali/`) untouched.
- Performance work; CUDA kernels → Phase N1. CPU kernels stay unoptimized oracles.
- Shared-template `BackendHybridTestBase` instantiation → standing [V-7] follow-up.

## Constraints (traced)

- Constitution: engine/backend boundary (nothing under `engine/` includes the
  abstraction), CPU-oracle-not-target, static-shape admission at load.
- No new third-party runtime dependency for NATIVE; toolchain pins
  (torchvision/ultralytics/opset) recorded in the provisioning step.
- Orchestration discipline of this branch holds: packet committed before each
  dispatch, one ledger row per attempt, worker never commits, orchestrator
  re-scores (see `orchestration.md` in this directory).

## Open questions

- [Q1] `Resize` mode/attrs in the fixture (expected nearest; pin at implementation).
- [Q2] `Mod` fmod=1 semantics vs ORT on negatives (hand-computed test decides).
- [Q3] `TopK` largest/sorted attrs and `Slice` input-form (inputs, not attrs,
  at opset 18) — pin alongside [Q1] during Y1.
