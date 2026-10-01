# Plan — YOLO26 Detection on the Native Engine

Thin phases; each ends runnable with its acceptance. Branch stays
`feature/native-engine-cpu-spine`; merge hold lifts only when [YV-5] passes.

## Phase Y1 — IR + loader for the detection surface

- Constant folding: `Constant` nodes (float32 + int64 values) become
  initializers at load; the 64 survey nodes are the oracle.
- Dtypes: `Bool` joins the IR; intermediate tensors may be float32/int64/bool;
  graph I/O stays float32-only. Planner sizes buffers by dtype.
- Allowlist + attribute policies for the 23 new ops (opset 18); admit
  `Flatten`. Resolve [Q1]–[Q3] and record the answers as decisions here.
- Checks: loader tests assert the survey counts (485 nodes incl. 64 folded
  Constants, 204+64 initializers by dtype); negative tests for the new
  rejections.
- Acceptance: `ctest --test-dir build-native -R "engine_loader|engine_shapes|engine_plan"`.

## Phase Y2 — Elementwise / unary / reduction kernels

`Mul`, `Div`, `Sub`, `Sigmoid`, `Cast`, `Softmax`, `ReduceMax`, `Mod`,
`Equal`, `Where`. Hand-computed unit tests incl. non-default configs
(`Softmax` axis=2, `Mod` fmod sign behavior vs ORT, `Cast` to/from
float32/int64/bool).

- Acceptance: `ctest --test-dir build-native -R engine_kernels`.

## Phase Y3 — Data-movement / shape / selection kernels

`Concat`, `Split`, `Unsqueeze`, `Expand`, `Transpose`, `Slice`, `Gather`,
`GatherElements`, `Resize` (nearest per [Q1]), `Flatten`, `Shape`,
`ConstantOfShape`, `TopK` (K constant → static output shapes).

- Acceptance: `ctest --test-dir build-native -R engine_kernels`, plus an
  end-to-end engine run of the fixture to a finite `[1,300,6]`.

## Phase Y4 — Parity harness + validation + hold lift

- Provisioning: deterministic `yolo26n` NMS-embedded export (pinned
  ultralytics/torch/opset) into the build tree; CMake target
  `native_yolo_parity_fixture`. Same discipline as [T-27a].
- Test `native_yolo_parity_test`: identical file + input through NATIVE and
  ONNX_RUNTIME, elementwise compare, missing fixture fails.
- Execute all checks in `validation.md`, record evidence; CHANGELOG entry;
  lift the merge hold → PR into `develop`.

## Notes

- Record deviations here with reasons. Phase order is load-bearing (Y1 before
  Y2/Y3; Y4 last). If the survey fixture differs from the provisioned one
  (ultralytics version drift), re-survey and note it — do not silently widen
  the op set.
