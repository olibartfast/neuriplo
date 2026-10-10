# Orchestration — neuriplo-tasks Operator Coverage on the Native Engine

Workflow: `orchestrate-ai-coding-workflows` (maintainer rule, 2026-10-10: all
development goes through it). Branch discipline from the earlier native
packets holds: a packet is committed here before its worker is dispatched; one
ledger row per attempt; workers never commit; the orchestrator re-scores,
integrates, and commits. Spec files stay orchestrator-owned.

## Roles and routing

| Role | Who | Tier | Owns |
| --- | --- | --- | --- |
| Planner / orchestrator | main Claude Code session | strongest | packets, acceptance runs, integration, model-zoo parity, commits, this file |
| Implementer | Claude Code subagent, one per packet, fresh context | mid (`sonnet`) | the packet's writable paths only |
| Reviewer (optional, per packet) | read-only subagent (`Plan` type: no Edit/Write tools) | strongest | a review report, never a diff |

Defects go back to a fresh implementer in a corrected packet (`<id>-repair`);
the planner does not patch worker output. Model-zoo parity (minutes to tens of
minutes per export) is planner-run after integration, never a worker
acceptance.

## Boundaries: enforced vs advisory

- **Enforced:** each implementer runs in its own git worktree
  (`../neuriplo-w-<packet>`, branch `wip/native-<packet>`, created by the
  planner from the packet's starting revision); the reviewer has no
  write/edit tools; the planner is the only role that commits or pushes.
- **Advisory (review obligation):** writable-path lists and the
  never-commit rule. Claude Code subagents can write anywhere in their tree,
  so the planner checks `git status` / `git diff --stat` against the packet's
  writable list before integrating, and rejects out-of-scope edits.
- **Resource limits (advisory, maintainer RAM constraint):** builds use
  `nice -n 10 ... --parallel 2`; one implementer at a time.

## Acceptance (planner-owned; workers run it once, as their final action)

```bash
cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  && nice -n 10 cmake --build build-native --parallel 2 \
  && NEURIPLO_NATIVE_FIXTURE=$HOME/.cache/neuriplo-op-inventory/fixtures/resnet18.onnx \
     ctest --test-dir build-native -R "engine_" --output-on-failure \
  && ./scripts/quality/format.sh --check && git status --short
```

Integration gate (planner, after merging a packet): the same build in
`build-parity` (NATIVE + ONNX_RUNTIME), `NativeInferTest`, ResNet-18 and
YOLO26 parity, then `native_model_zoo_parity_test` on every manifest whose
families the packet touches, plus the YOLO and Tier 2 manifests as regression.

## Packet index

- T1 — tier 1 (YOLO) — orchestrator-implemented, see ledger
- T2 — tier 2 (transformer encoders) — orchestrator-implemented, see ledger
- P3a — elementwise math: Abs, Neg, Exp, Log, Cos, Sin, Clip
- P3b — comparison and logic: Greater, Less, And, Not
- P3c — ReduceSum, Range, Pad (with load-time evaluation)
- P3d — GridSample, Einsum
- P3e — int64 graph inputs, blocked on [Q-2]

### Packet P3a — elementwise math

- **Starting revision:** branch head at dispatch.
- **Writable:** new `engine/src/kernels/{Abs,Neg,Exp,Log,Cos,Sin,Clip}.cpp`;
  modify `engine/src/kernels/Kernels.hpp`, `engine/src/kernels/CpuKernels.cpp`,
  `engine/CMakeLists.txt` (kernel source list), `engine/src/ModelLoader.cpp`
  (allowlist entries and the accepted-ops message only),
  `engine/src/ShapeInference.cpp` (`compute_outputs` cases only),
  `engine/test/KernelsTest.cpp` (append tests; update the header op list).
  Nothing else.
- **Read-only:** every other file; in particular `Floor.cpp` / `Pow.cpp` as
  the house style for unary / broadcasting kernels, and the acceptance suite.
- **Required final state:**
  1. Abs and Neg accept float32 or int64 (output dtype = input dtype); Exp,
     Log, Cos, Sin accept float32. Elementwise, same dims.
  2. Clip: float32 input; optional `min` and `max` inputs, each a scalar
     float32 tensor or an omitted input (empty name); omitted means
     unbounded. Output = min(max(x, min), max). No attributes (opset 18).
  3. Each op admitted by the loader with no attributes, in the kernel table,
     and shaped by `compute_outputs` (dims of input 0, dtype as above).
  4. `KernelsTest.cpp`: per op a hand-computed case; Clip with both bounds,
     only min, only max; Abs/Neg int64; one negative case (dtype mismatch)
     whose message names the op.
  5. Comments match the surrounding density; clang-format clean for the
     kernel files and `KernelsTest.cpp` (do not run clang-format on
     `ModelLoader.cpp`, `ShapeInference.cpp`, or `ShapeInferenceTest.cpp`;
     they use a different style and are outside the format check).
- **Permitted commands:** the acceptance build and ctest, targeted test
  binaries, `clang-format` on writable kernel/test files, `git diff`,
  `git status`. No commits, no pushes, no network.
- **Budget:** 40 turns. **Handback:** `PACKET P3a HANDBACK pass|fail`, the
  acceptance output tail, deviations, `git status --short`.

### Packet P3b — comparison and logic

Written at dispatch (after P3a integrates), against the head it starts from.

### Packet P3c — ReduceSum, Range, Pad

Written at dispatch. Must extend the load-time evaluator, because the DETR
exports compute Range limits and Pad amounts from shape subgraphs.

### Packet P3d — GridSample, Einsum

Written at dispatch. Survey: GridSample 4-D, bilinear, `padding_mode` zeros,
`align_corners` 0 (every site); Einsum `bchw,bnc->bnhw` (RF-DETR-Seg).

## Run ledger

| Attempt | Packet | Implementer | Acceptance (first pass) | Integration gate | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | T1 | orchestrator (not delegated) | pass | pass: 13 YOLO exports | Implemented before the orchestration rule; commit `fbefc95` |
| 2 | T2 | orchestrator (not delegated) | pass after two orchestrator fixes | pass: 6 transformer exports + YOLO regression | Implemented before the rule; the YOLO regression rerun caught two defects (empty initializers, empty Resize scales), fixed by the orchestrator |
| 3 | P3a | worker (`sonnet`, worktree `wip/native-p3a`) | pass on the worker's second run (the first build failed on a typo in its own generated `Abs.cpp`; deviation: acceptance run twice) | pass: engine 12/12, NativeInferTest, ResNet-18 and YOLO26 parity, format | Diff inside the writable list; planner review found no defects |
