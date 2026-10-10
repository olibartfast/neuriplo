# Orchestration — neuriplo-tasks Operator Coverage on the Neuriplo Engine

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
cmake -S . -B build-ne -DDEFAULT_BACKEND=NEURIPLO_ENGINE -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  && nice -n 10 cmake --build build-ne --parallel 2 \
  && NEURIPLO_ENGINE_FIXTURE=$HOME/.cache/neuriplo-op-inventory/fixtures/resnet18.onnx \
     ctest --test-dir build-ne -R "engine_" --output-on-failure \
  && ./scripts/quality/format.sh --check && git status --short
```

Integration gate (planner, after merging a packet): the same build in
`build-parity` (NEURIPLO_ENGINE + ONNX_RUNTIME), `NeuriploEngineInferTest`, ResNet-18 and
YOLO26 parity, then `neuriplo_engine_model_zoo_parity_test` on every manifest whose
families the packet touches, plus the YOLO and Tier 2 manifests as regression.

## Packet index

- T1 — tier 1 (YOLO) — orchestrator-implemented, see ledger
- T2 — tier 2 (transformer encoders) — orchestrator-implemented, see ledger
- P3a — elementwise math: Abs, Neg, Exp, Log, Cos, Sin, Clip
- P3b — comparison and logic: Greater, Less, And, Not
- P3c — ReduceSum, Range, Pad (with load-time evaluation)
- P3d — GridSample, Einsum
- P3e — int64 and bool graph I/O ([D-14]); written at dispatch
- R1 — rename the backend `NEURIPLO_ENGINE` → `NEURIPLO_ENGINE` (maintainer, 2026-10-10)

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

### Packet R1 — rename `NATIVE` to `NEURIPLO_ENGINE`

Maintainer decision 2026-10-10: the backend must not be called `NATIVE`.
New identity: ID `NEURIPLO_ENGINE`, directory `backends/neuriplo-engine/`,
adapter `NeuriploEngineInfer`. The engine library (`engine/`,
`neuriplo_engine`, namespace `engine`) keeps its name. Spec files are
orchestrator-owned and updated by the planner, not the worker.

- **Writable:** everything under `backends/native/` (moved with `git mv` to
  `backends/neuriplo-engine/`), `backends/src/BackendRuntimeRegistry.cpp`,
  `cmake/Native.cmake` (moved to `cmake/NeuriploEngine.cmake`),
  `cmake/BackendRegistry.cmake`, `cmake/LinkBackend.cmake`,
  `cmake/versions.cmake` (comment), `docs/backends.yaml`,
  `docs/DEPENDENCY_MANAGEMENT.md` (regenerated only, via
  `python3 scripts/gen_backend_docs.py`), `engine/README.md`,
  `engine/test/*.cpp` and `engine/test/CMakeLists.txt` (fixture variable
  only), `.github/workflows/ci.yml` (the `native` job only), `Readme.md`,
  `CHANGELOG.md` (`[Unreleased]` only). Nothing else.
- **Mapping (exact):** `NATIVE` → `NEURIPLO_ENGINE` (backend ID, `USE_NATIVE`
  → `USE_NEURIPLO_ENGINE`, `NEURIPLO_BACKEND_NATIVE_*` →
  `NEURIPLO_BACKEND_NEURIPLO_ENGINE_*`); registry display name
  "Native Engine" → "Neuriplo Engine"; `NativeInfer` → `NeuriploEngineInfer`;
  `NativeRuntimeFactory` → `NeuriploEngineRuntimeFactory`; `NativeInferTest`
  → `NeuriploEngineInferTest`; `native_parity*`, `native_yolo_parity*`,
  `native_model_zoo_parity*` → `neuriplo_engine_parity*`,
  `neuriplo_engine_yolo_parity*`, `neuriplo_engine_model_zoo_parity*`;
  `NEURIPLO_NATIVE_FIXTURE` → `NEURIPLO_ENGINE_FIXTURE`; CI job `native` →
  `neuriplo-engine`, `build-native` → `build-neuriplo-engine`.
- **Must not change:** "native" in its ordinary sense (native-endian, native
  handles, `-march=native`, `GGML_NATIVE`, C# `class Native`, Gemma's native
  tokens), `engine/` and `neuriplo_engine`, `specs/**`.
- **Acceptance (once, final action):**
  ```bash
  cmake -S . -B build-ne -DDEFAULT_BACKEND=NEURIPLO_ENGINE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
    && nice -n 10 cmake --build build-ne --parallel 2 \
    && NEURIPLO_ENGINE_FIXTURE=$HOME/.cache/neuriplo-op-inventory/fixtures/resnet18.onnx \
       ctest --test-dir build-ne -R "^engine_|NeuriploEngineInferTest" --output-on-failure \
    && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check \
    && ! git grep -nIE "\bNATIVE\b|NativeInfer|NativeRuntimeFactory|USE_NATIVE|backends/native|Native\.cmake|native_(parity|yolo|model_zoo)|NEURIPLO_NATIVE_FIXTURE" -- ':!specs' ':!CHANGELOG.md' \
    && git status --short
  ```
- **Budget:** 60 turns. **Handback:** `PACKET R1 HANDBACK pass|fail`,
  acceptance tail, deviations, `git status --short`.
- **Scoreboard correction (planner, after attempt 4):** the original filter
  `engine_` also matched the renamed `neuriplo_engine_parity*` tests, which
  need provisioned fixtures; it is now `^engine_`. CHANGELOG.md leaves the
  grep, because released sections keep the old name; the planner reviews
  `[Unreleased]` by hand.

### Packet R1-repair

Fresh worker, same worktree, continuing from the R1 diff.

- **Writable:** `backends/neuriplo-engine/**`, `CHANGELOG.md`, `Readme.md`.
- **Required final state:**
  1. `./scripts/quality/format.sh --check` passes: run `clang-format -i`
     only on the files under `backends/neuriplo-engine/` that the check
     reports.
  2. `CHANGELOG.md`: everything from `## [0.10.0]` down is byte-identical to
     the starting revision (released history keeps `NATIVE`); `[Unreleased]`
     uses `NEURIPLO_ENGINE`.
  3. `Readme.md`: the link into `specs/roadmap.md` targets the renamed
     heading `#neuriplo-engine-track`, and its link text reads "Neuriplo
     Engine Track"; link text "Native engine, Phase N0" reads "Neuriplo
     engine, Phase N0".
  4. gtest names `NativeAndOnnxRuntimeAgreeElementwise` and
     `NativeAndOnnxRuntimeConfidentRowsAgree` become
     `NeuriploEngineAndOnnxRuntimeAgreeElementwise` and
     `NeuriploEngineAndOnnxRuntimeConfidentRowsAgree`.
- **Acceptance:** the corrected R1 scoreboard above, once. The planner then
  checks that `git diff 2b1d5bf -- CHANGELOG.md` touches `[Unreleased]` only.

### Packet P3b — comparison and logic

- **Starting revision:** `9da8ec5`.
- **Writable:** new `engine/src/kernels/{Greater,Less,And,Not}.cpp`; modify
  `engine/src/kernels/Kernels.hpp`, `engine/src/kernels/CpuKernels.cpp`,
  `engine/CMakeLists.txt` (source list), `engine/src/ModelLoader.cpp`
  (allowlist + accepted-ops message only), `engine/src/ShapeInference.cpp`
  (`compute_outputs` cases and the static evaluator `eval_node` only),
  `engine/test/KernelsTest.cpp` and `engine/test/ShapeInferenceTest.cpp`
  (append tests only). Nothing else.
- **Required final state:**
  1. Greater and Less: two inputs sharing float32 or int64 dtype, NumPy
     multidirectional broadcasting, bool output. And: two bool inputs,
     broadcasting, bool output. Not: one bool input, bool output. House
     style: `Equal.cpp`.
  2. Loader admits all four with no attributes; shape rules give Bool with
     the broadcast shape (Not: input dims).
  3. The static evaluator (`eval_node`) resolves Greater/Less over int64
     operands to a bool `EvalTensor`, and And/Not over bool operands, the
     way it already handles Equal, so shape subgraphs that compare shape
     values stay load-time constants.
  4. Tests: a hand-computed broadcast case per op, an int64 case for Greater
     and Less, a negative dtype case naming the op, and one
     `TEST_F(EngineShapes, ...)` proving Greater over a Shape-derived int64
     value resolves statically (feeding Where or a Cast to int64 that feeds
     a Reshape).
  5. clang-format only on new kernel files; existing files edited by hand.
- **Acceptance:** the packet scoreboard at the top of this file, with
  `-DDEFAULT_BACKEND=NEURIPLO_ENGINE`, build dir `build-ne`, and ctest filter
  `^engine_`.
- **Budget:** 40 turns.

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
| 3 | P3a | worker (`sonnet`, worktree `wip/native-p3a`) | pass on the worker's second run (the first build failed on a typo in its own generated `Abs.cpp`; deviation: acceptance run twice) | pass: engine 12/12, NeuriploEngineInferTest, ResNet-18 and YOLO26 parity, format | Diff inside the writable list; planner review found no defects |
| 4 | R1 | worker (`sonnet`, worktree `wip/native-r1`) | fail: scoreboard defect (planner-owned filter matched fixture-dependent parity tests) | planner re-score with corrected scoreboard: tests 13/13, docs in sync, no leftover names; format check fails on two moved test files | Review also found: released 0.10.0 CHANGELOG entry renamed; README anchor and two gtest names left. Sent to R1-repair |
| 5 | R1-repair | worker (`sonnet`, same worktree) | pass | pass: engine 12/12 + NeuriploEngineInferTest, ResNet-18 and YOLO26 parity, docs, format; `act --dryrun` of the renamed CI job green (full local act still blocked by the registry DNS fault noted in Y12; GitHub CI is the real run) | Clean scope; commit `8d6dece` |
| 6 | P3b | worker (`sonnet`, worktree `wip/native-p3b`) | pass | pass: engine 12/12 + NeuriploEngineInferTest, ResNet-18 and YOLO26 parity, format | Clean scope; review checked the generated Greater/Less comparison lines and the evaluator branches. Deviation: the shape test also routes through And (extra coverage) |
