# Orchestration — YOLO26 Detection on the Native Engine

Branch discipline from the spine phase holds: a packet is committed here
before every worker dispatch; one ledger row per attempt; workers never
commit; the orchestrator re-scores and commits. Spec files stay
orchestrator-owned; workers get bounded writable paths per packet.

Packet index:

- Y1a below — IR dtypes + Constant folding + Flatten (no new-op policies yet)
- Y1b (next) — allowlist + attribute policies + shape rules for the 23 ops, [Q1]–[Q3]

### Packet Y1a — IR dtypes, Constant folding, Flatten admission

Survey correction (orchestrator): the fixture needs **23** new ops, not 22
(`Mul, Sigmoid, Concat, Split, Unsqueeze, Expand, Transpose, GatherElements,
Gather, Cast, Softmax, Resize, Div, Slice, TopK, ConstantOfShape, Equal,
Where, Shape, Sub, ReduceMax, Flatten, Mod`); spec files corrected in the same
commit as this packet. This packet covers only the dtype/folding foundation —
no allowlist or shape-rule changes for the 23 ops (those are Y1b), no kernels.

- **Writable:** `engine/include/engine/Graph.hpp`, `engine/src/ModelLoader.cpp`,
  `engine/src/ShapeInference.cpp` and `engine/src/MemoryPlanner.cpp` only
  insofar as dtype handling requires, `engine/test/LoaderTest.cpp`,
  `engine/test/MemoryPlannerTest.cpp` (extend in place; no new test files
  unless the registration pattern in `engine/test/CMakeLists.txt` demands it —
  read it first). Never: `specs/**`, `backends/**`, `cmake/**`,
  `scripts/**`, kernel files, `Device`/`Executor` sources.
- **Read-only:** `engine/include/engine/Shapes.hpp`, `Plan.hpp`, `Device.hpp`;
  `scripts/2026-09-17` spine specs [D-9]/[R-12] notes for the dtype precedent.
- **Required final state:**
  1. `DataType::Bool` added (`Graph.hpp:37-41`); intermediate tensors may be
     float32/int64/bool; graph inputs/outputs stay float32-only (reject
     otherwise with node context — verify the current enforcement, do not
     assume it).
  2. `Constant` nodes fold at load into `initializers` (float32 + int64;
     survey oracle: 64 Constants = 53 int64 + 11 float). Folded nodes vanish
     from `graph.nodes` and consumers rewire to the initializer name. A
     `Constant` of any other dtype is a load rejection with node context.
  3. `Flatten` loads (attribute `axis`; survey uses axis=1).
  4. Planner sizes buffers by dtype (bool 1, int64 8, float32 4); mixed-dtype
     graphs plan without overlap errors.
  5. Tests are hermetic (hand-built fixtures; do NOT depend on
     `/tmp/yolo26survey/yolo26n.onnx`). That survey file may be used as a
     local read-only oracle to confirm counts (485 nodes → 421 after folding;
     204 → 268 initializers) but no test may open it.
- **Budget:** 12 turns. **Handback:** `GROUP Y1a HANDBACK pass|fail`, one line
  per obligation with evidence, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel 6 && ctest --test-dir build-native -R "engine_loader|engine_plan" --output-on-failure && ./scripts/quality/format.sh --check && git diff --stat
  ```

## Run ledger

| Attempt | Group | Role | Model | Turns | Wall clock | First-pass acceptance | Interventions | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | survey | Orchestrator | muse-spark | — | — | n/a | 0 | Fixture surveyed: yolo26n NMS-embedded, opset 18, 485 nodes, 23 new ops, [1,300,6] static; packet written |
| 1 | Y1a (implement) | Implementer | opencode `general` subagent | — | — | Pass (orchestrator re-scored: 5 files in scope, `engine_loader|engine_plan` 4/4 green) | 0 | `DataType::Bool`, f32-only I/O enforced, Constant folding f32+i64 with rewiring, Flatten admission, dtype-aware planner; hermetic tests only |
