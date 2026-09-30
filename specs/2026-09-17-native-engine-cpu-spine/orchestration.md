# Orchestration — Native Engine CPU Reference Spine (Phase N0)

How Phase N0 is executed: roles, routing, packets, boundaries, and the run
ledger. Every delegation below is traceable to a packet in this file and a
ledger row. Packets are committed **before** their worker runs; the ledger is
updated when the handback lands.

## Roles and routing

Routed by capability tier and role name, per `orchestrate-ai-coding-workflows`.
Model assignments live in `~/.config/opencode/opencode.json` (global
OpenCode config; repo has no `.opencode/` override):

| Role | Tier | Model (as configured) | Owns |
| --- | --- | --- | --- |
| Specifier | Strongest | session driver | This packet, [Q-n]/[D-n] decisions, accepting or rejecting output |
| Architect | Strongest | `deepinfra/tencent/Hy3` | Ambiguous/expensive decisions (escalation only) |
| Planner | Mid | `meta/muse-spark-1.3-contributor` | Handoff packets, routing defects to fresh workers |
| Implementer | Cheap | `deepinfra/zai-org/GLM-5.3-Flash` | One packet, named paths only |
| Reviewer | Strongest, read-only | `deepseek/deepseek-flash` ("DeepSeek V4.1 Flash", DeepSeek API) | Diff review against the packet; may reject, never writes |

Note (2026-09-30): the reviewer mapping was broken (`deepseek/deepseek-v4.1-flash`,
not a real endpoint) and fixed to `deepseek/deepseek-flash`. Group 1 ran
without a reviewer gate for that reason; Groups 2+ reinstate it.

## Acceptance

Owned by the specifier, read-only to anything it scores — `validation.md`
commands. Per-group filters score implementation groups until parity lands;
the full command applies from Group 7. The acceptance command runs **once**
per attempt, verbatim, as the worker's final action; targeted checks may run
freely before it.

## Enforced vs advisory

| Rule | Enforcement |
| --- | --- |
| Reviewer writes nothing (`permission: edit: deny` in `agents/reviewer.md`) | Harness-enforced |
| Worker writable paths (per packet) | Advisory (prompt + orchestrator diff review before commit) |
| No commits/pushes by workers (orchestrator commits between packets) | Advisory + verified (`git status` before commit) |
| Acceptance run once, verbatim, no repair-and-rerun | Advisory + verified (handback must paste the tail) |
| Spec files (`specs/**`) writable by specifier/orchestrator only | Advisory + verified |
| Clean tree before dispatch; never stash/revert чужое work | Orchestrator-verified |

## Handoff packets

### Packet T-3 — fixture node dump (probe, read-only)

- **Worker:** implementer. **Writable:** nothing (read-only probe).
- **Task:** read `export_torchvision_classifier.py`, run the export into a temp
  dir if torch/torchvision/onnx exist (never write to the hard-coded
  `/workspace/`), dump node op_types + attributes + opset + IO shapes.
- **Result:** opset 18, 49 nodes, no BatchNorm → decisions [D-8], [R-5]/[V-4a]
  updates. No repo files touched.

### Packet Group 1 — skeleton and build seam (T-4..T-7, amended)

Branch synced with `origin/develop`; baseline default build green 35/35.
Q-1 resolved as [D-7] before dispatch; packet touches no parsing code.

- **Writable:** new `engine/` (`engine/CMakeLists.txt` → static lib
  `neuriplo_engine`, `engine/include/` headers, stub source); new
  `cmake/Native.cmake` (module `Native`); `cmake/BackendRegistry.cmake`
  (NATIVE entry only); `cmake/versions.cmake` (declared-null validator case
  only); `CMakeLists.txt` (engine hook only); `docs/backends.yaml` (NATIVE
  entry only) + GEN-regenerated files; T-6 guard inside the writable set;
  `backends/native/test/CMakeLists.txt` as an **empty stub** (orchestrator
  amendment, plan.md Notes — `neuriplo_add_backend_tests` runs an unguarded
  `add_subdirectory` over every enabled backend's `TEST_DIR`, so
  `-DDEFAULT_BACKEND=NATIVE` with tests on cannot configure without it).
  Never: `specs/**`, rest of `backends/native/` (Group 6), `versions.env`,
  Docker, workflows, `src/**`, `include/**`, `backends/src/**`.
- **Read (1 turn, parallel):** plan.md:24-51 + Notes tail; BackendRegistry
  :1-25,:177-197; CMakeLists.txt:29-70; versions.cmake:155-230;
  OpenCVdnn.cmake; backends.yaml:1-45,:193-218; requirements [R-1]/[D-1]/
  [D-5]/[D-7]; validation [V-1]/[V-9]/[V-10].
- **Final state (T-5 order load-bearing):** (1) versions.cmake FIRST branch
  `VERSION_VAR_NAME STREQUAL "NEURIPLO_NO_EXTERNAL_SDK"` passing validation;
  (2) `cmake/Native.cmake`; (3) registry `NATIVE` + `Native` /
  `backends/native/test` / `NEURIPLO_NO_EXTERNAL_SDK`. T-4: standalone-capable
  `engine/` (CMake ≥3.10, CXX 17, `$<BUILD_INTERFACE:>` includes), default
  build compiles no engine sources. T-6: configure-time guard over engine TUs;
  prove it fires on a temp offender, then revert; engine `*.hpp/*.cpp` must
  not contain `neuriplo`/`InferenceInterface` (validation grep). T-7: yaml
  entry (`version_var`/`setup_script`/`dir_var`/`dir_default: null`,
  `dockerfile` omitted, `test_exe: null`, ONNX, x86_64+arm64, gpu false);
  regen + `--check` clean. Accepted: null version renders a cosmetic `` `None` ``
  GEN row (generator out of scope).
- **Checks:** default `OPENCV_DNN` configure/build/ctest green with NATIVE
  registered; `-DDEFAULT_BACKEND=NATIVE` configures+builds empty engine.
- **Acceptance (once, verbatim):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```
- **Budget:** 12 turns. **Handback:** `GROUP 1 HANDACK pass|fail`, one line per
  T-n with evidence, acceptance tail, deviations, NO-GO, `git status` list.

### Packet Group 2 — ONNX loading and graph IR (T-8..T-10)

Group 1 skeleton landed — extend, do not restructure. Baseline default build
green 77/77. [D-7]/[D-8] decided before dispatch; packet touches no kernels,
planner, executor, or adapter.

- **Writable:** new `engine/include/engine/Graph.hpp` (IR types + exception),
  new `engine/include/engine/ModelLoader.hpp` (the ONE isolating interface),
  new `engine/src/ModelLoader.cpp`, new `engine/src/WireReader.hpp`
  (private, under `src/` so the T-6 glob covers it), new
  `engine/src/WireReader.cpp`, new `engine/test/CMakeLists.txt`, new
  `engine/test/LoaderTest.cpp`; edit `engine/CMakeLists.txt` (append the two
  new sources + a `BUILD_INFERENCE_ENGINE_TESTS`-guarded
  `add_subdirectory(test)` only); minimal forward-include touches to
  `engine/include/engine/Engine.hpp` / `engine/src/Engine.cpp` only (no
  renames, no namespace change, no restructure). Never: `specs/**`,
  `backends/**` (incl. `backends/native/test`, Group 6), `cmake/**`,
  top-level `CMakeLists.txt`, `docs/**`, `versions.env`, Docker, workflows,
  `src/**`, `include/**`, `backends/src/**`.
- **Read (1 turn, parallel):** plan.md:52-62; requirements.md:23-28
  ([R-2]/[R-3]), :33-37 ([R-5]), :122-132 ([D-7]/[D-8]), :189-201 ([A-1]);
  validation.md:45-50 ([V-2]/[V-3]); engine/CMakeLists.txt:1-46;
  engine/include/engine/Engine.hpp; engine/src/Engine.cpp;
  `export_torchvision_classifier.py:1-26` (fixture provenance only — never
  run, never write its hard-coded `/workspace/`); cmake/Native.cmake
  (read-only); BackendRegistry NATIVE entry (read-only); SetupTests.cmake:1-12
  (why engine tests wire via `engine/CMakeLists.txt`, not the registry).
- **Final state (T-8 reader per D-7):** hand-written protobuf wire-format
  reader (varint, fixed32/64, length-delimited; fields by tag/wire-type only)
  scoped to the `ModelProto` subset [R-2] needs — nodes, attributes,
  initializers, graph IO, dtypes, shapes. Lives in `engine/src/WireReader.*`
  behind the ONE public entry `engine::LoadGraphFromFile(const std::string&)`
  in `ModelLoader.hpp`; no protobuf/ONNX names, tags, or schema constants in
  public headers. No protobuf dependency in any build closure (no
  `find_package`, no `FetchContent`, no vendored proto, no toolchain change;
  GTest scoped to `engine/test/`). Opset-18 schemas per [D-8].
- **Final state (T-9 IR):** types in `Graph.hpp`, namespace `engine`:
  `ModelLoadException : std::runtime_error`; `enum class DataType` (float32 +
  unknown-for-rejection); `TensorInfo{dtype, dims}`; `Attribute` (typed
  scalar/list variant over the [A-1] sets); `Node{name, op_type, inputs,
  outputs, attributes}`; `Graph{nodes, tensors, initializers, inputs,
  outputs}` — document order after verifying the topological property, value
  table for every named value, initializers as a dtype-tagged `Initializer`
  (`Float32` weights + `Int64` shape constants per [D-9]), inputs vs
  initializers vs intermediates explicitly distinct. Attribute surface is
  exactly [A-1]; `Add`/`Relu`/`MatMul` take no attributes.
- **Final state (T-10 rejection):** load-time `ModelLoadException` naming node
  name + op type for all four classes (unknown op, unsupported attribute,
  unsupported initializer dtype — `FLOAT16` etc.; `FLOAT` and `INT64` constants
  are accepted per [D-9], graph IO stays FLOAT-only — dynamic dims). Never at
  inference, never by guessing. T-6 holds: no `neuriplo`,
  `InferenceInterface`, or `include/neuriplo` string anywhere under `engine/`.
- **Checks:** `engine/test/LoaderTest.cpp`, ctest `engine_loader` (positive)
  + `engine_loader_negative` (three negatives). Positive asserts on real
  opset-18 fixture bytes (49 nodes, [A-1] histogram, initializer count, IO
  `FLOAT [1,3,224,224]->[1,1000]`; fixture via env/cache var, missing FAILS
  never skips/mocks); hermetic hand-encoded wire-bytes cases keep CI green
  without torch. Negatives assert message CONTENT per [V-3]. T-6 green;
  default build compiles no engine sources.
- **Acceptance (once, verbatim):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R engine_loader --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```
- **Budget:** 12 turns. **Handback:** `GROUP 2 HANDACK pass|fail`, one line
  per T-n with evidence, acceptance tail, deviations, NO-GO, `git status`.
- **Field notes (verified live 2026-09-30 via python onnx, do NOT rely on
  memory):** ModelProto `ir_version=1, producer_name=2, doc_string=6, graph=7,
  opset_import=8`; GraphProto `node=1, name=2, initializer=5, input=11,
  output=12, value_info=13`; NodeProto `input=1, output=2, name=3, op_type=4,
  attribute=5`; AttributeProto `name=1, f=2, i=3, s=4, t=5, g=6, floats=7,
  ints=8, strings=9, type=20`; TensorProto `dims=1, data_type=2, float_data=4,
  name=8, raw_data=9, external_data=13, data_location=14`. Notably
  ModelProto.graph is 7 (not 6) and GraphProto.initializer is 5 (not 11).
- **Fixture uses EXTERNAL data** (`resnet18.onnx` + `resnet18.onnx.data`
  companion): the loader must resolve `data_location=EXTERNAL` via
  `external_data` location/offset/length relative to the model directory.
- **Split 2026-09-30 (worker-window, second time):** 2b overflowed on reads +
  a stalled mid-file write. 2c = complete `ModelLoader.cpp` single write
  only; 2d = tests + CMake wiring + acceptance.

### Packet Group 3a — static shape inference (T-11, shape half of T-27)

Implementation split of Group 3 (3a shapes, 3b memory planner, 3c device seam),
following the Group 2 precedent: size each packet for the worker window. Group 2
landed; baseline default build green 77/77, engine loader 14/14. This packet
touches no loader, parser, kernels, planner, executor, or adapter code.

- **Writable (nothing beyond these):**
  - new `engine/include/engine/Shapes.hpp` — the public shape-inference surface,
    namespace `engine`
  - new `engine/src/ShapeInference.cpp`
  - new `engine/test/ShapeInferenceTest.cpp`
  - edit `engine/CMakeLists.txt` — add `src/ShapeInference.cpp` to the
    `neuriplo_engine` source list only
  - edit `engine/test/CMakeLists.txt` — append one `add_executable`
    (`engine_shapes_test`) and two `add_test` (`engine_shapes`,
    `engine_shapes_negative`), copying the `engine_loader` wiring shape
  - Never: `specs/**`, `cmake/**`, `backends/**`, `docs/**`, `versions.env`,
    Docker, workflows, `src/**`, `include/**`, or any other `engine/` file —
    `Graph.hpp`, `ModelLoader.*`, `WireReader.*`, `Engine.*` are read-only.
- **Read-only (read, never modify):**
  - `engine/include/engine/Graph.hpp` (the IR types the result populates)
  - `engine/include/engine/ModelLoader.hpp` (exception type, loading contract)
  - `engine/test/LoaderTest.cpp` and `engine/test/CMakeLists.txt` (test style,
    hermetic wire encoders, target/test naming)
  - `specs/2026-09-17-native-engine-cpu-spine/requirements.md` — [R-3], [R-5],
    [D-9], [A-1], [A-3]; `plan.md` — T-11 and T-27; `validation.md` — [V-5],
    [V-13]
- **Fixed interface (exact identifiers):** in `engine/include/engine/Shapes.hpp`,
  namespace `engine`:
  - `using ShapeMap = std::map<std::string, std::vector<int64_t>>;`
  - `struct InferredShapes { std::map<std::string, TensorInfo> tensors; };`
  - `InferredShapes InferShapes(const Graph& graph, const ShapeMap& input_dims);`
- **Required final state (T-11):** `InferShapes` computes the concrete
  `TensorInfo` (dtype + dims) of every tensor reachable from the graph without
  mutating `graph`, and introduces no new dependency. Seed the value table with
  the graph inputs (dims from `input_dims`, dtype `Float32`; a name absent from
  `input_dims` or carrying a non-positive dim is a load error naming that
  input), then with every initializer (its stored `dtype` and `dims`). Walk
  `graph.nodes` in order and compute each node's outputs from its input types
  and attributes; do not trust the optional `value_info` table (`graph.tensors`)
  for intermediates — inference must derive them. After the walk every node
  input and every declared output must resolve; otherwise throw
  `ModelLoadException` whose message contains the offending node name and
  op type. Compute stays `Float32`; `Int64` appears only for constant shape/axes
  operands. Ops and opset-18 semantics:
  - `Add`: NumPy multidirectional broadcast of the two input shapes.
  - `Relu`: output shape equals input shape.
  - `MatMul`: ONNX MatMul, including 1-D operand promotion (prepend/append a 1,
    squeeze it from the result) and batch-dimension broadcasting.
  - `Gemm`: A and B are rank-2 (Apply transA/transB, both default 0); output is
    `[A.rows, B.cols]`. `alpha`/`beta` do not affect shape.
  - `Conv`: NCHW, rank-4 input and rank-4 weight `[M, C/group, kH, kW]`; require
    `C % group == 0` and `weight[1] == C/group`; output `[N, M, oH, oW]` with
    `o = floor((in + pad_begin + pad_end - dilation*(k-1) - 1)/stride) + 1`,
    `auto_pad` in {NOTSET, VALID, SAME_UPPER, SAME_LOWER}; a non-positive output
    dim is an error. An optional third input (bias) is rank-1 length `M`.
  - `MaxPool`: NCHW, rank-4; `kernel_shape` required, same output formula with
    `ceil_mode` (ceil instead of floor) and `auto_pad` as above.
  - `ReduceMean`: `axes` come from the optional second input (an `Int64`
    constant; the fixture's `val_226` is `[-1,-2]`); when absent or empty and
    `noop_with_empty_axes == 0`, reduce every axis; `keepdims` defaults to 1.
    Axes must be in range and unique.
  - `Reshape`: target dims come from the second input (an `Int64` constant; the
    fixture's `val_230` is `[1,512]`). A `0` copies the input dim at that index
    unless `allowzero == 1`, in which case it is a literal zero; at most one
    `-1` is inferred from the element count; the target product must equal the
    input element count. Every other value must be positive.
- **Required final state (tests, worker output not acceptance):** hermetic
  hand-encoded graphs cover `Add` broadcast, `Reshape` with `0`/`-1`, `ReduceMean`
  keepdims on/off, `Conv` stride/pad, `MaxPool` ceil, `Gemm` transposes, and the
  negative classes (reshape product mismatch, out-of-range axes, missing input
  dims, unresolved input), each negative asserting the node/op context. Loader
  tests keep their fixture-provisioning rule: a missing fixture FAILS, never
  skips. Fixture cases assert the first `Conv` output `[1,64,112,112]`, the final
  `Gemm` output `[1,1000]`, that every graph tensor resolves, and that a second
  call with input `[2,3,224,224]` yields output `[2,1000]` from the same loaded
  graph with no reload ([V-13] shape half).
- **Working method:** for every writable file write the complete final content;
  read the real file before changing it; build and run targeted checks
  (`cmake --build build-native`, the `engine_shapes*` and `engine_loader*` ctests)
  as often as useful.
- **Budget:** 14 turns. **Handback:** `GROUP 3a HANDBACK pass|fail`, one line per
  obligation with evidence, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R "engine_shapes|engine_loader" --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```
  Stop after it, pass or fail, and report. Do not repair and rerun.

### Packet Group 3a-rework — strict Reshape, honest per-shape seam (attempt 2)

Attempt 1 (ledger row 9) returned a scoreboard pass but is **rejected by the
orchestrator**: `reshape_shape` in `engine/src/ShapeInference.cpp` grew a
non-standard "carry the input batch forward" branch (currently ~lines 451-464)
so the fixture could be inferred at batch 2, and
`EngineShapes.FixtureSecondShapeWithoutReload` depends on it. ONNX `Reshape`
uses the shape operand exactly; the pinned fixture's `Reshape(mean, [1,512])` is
batch-1 in shape space, so a batch-2 inference must fail, not be rescued. This
packet is the corrected rework; harness note: the worker role is the Kilo
`general` subagent (the opencode-routed GLM worker is not available in this
session); prompts are advisory, so the review below is the enforcement.

- **Writable (nothing beyond these):** same five paths as Packet Group 3a
  (`engine/include/engine/Shapes.hpp`, `engine/src/ShapeInference.cpp`,
  `engine/test/ShapeInferenceTest.cpp`, and the two `CMakeLists.txt` files).
  Never anything else.
- **Required final state:**
  1. In `reshape_shape`: remove the batch-carry special case entirely. After
     applying `0`-copy, `allowzero`, and the single `-1` inference, require the
     output element count to equal the input element count **exactly**;
     otherwise `fail(node, "Reshape target product does not match input
     elements")`. No reference to batch, leading dimension, or exported shape.
  2. `EngineShapes.FixtureSecondShapeWithoutReload` must not exist. Replace it
     with a hermetic `Graph` (hand-encoded) whose output shape follows its
     input — a single `Relu` on `[N,3,4,4]`, or an `Add` of `[N,3,4,4]` with a
     `[3,4,4]` constant. Load it once, call `InferShapes` twice on the same
     `Graph` with `{"x",{1,3,4,4}}` and `{"x",{4,3,4,4}}`, and assert the two
     output shapes `[1,3,4,4]` and `[4,3,4,4]`. This is the [V-13] shape half:
     same loaded graph, two shapes, no reload.
  3. Keep the batch-1 fixture assertions in `EngineShapes.FixtureInferShapes`
     (`[1,64,112,112]` first `Conv`, `[1,1000]` output, every tensor resolves).
     Add a fixture case asserting that inferring the same loaded fixture with
     input `[2,3,224,224]` throws `ModelLoadException` naming the `Reshape`
     node — that is the correct behaviour for the batch-pinned target.
  4. Keep the `ReshapeProductMismatchNamesNode` negative. Everything else from
     Packet Group 3a stands unchanged.
- **Budget:** 8 turns. **Handback:** `GROUP 3a HANDBACK pass|fail` (attempt 2),
  one line per obligation with evidence, acceptance tail, deviations, NO-GO,
  `git status --short`.
- **Acceptance (once, verbatim, final action):** same command as Packet Group 3a.

### Packet Group 3b — liveness and arena memory plan (T-12, plan half of T-27)

Second split of Group 3. Group 3a landed (`InferShapes`); baseline 77/77
default, engine_shapes/loader 6/6. This packet touches no loader, parser,
inference, kernels, executor, or adapter code.

- **Writable (nothing beyond these):**
  - new `engine/include/engine/Plan.hpp` — public planner surface, namespace `engine`
  - new `engine/src/MemoryPlanner.cpp`
  - new `engine/test/MemoryPlannerTest.cpp`
  - edit `engine/CMakeLists.txt` — add `src/MemoryPlanner.cpp` to `neuriplo_engine` only
  - edit `engine/test/CMakeLists.txt` — add `engine_plan_test` (`MemoryPlannerTest.cpp`,
    `-Wall -Wextra -Wpedantic`, links `neuriplo_engine` + GTest) and two tests
    `engine_plan` (`EnginePlan.*`) and `engine_plan_negative`
    (`EnginePlanNegative.*`), and include them in the fixture env-var property so
    fixture cases get `NEURIPLO_NATIVE_FIXTURE`
  - Never: `specs/**`, `cmake/**`, `backends/**`, `docs/**`, `versions.env`,
    Docker, workflows, `src/**`, `include/**`, or any other `engine/` file — all
    of `Graph.hpp`, `Shapes.hpp`, `ModelLoader.*`, `ShapeInference.cpp`,
    `WireReader.*`, `Engine.*` and the existing tests are read-only.
- **Read-only:** `engine/include/engine/Graph.hpp`, `engine/include/engine/Shapes.hpp`,
  `engine/test/ShapeInferenceTest.cpp` (hermetic encoders and style),
  `requirements.md` [R-6], [R-12], [D-4]; `plan.md` T-12 and T-27;
  `validation.md` [V-5], [V-13].
- **Fixed interface (exact identifiers)** in `engine/include/engine/Plan.hpp`,
  namespace `engine`:
  - `struct BufferAssignment { int64_t offset = 0; int64_t size = 0; };`
  - `struct MemoryPlan { std::map<std::string, BufferAssignment> buffers; int64_t arena_size = 0; };`
  - `MemoryPlan PlanMemory(const Graph& graph, const InferredShapes& shapes);`
- **Required semantics:**
  - Arena-allocated tensors are the node outputs that are neither graph inputs
    nor initializers (graph outputs included: they must live to the end).
    Inputs, initializers, and int64 constants are external and appear in no
    buffer. Each buffer `size` = element count from `shapes` × dtype byte size
    (Float32 4, Int64 8).
  - Liveness interval per arena tensor: first definition = index of the
    producing node; last use = the greatest index of a later node consuming it,
    or `nodes.size()` when it is a graph output. A tensor that is never consumed
    and is not an output has an empty lifetime and gets no buffer.
  - Assign each buffer the lowest 64-byte-aligned offset that does not overlap
    any tensor whose lifetime interval intersects its own. Process tensors in a
    deterministic order: first definition ascending, then name ascending.
    `arena_size` = the smallest 64-byte multiple that covers every assignment.
  - Throw `ModelLoadException` if a node output has no inferred shape, if an
    inferred size is non-positive, or if the graph is empty. Do not mutate
    `graph` or `shapes`.
- **Required tests (worker output, not acceptance):** assert (a) two tensors
  with disjoint lifetimes share the same offset (reuse) and two with overlapping
  lifetimes do not overlap; (b) offsets are 64-byte aligned and within
  `[0, arena_size)`; (c) `arena_size` is strictly less than the sum of all
  arena-tensor sizes for a graph with sequential lifetimes — the [V-5] reuse
  proof; (d) an input/initializer name is absent from `buffers`; (e) a graph
  output has a buffer; (f) the [V-13] plan half: the same loaded graph + two
  different `InferredShapes` (hermetic graph, e.g. two `Conv` nodes, planned at
  two spatial sizes) produce two correct plans with no reload, and a fixture
  plan is non-empty with `arena_size > 0`. Negative: a node output missing from
  `shapes` names the tensor; an empty graph is rejected.
- **Working method:** complete final content per file; read before editing; run
  targeted checks freely. Budget 12 turns. **Handback:** `GROUP 3b HANDBACK
  pass|fail`, obligation lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R "engine_plan|engine_shapes|engine_loader" --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 3c — device-layer seam, CPU implementation (T-13)

Final split of Group 3. 3a/3b landed; baseline 77/77 default, engine
loader/shapes/plan 6/6. This packet fixes the kernel calling convention Group 4
implements and Group 5 executes, so it lands before either.

- **Writable (nothing beyond these):**
  - new `engine/include/engine/Device.hpp` — public seam, namespace `engine`
  - new `engine/src/CpuDevice.cpp`
  - new `engine/test/DeviceTest.cpp`
  - edit `engine/CMakeLists.txt` — add `src/CpuDevice.cpp` to `neuriplo_engine` only
  - edit `engine/test/CMakeLists.txt` — add `engine_device_test` (`DeviceTest.cpp`,
    `-Wall -Wextra -Wpedantic`, links `neuriplo_engine` + GTest) and one test
    `engine_device` (`EngineDevice.*`)
  - Never: `specs/**`, `cmake/**`, `backends/**`, `docs/**`, `versions.env`,
    Docker, workflows, `src/**`, `include/**`, or any other `engine/` file —
    `Graph.hpp`, `Shapes.hpp`, `Plan.hpp`, `ModelLoader.*`, `ShapeInference.cpp`,
    `MemoryPlanner.cpp`, `WireReader.*`, `Engine.*`, and the existing tests are
    read-only.
- **Read-only:** `engine/include/engine/Graph.hpp`, `engine/include/engine/Plan.hpp`,
  `requirements.md` [R-4] and [D-4]; `plan.md` T-13.
- **Fixed interface (exact identifiers)** in `engine/include/engine/Device.hpp`,
  namespace `engine`, including `<cstdint> <string> <vector>` + `"engine/Graph.hpp"`:
  - `struct TensorView { void* data = nullptr; DataType dtype = DataType::Unknown; std::vector<int64_t> dims; };`
  - `using KernelFn = void (*)(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);`
  - `class Allocator { public: virtual ~Allocator() = default; virtual void* allocate(int64_t bytes) = 0; virtual void release(void* buffer) = 0; };`
  - `class Transfer { public: virtual ~Transfer() = default; virtual void host_to_device(void* dst, const void* src, int64_t bytes) = 0; virtual void device_to_host(void* dst, const void* src, int64_t bytes) = 0; };`
  - `class KernelTable { public: virtual ~KernelTable() = default; virtual KernelFn find(const std::string& op_type) const = 0; };`
  - `class Device { public: virtual ~Device() = default; virtual const char* name() const = 0; virtual Allocator& allocator() = 0; virtual Transfer& transfer() = 0; virtual const KernelTable& kernels() const = 0; };`
  - `const Device& CpuDevice();`
- **Required CPU semantics ([R-4], one implementation):**
  - `CpuDevice()` returns a reference to one process-lifetime object; `name()`
    returns exactly `"cpu"`.
  - `Allocator::allocate` returns 64-byte-aligned storage of at least `bytes`
    (the planner's offsets assume that alignment) or `nullptr` on failure;
    `release(nullptr)` is a no-op. No zeroing is required.
  - `Transfer` copies `bytes` bytes by `memcpy` in both directions; `bytes == 0`
    is a no-op and the pointers may then be null.
  - `CpuKernel` table `find` returns `nullptr` for an unknown op type. The table
    is empty in this packet — Group 4 fills it with the reference kernels. Do not
    invent a global mutable registry; the device owns its table.
  - No dependency beyond the C++17 standard library; no backend-abstraction
    strings (`neuriplo`, `InferenceInterface`, `include/neuriplo`).
- **Required tests (`EngineDevice.*`):** `CpuDevice()` reference is stable across
  two calls and `name()` equals `"cpu"`; `allocate(128)` is non-null and
  64-byte-aligned; the block is writable/readable; `release` accepts it and
  `release(nullptr)` is safe; `host_to_device` then `device_to_host` round-trips
  a byte pattern and a zero-byte copy is a no-op; `kernels().find("NoSuchOp")`
  is `nullptr`. Group 4 adds the "known op resolves" assertion.
- **Working method:** complete final content per file; read before editing; run
  targeted checks freely. Budget 10 turns. **Handback:** `GROUP 3c HANDBACK
  pass|fail`, obligation lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R "engine_device|engine_plan|engine_shapes|engine_loader" --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 4a — kernel plumbing and elementwise/reduction kernels (T-14, part 1)

Group 4 is split by worker window: 4a plumbing + `Relu`/`Add`/`Reshape`/
`ReduceMean`; 4b `Gemm`/`MatMul`; 4c `Conv`/`MaxPool`. Group 3 landed; baseline
77/77 default, engine loader/shapes/plan/device 7/7. This packet touches no
loader, parser, inference, planner, executor, or adapter code.

- **Writable (nothing beyond these):**
  - new `engine/src/kernels/Kernels.hpp` — private kernel declarations
  - new `engine/src/kernels/CpuKernels.cpp` — the CPU operator table
  - new `engine/src/kernels/Relu.cpp`, `Add.cpp`, `Reshape.cpp`, `ReduceMean.cpp`
  - new `engine/test/KernelsTest.cpp`
  - edit `engine/src/CpuDevice.cpp` — `CpuKernelTable::find` delegates to the new table
  - edit `engine/include/engine/Graph.hpp` — append `InferenceException` only
  - edit `engine/CMakeLists.txt` — add the new `src/kernels/*.cpp` to `neuriplo_engine`
  - edit `engine/test/CMakeLists.txt` — add `engine_kernels_test` and tests
    `engine_kernels`/`engine_kernels_negative`
  - Never: `specs/**`, `cmake/**`, `backends/**`, `docs/**`, `versions.env`,
    Docker, workflows, `src/**`, `include/**`, or any other `engine/` file;
    `Device.hpp`, `Shapes.hpp`, `Plan.hpp`, `ModelLoader.*`, the existing
    `src/*.cpp` and tests are read-only.
- **Read-only:** `engine/include/engine/Device.hpp`, `Shapes.hpp`, `Plan.hpp`,
  `Graph.hpp` (except the one append), `requirements.md` [R-5], [D-4], [A-1];
  `plan.md` T-14/T-15; `validation.md` [V-4]/[V-4a].
- **Fixed interface:**
  - In `Graph.hpp`, namespace `engine`, appended beside `ModelLoadException`:
    `class InferenceException : public std::runtime_error { public: using std::runtime_error::runtime_error; };`
    with a one-line comment. Kernels raise it for inference-time failures.
  - In `engine/src/kernels/Kernels.hpp`, namespace `engine::kernels`, each
    matching `KernelFn` exactly:
    `void Relu(const Node&, const std::vector<TensorView>&, const std::vector<TensorView>&);`
    and the same for `Add`, `Reshape`, `ReduceMean` (4a), with `Gemm`, `MatMul`
    (4b) and `Conv`, `MaxPool` (4c) added by later packets. Plus
    `KernelFn FindCpuKernel(const std::string& op_type);`
- **Required semantics (naive, readable, single-threaded, no SIMD — [D-4]):**
  - Shared: read shapes from the `TensorView`s; compute element counts from
    `dims`; treat every buffer as `float32` except a shape/axes operand
    (`DataType::Int64`). On any inconsistency — wrong arity, wrong dtype, size
    mismatch, non-positive dim — throw `InferenceException` naming the op type
    and node name.
  - `Relu`: elementwise `max(0, x)` over the output element count.
  - `Add`: NumPy multidirectional broadcast of the two input `dims` into the
    output `dims`; each output element sums the broadcast-mapped inputs.
  - `Reshape`: copy input elements to output in order (identity over the flat
    buffer); read the `Int64` shape operand when present and verify its product
    equals the input element count.
  - `ReduceMean`: axes from the optional second input (`Int64`); when absent, or
    present-but-empty with `noop_with_empty_axes == 0`, reduce every axis;
    `noop_with_empty_axes == 1` with empty axes is an identity copy; `keepdims`
    defaults to 1. Validate axes in range and unique. Average over the reduced
    set with float32 accumulation.
  - `CpuKernels.cpp` builds one static lookup from the exact op-type strings to
    the functions (`"Relu"`, `"Add"`, `"Reshape"`, `"ReduceMean"`); unknown
    returns `nullptr`. `CpuDevice.cpp`'s `find` forwards to it. No global mutable
    registry; the table is a function-local static.
- **Required tests (worker output):** hand-computed small cases, values written
  in the test, not captured from another runtime — `Relu` on a mixed-sign
  vector; `Add` with broadcasting (`[2,3]` + `[3]`, and `[2,1]` + `[1,3]`);
  `Reshape` reorder and rank change; `ReduceMean` over one axis, over all axes,
  with `keepdims` 0 and 1, and with an `Int64` axes operand. A table check
  asserts `find("Relu")`/`find("Add")`/`find("Reshape")`/`find("ReduceMean")`
  are non-null and `find("NoSuchOp")` is null. Negative suite: dtype mismatch,
  wrong arity, and a size mismatch each throw `InferenceException` naming the op.
- **Working method:** complete final content per file; read before editing;
  targeted checks freely. Budget 12 turns. **Handback:** `GROUP 4a HANDBACK
  pass|fail`, obligation lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R "engine_kernels|engine_device|engine_plan|engine_shapes|engine_loader" --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 4b — linear-algebra kernels (T-14, part 2)

Extends Group 4a. Baseline 77/77 default, engine ctests 9/9.

- **Writable:** new `engine/src/kernels/Gemm.cpp`, `engine/src/kernels/MatMul.cpp`;
  edit `engine/src/kernels/Kernels.hpp` (declare the two), `engine/src/kernels/CpuKernels.cpp`
  (add `"Gemm"`/`"MatMul"`), `engine/CMakeLists.txt` (add sources),
  `engine/test/KernelsTest.cpp` (add cases). Never anything else; 4a files not
  listed here are read-only, as are `Graph.hpp`, `Device.hpp`, and the other tests.
- **Read-only:** `engine/src/kernels/Add.cpp`/`ReduceMean.cpp` (broadcast/stride
  style), `requirements.md` [R-5]/[A-1]; `plan.md` T-14/T-15; `validation.md` [V-4].
- **Required semantics (naive, float32, `InferenceException` with op+node on any
  inconsistency):**
  - `Gemm`: two or three inputs (`A`, `B`, optional `C`) and one output.
    `transA`/`transB` are int attributes defaulting to 0, `alpha`/`beta` are
    float attributes defaulting to 1.0. `A` and `B` are rank-2; `C` is rank-1
    `[N]` or rank-2 `[M,N]` and broadcasts to the output `[M,N]`. Compute
    `Y[i,j] = alpha * sum_k A'(i,k) B'(k,j) + beta * Cb(i,j)` where `A'` is `A`
    or `Aᵀ` per `transA` and likewise for `B`. The output dims are
    `outputs[0].dims`; validate `[M,N]` against the transposed operands.
  - `MatMul`: two inputs, one output. Follow ONNX MatMul: promote a 1-D lhs to
    `[1,K]` and a 1-D rhs to `[K,1]` (dropping the added dim from the result),
    broadcast the batch dimensions, and contract the last axis of `A` with the
    second-to-last of `B`. Naive index math over the output; validate inner dims.
- **Required tests:** hand-computed values in the test — `Gemm` without and with
  `transA`/`transB`, with `alpha`/`beta` != 1 and a `[N]` bias; `MatMul` 2-D,
  batched 3-D, and 1-D promotion on each side. Extend the table check so
  `find("Gemm")`/`find("MatMul")` are non-null (4c adds `Conv`/`MaxPool`).
  Negative: a Gemm inner-dim mismatch and a MatMul inner-dim mismatch throw.
- **Budget:** 12 turns. **Handback:** `GROUP 4b HANDBACK pass|fail`, obligation
  lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R "engine_kernels|engine_device|engine_plan|engine_shapes|engine_loader" --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 4c — spatial kernels (T-14, part 3 complete)

Completes the [R-5] kernel set. Baseline 77/77 default, engine ctests 9/9.

- **Writable:** new `engine/src/kernels/Conv.cpp`, `engine/src/kernels/MaxPool.cpp`;
  edit `engine/src/kernels/Kernels.hpp` (declare the two), `engine/src/kernels/CpuKernels.cpp`
  (add `"Conv"`/`"MaxPool"`), `engine/CMakeLists.txt` (add sources),
  `engine/test/KernelsTest.cpp` (add cases). Never anything else; the 4a/4b
  kernels, `Graph.hpp`, `Device.hpp`, and the other tests are read-only.
- **Read-only:** `engine/src/kernels/Gemm.cpp`, `ReduceMean.cpp` (attribute and
  stride style), `engine/include/engine/Shapes.hpp` (the reference shape
  formulas), `requirements.md` [R-5]; `plan.md` T-14/T-15; `validation.md` [V-4]/[V-4a].
- **Required semantics (naive, float32, `InferenceException` with op+node on any
  inconsistency):**
  - `Conv`: NCHW input `[N,C,H,W]`, weight `[M, C/group, kH, kW]`, optional bias
    `[M]`; `group`, `strides`, `dilations`, `pads`, `auto_pad` attributes with
    the same notation as shape inference. Output `[N, M, oH, oW]` where
    `o = floor((in + pad_begin + pad_end - dilation*(k-1) - 1)/stride) + 1`.
    `auto_pad` NOTSET/VALID/SAME_UPPER/SAME_LOWER as in shape inference.
    7-deep index loop; validate every dimension and the group split.
  - `MaxPool`: NCHW `[N,C,H,W]`; `kernel_shape` required; `strides`, `dilations`,
    `pads`, `ceil_mode`, `auto_pad`. Same window formula with `ceil_mode`
    choosing `ceil`; out-of-window positions are `-inf` for the max. Only the
    primary output is written. Validate the shapes.
- **Required tests (hand-computed values in the test):** `Conv` 1×1 with identity
  weight and with a known 2×2 kernel; `Conv` with bias; a `group == C` depthwise
  case; `Conv` with non-default stride and padding; `MaxPool` 2×2 stride 2 over
  a known 4×4; `MaxPool` with padding and `ceil_mode`. Extend the table check so
  all eight op types resolve. Negative: a `Conv` channel mismatch and a
  `MaxPool` shape mismatch throw `InferenceException` naming the op.
- **Working method:** complete final content per file; read before editing;
  targeted checks freely. Budget 14 turns. **Handback:** `GROUP 4c HANDBACK
  pass|fail`, obligation lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R "engine_kernels|engine_device|engine_plan|engine_shapes|engine_loader" --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 5 — sequential executor (T-16/T-17)

Group 4 complete (all eight kernels, 31 kernel cases green). Baseline 77/77
default, engine ctests 9/9.

- **Writable:** new `engine/include/engine/Executor.hpp`, new `engine/src/Executor.cpp`,
  new `engine/test/ExecutorTest.cpp`; edit `engine/CMakeLists.txt` (add the source),
  `engine/test/CMakeLists.txt` (add `engine_executor_test` and tests `engine_executor`/
  `engine_executor_negative`, with the fixture env property). Never anything else.
- **Read-only:** `Graph.hpp`, `Shapes.hpp`, `Plan.hpp`, `Device.hpp`,
  `ModelLoader.hpp`, the kernels and their tests, `ShapeInference.cpp`,
  `MemoryPlanner.cpp`; `requirements.md` [R-6], [R-7], [R-12]; `plan.md` T-16/T-17;
  `validation.md` [V-5], [V-7].
- **Fixed public interface (exact), `engine/include/engine/Executor.hpp`, namespace
  `engine`, includes `<map> <string> <vector>` + `"engine/Graph.hpp"` +
  `"engine/Shapes.hpp"` + `"engine/Plan.hpp"` + `"engine/Device.hpp"`:**
  - `struct InferenceResult { std::map<std::string, std::vector<float>> outputs; };`
  - `class Model` with:
    - `Model(const Graph& graph, const ShapeMap& input_dims, const Device& device);`
    - non-copyable, non-movable; destructor releases the arena through the device
    - `const Graph& graph() const;`
    - `const InferredShapes& shapes() const;`
    - `const MemoryPlan& plan() const;`
    - `InferenceResult Run(const std::map<std::string, std::vector<float>>& inputs);`
- **Required semantics:**
  - Construction runs `InferShapes` then `PlanMemory` for the given `input_dims`
    and allocates the arena **once** through `device.allocator()` (release in the
    destructor). The device is chosen once for the graph ([R-7]).
  - `Run` allocates no arena memory: it binds each graph input to the caller's
    vector (size must equal the planned element count; a missing input or wrong
    size throws `InferenceException` naming the input), binds initializers to
    their stored values, walks `graph.nodes` in order, builds `TensorView`s from
    the inferred shapes and arena offsets, looks each op up with
    `device.kernels().find(node.op_type)`, and calls it (a missing kernel is an
    `InferenceException` naming the op and node). Graph outputs are copied out of
    the arena into `InferenceResult::outputs` keyed by output name.
  - Arena offsets come from `plan.buffers`; an intermediate with no buffer is an
    error. Alias no output with an input.
  - Second and later `Run` calls reuse the arena; results are identical.
- **Required tests:** a hermetic graph (hand-encoded, e.g. an initializer fed
  through `Add` then `Relu`, or two `Conv`s) run once and checked against
  hand-computed numbers; a counting `Device` (test-side wrapper around
  `CpuDevice` with an `Allocator` that counts calls) proves the arena is
  allocated exactly once across construction plus two `Run`s, and that
  `plan.arena_size` is below the sum of intermediate sizes ([V-5]a); the fixture
  end-to-end test loads ResNet-18, constructs a `Model` for `[1,3,224,224]`, runs
  a fixed input, and asserts the output is `[1,1000]` and all finite ([T-17],
  fixture via `NEURIPLO_NATIVE_FIXTURE`, missing fixture FAILS never skips).
  Negative: missing input name, wrong input size, and an op with no kernel each
  throw `InferenceException`.
- **Working method:** complete final content per file; read before editing;
  targeted checks freely (the fixture run may take seconds — set a generous
  timeout). Budget 14 turns. **Handback:** `GROUP 5 HANDBACK pass|fail`,
  obligation lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && ctest --test-dir build-native -R "engine_executor|engine_kernels|engine_device|engine_plan|engine_shapes|engine_loader" --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 5-rework — inclusive liveness, no alias workaround (attempt 2)

Attempt 1 (ledger row 16) returned a scoreboard pass but is **rejected by the
orchestrator**. Its executor added a transient heap copy of any arena input
whose planned range overlaps an output of the same node, because
`MemoryPlanner.cpp` gives a tensor the half-open lifetime `[definition,
last_use)` — so a value consumed by node *j* is considered dead before node *j*
runs and its bytes may be handed to node *j*'s output. That is unsafe for any
kernel that reads its inputs while writing its output (Conv, ReduceMean produced
inf/NaN), and it blocks the Phase N1 device path, which cannot do host copies in
`Run`. The fix belongs in the planner; the executor workaround must go.

- **Writable:** `engine/src/MemoryPlanner.cpp`, `engine/test/MemoryPlannerTest.cpp`,
  `engine/src/Executor.cpp`, `engine/test/ExecutorTest.cpp` (and
  `engine/include/engine/Executor.hpp` only if a comment needs correcting).
  `engine/CMakeLists.txt` / `engine/test/CMakeLists.txt` are already correct —
  read-only. Never anything else.
- **Required changes:**
  1. In `PlanMemory`, lifetimes are INCLUSIVE `[definition, last_use]`: a value
     stays live through the node that consumes it. Overlap is closed-interval
     intersection (`a.begin <= b.end && b.begin <= a.end`). A graph output's
     last use is `nodes.size() - 1`. Update the file comment to state the
     invariant: **no node output may share bytes with any input of its defining
     node.**
  2. Delete the alias workaround from `Executor.cpp` entirely — the
     `aliased_inputs` vector, the overlap scan, and the `RangesOverlap` helper.
     `Run` binds each available tensor's view directly; the planner now
     guarantees safety. Fix the file comment accordingly.
  3. Rewrite the plan tests that encoded the half-open assumption:
     - `EnginePlan.DisjointLifetimesShareOffset`: use a graph whose arena
       lifetimes are genuinely disjoint under inclusive liveness (e.g. two
       independent `Relu` chains) and assert the intended pair shares an offset
       while overlapping ones do not.
     - `EnginePlan.SequentialArenaSmallerThanSum`: same kind of graph; keep the
       assertion that `arena_size` is strictly below the sum of buffer sizes.
     Keep the other plan tests passing (the fixture and per-shape tests use
     `> 0` / `!=`, so they should not need new expectations).
  4. Add a regression test `EnginePlan.OutputNeverAliasesItsInput`: for a
     hermetic node that consumes one tensor and produces another (and, if
     convenient, over the fixture plan), assert no output buffer's byte range
     overlaps any input buffer's byte range of the same node. This is the
     invariant the workaround existed to hide.
  5. Keep the executor tests; ensure the fixture end-to-end run still returns a
     finite `[1,1000]` output now that no copy happens (it must, because the
     planner no longer aliases), and keep the `TIMEOUT 1800`.
- **Budget:** 14 turns. **Handback:** `GROUP 5 HANDBACK pass|fail` (attempt 2),
  obligation lines, acceptance tail, deviations, NO-GO, `git status --short`.
- **Acceptance (once, verbatim, final action):** same command as Packet Group 5.

### Packet Group 6a — NATIVE adapter, factory, and registry wiring (T-18..T-21)

Group 5 complete; engine ctests 11/11, default 77/77. This is the only packet
that touches `backends/**`; the engine side is read-only. The adapter is where
the backend abstraction and the first-party engine meet.

- **Writable:**
  - new `backends/native/src/NativeInfer.hpp`, `backends/native/src/NativeInfer.cpp`
  - new `backends/native/src/NativeRuntimeFactory.hpp`
  - edit `backends/native/test/CMakeLists.txt` (replace the placeholder with a real target)
  - new `backends/native/test/NativeInferTest.cpp`
  - edit `cmake/Native.cmake` (append the adapter source, add `USE_NATIVE`, link the engine)
  - edit `cmake/LinkBackend.cmake` (add the `NATIVE` branch: include `backends/native/src`, link `neuriplo_engine`)
  - edit `backends/src/BackendRuntimeRegistry.cpp` (guarded include + registration entry)
  - Never: `specs/**`, `engine/**`, `docs/**`, `versions.env`, Docker, workflows,
    other `backends/**`, `include/**`, `src/**`.
- **Read-only:** `backends/src/InferenceInterface.hpp`, `InferenceMetadata.hpp`,
  `IAllocator.hpp`, `ITensorConverter.hpp`, `HostTensorConverter.hpp`,
  `IBackendRuntimeFactory.hpp`, `BackendRuntimeRegistry.hpp`,
  `backends/onnx-runtime/src/ORTRuntimeFactory.hpp`, `cmake/ONNXRuntime.cmake`,
  `cmake/BackendRegistry.cmake` (NATIVE entry), `src/InferenceBackendSetup.cpp`
  (do not edit it); engine headers `ModelLoader.hpp`, `Executor.hpp`, `Shapes.hpp`,
  `Graph.hpp`; `requirements.md` [R-8], [R-9], [R-10] context; `plan.md`
  T-18..T-21; `validation.md` [V-7], [V-8].
- **Adapter contract (fixed):**
  - `class NativeInfer : public InferenceInterface` in
    `backends/native/src/NativeInfer.hpp`. Constructor
    `NativeInfer(const std::string& model_path, bool use_gpu = false, size_t batch_size = 1,
    const std::vector<std::vector<int64_t>>& input_sizes = {})`:
    - if `use_gpu`, throw the GLOBAL `InferenceException` (from `InferenceInterface.hpp`) with a
      message naming the phase limitation (`NATIVE runs on CPU only in this phase`); no silent CPU.
    - load the model with `engine::LoadGraphFromFile(model_path)`, translate
      `engine::ModelLoadException` to the global `ModelLoadException`.
    - build the input `ShapeMap`: for each declared graph input in order, use
      `input_sizes[i]` when provided and non-empty, else that input's declared
      dims; all dims must be positive.
    - construct `engine::Model` with `engine::CpuDevice()`; set `inference_metadata_`
      from the graph's declared inputs and outputs (`addInput`/`addOutput` with
      `TensorDataType::Float32` and `batch_size`; shape from the resolved dims),
      and set `state_ = BackendState::Ready`.
  - `get_infer_results`: `validate_input`; map input `i` (positional) to the i-th
    graph input, require its byte size to equal `elements * sizeof(float)`,
    `memcpy` into a `std::vector<float>`, call `Model::Run`, and return the outputs
    in graph-output order as `std::vector<TensorElement>` floats plus their shapes.
    Translate `engine::InferenceException` to the global `InferenceExecutionException`.
  - `get_infer_results_raw`: same input path, but fill `RawOutputTensor`
    (`dtype = TensorDtype::FP32`, `bytes` = the float bytes, `shape` = the output dims)
    by copying the engine output buffers, without building `TensorElement`s.
  - `engine` failures must never escape as `engine::`-typed exceptions across the
    adapter boundary; translate both engine exception types.
  - `class NativeRuntimeFactory : public IBackendRuntimeFactory` in
    `NativeRuntimeFactory.hpp`: `create_backend` throws the global `InferenceException`
    when `use_gpu` (and otherwise returns `std::make_unique<NativeInfer>(...)`),
    `create_allocator` returns `HostAllocator`, `create_converter` returns
    `HostTensorConverter`, `name()` returns `"NativeRuntimeFactory"`.
- **Build/registry wiring:** mirror `cmake/ONNXRuntime.cmake` for source append,
  `add_compile_definitions(USE_NATIVE)`, and the engine link; add the `NATIVE`
  branch to `neuriplo_link_backend_to`; register
  `{"NATIVE", "Native Engine", &make_factory<NativeRuntimeFactory>, false}` under
  `#ifdef USE_NATIVE` in `BackendRuntimeRegistry.cpp`.
- **Required tests (`backends/native/test/NativeInferTest.cpp`, executable wired by
  the test `CMakeLists.txt` like `ONNXRuntimeInferTest`; fixture resolved via the
  `NEURIPLO_NATIVE_FIXTURE` env var or cache var with the same `/tmp/opencode/...`
  fallback, and a missing fixture FAILS never skips):** metadata reports one input
  and one output (`[1,1000]`); `get_infer_results` on `1*3*224*224` floats returns
  one output of 1000 floats; `get_infer_results_raw` returns `dtype == FP32`, shape
  `{1,1000}`, `bytes.size() == 4000`; the raw and typed paths agree elementwise;
  `NativeRuntimeFactory::create_backend(path, true, ...)` throws the global
  `InferenceException` whose message names the CPU-only limitation; a missing model
  path throws the global `ModelLoadException`.
- **Budget:** 16 turns. **Handback:** `GROUP 6a HANDBACK pass|fail`, obligation
  lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel && ctest --test-dir build-native -R "engine_|NativeInfer" --output-on-failure && cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv --parallel && ctest --test-dir build-ocv --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 6b — public GPU-request boundary (T-21, [V-8]b)

Closes Group 6. 6a landed the adapter, factory, and wiring (12/12 native ctests).
This packet only adds the public-boundary assertions to the NATIVE adapter tests.

- **Writable:** `backends/native/test/NativeInferTest.cpp`; edit
  `backends/native/test/CMakeLists.txt` only if an include dir or source is
  missing. Never anything else; `src/InferenceBackendSetup.cpp` is READ-ONLY.
- **Read-only:** `include/InferenceBackendSetup.hpp`, `backends/src/InferenceInterface.hpp`,
  `requirements.md` [R-9]; `plan.md` T-21; `validation.md` [V-8].
- **Required tests (append to `NativeInferTest.cpp`):** with the fixture and
  `NEURIPLO_NATIVE_FIXTURE` handling as in 6a —
  - `EngineOptions{model_path=fixture, backend_id="NATIVE", use_gpu=true}` passed
    to `setup_inference_engine` returns `nullptr` (the factory's throw is
    translated by the public boundary; this is [V-8]b).
  - the legacy overload `setup_inference_engine(fixture, /*use_gpu=*/true, 1, {})`
    returns `nullptr`.
  - a control case with `use_gpu=false` and `backend_id="NATIVE"` returns a
    non-null backend whose `state()` is Ready, proving the null results come from
    the GPU request and not a general failure.
  - no inference runs in the rejected cases (the factory throws before an engine
    is constructed).
- **Budget:** 8 turns. **Handback:** `GROUP 6b HANDBACK pass|fail`, obligation
  lines, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native --parallel && ctest --test-dir build-native -R "engine_|NativeInfer" --output-on-failure && cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv --parallel && ctest --test-dir build-ocv --output-on-failure && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```

### Packet Group 7a — fixture provisioning and NATIVE↔ONNX_RUNTIME parity (T-27a, T-22, [V-6])

Group 6 complete; with ONNX_RUNTIME also enabled the engine/adapter suites grow.
This packet adds deterministic fixture provisioning and the parity harness — the
payoff proof for the phase: the same file and the same input through `NATIVE`
and `ONNX_RUNTIME`, compared elementwise.

- **Writable:** new `backends/native/test/provision_parity_fixture.py`, new
  `backends/native/test/ParityTest.cpp`, edit `backends/native/test/CMakeLists.txt`.
  Never anything else; `specs/**`, `cmake/**`, `engine/**`, other `backends/**`,
  `docs/**` are read-only.
- **Read-only:** `backends/onnx-runtime/test/export_torchvision_classifier.py`
  (the export to mirror), `backends/onnx-runtime/test/CMakeLists.txt` (ORT test
  target shape and `ONNX_RUNTIME_LIBRARY` lookup), `backends/src/InferenceInterface.hpp`,
  `requirements.md` [R-10] and [A-2]; `plan.md` T-27a/T-22; `validation.md` [V-6].
- **Provisioning (`provision_parity_fixture.py`, [T-27a]):** takes an output
  directory argv; builds `torchvision.models.resnet18(pretrained=True).eval()` and
  exports it with `opset_version=18` and fixed `input_names=["input"]` /
  `output_names=["output"]` to `<dir>/resnet18.onnx`, mirroring the existing
  export so the emitted graph is the modern opset-18 set (`ReduceMean`, `Reshape`;
  no `Identity`/`Flatten`/`GlobalAveragePool`). Assert the exporter's torchvision
  version against a pinned constant (the environment is `0.27.0`; fail with a
  clear message otherwise), then `onnx.load` the result and assert `opset_import`
  version 18 and that no unsupported op appears. Print the resolved absolute model
  path (and node count) on success; exit non-zero on any failure. It must not write
  to the old hard-coded `/workspace/`.
- **Provisioning target + parity wiring** in `backends/native/test/CMakeLists.txt`,
  guarded by `if("ONNX_RUNTIME" IN_LIST NEURIPLO_ENABLED_BACKENDS)`: a custom target
  `native_parity_fixture` that runs the script with
  `-DOUTPUT_DIR=${CMAKE_BINARY_DIR}/native_parity_fixture`, a `native_parity_test`
  executable from `ParityTest.cpp`, and `add_test(NAME native_parity ...)`. The
  test target includes `backends/native/src`, `backends/onnx-runtime/src`,
  `backends/src`, `include`, `${ONNX_RUNTIME_DIR}/include`, glog; links `neuriplo`,
  `neuriplo_engine`, `${ONNX_RUNTIME_LIBRARY}`, GTest, glog (mirror the ORT test).
  Set `ENVIRONMENT` `NEURIPLO_PARITY_FIXTURE=${CMAKE_BINARY_DIR}/native_parity_fixture/resnet18.onnx`
  and a generous `TIMEOUT` (the fixture runs ~30 s per backend).
- **Parity test (`ParityTest.cpp`, [T-22]/[V-6]):** read `NEURIPLO_PARITY_FIXTURE`;
  if unset or absent, FAIL — never skip, never substitute a mock. Build `NativeInfer`
  and `ORTInfer` on the identical path with `use_gpu=false`; feed the identical
  input (`1*3*224*224` float32 bytes, a deterministic pattern written by the test);
  run `get_infer_results` on both; assert both return one output of 1000 elements;
  compare elementwise and compute the **maximum absolute difference**; assert it is
  within `1e-4` ([A-2]) and print the observed value so it can be recorded in
  `validation.md`. Fail if either backend produced no output or a differently sized
  output.
- **Budget:** 16 turns. **Handback:** `GROUP 7a HANDBACK pass|fail`, obligation
  lines, the observed max abs diff, acceptance tail, deviations, NO-GO, `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-parity --target native_parity_fixture && cmake --build build-parity --parallel && ctest --test-dir build-parity -R "parity" --output-on-failure
  ```

### Packet Group 7b — inventory, engine README, changelog (T-23, T-24, T-26 docs half)

Group 7a landed; parity green at 3.14713e-05. This packet makes the NATIVE
inventory and its documentation coherent. **Decision (orchestrator, specifier):**
NATIVE gets **no Dockerfile** — it has no external SDK and the CI backend matrix
(`.github/workflows/ci.yml`) pairs each row with a vendor Docker image, so NATIVE
is intentionally absent from that matrix and is exercised through the ordinary
`-DDEFAULT_BACKEND=NATIVE` configure. This packet records that absence explicitly
in `docs/backends.yaml`; no workflow file changes. Spec files (`specs/**`) and
`validation.md` stay orchestrator-owned.

- **Writable:** `docs/backends.yaml`, regenerated `docs/DEPENDENCY_MANAGEMENT.md`,
  new `engine/README.md`, `CHANGELOG.md`. Never anything else; `specs/**`,
  `.github/workflows/**`, `cmake/**`, `engine/*.hpp|*.cpp`, other `backends/**`
  are read-only.
- **Read-only:** `docs/backends.yaml` current NATIVE entry, `scripts/gen_backend_docs.py`,
  `README.md`, `specs/2026-09-17-native-engine-cpu-spine/requirements.md` (scope,
  [D-2]/[D-3]/[D-4], [R-11]), `specs/tech-stack.md` (engine rules), `engine/` sources.
- **Required final state:**
  1. `docs/backends.yaml` NATIVE entry: add `dockerfile: null` and change
     `test_exe: null` to `test_exe: NativeInferTest`; add a one-line comment above
     the entry stating the intentional no-Docker decision (no external SDK; not in
     the CI vendor-image matrix). Do not change the other fields.
  2. Run `python3 scripts/gen_backend_docs.py` and confirm `--check` is clean
     afterward.
  3. `engine/README.md`: a concise entrypoint stating (a) the engine is an
     interpreter — parse, infer shapes, plan memory, execute node by node, not a
     plan-building compiler; (b) device is chosen once per graph and a graph that
     cannot run entirely on the device is rejected at load; (c) CPU reference
     kernels are the correctness oracle for the future CUDA path and are
     **deliberately unoptimized — not a performance target**; (d) the boundary
     (nothing under `engine/` includes the backend abstraction); (e) a short
     directory map and how to build/test with `-DDEFAULT_BACKEND=NATIVE`
     (including `build-parity` for the ONNX_RUNTIME parity check). No spec legend
     IDs in the prose.
  4. `CHANGELOG.md` under `[Unreleased]` / `Added`: one entry for the first-party
     `NATIVE` backend — ONNX loader and graph IR, static shape inference, arena
     memory planner, CPU reference kernels, sequential executor, and the
     `InferenceInterface` adapter; validated elementwise against `ONNX_RUNTIME`
     (max abs diff 3.15e-05, budget 1e-4); a GPU request is rejected in this phase.
     Match the file's existing tone and wrapping.
- **Budget:** 10 turns. **Handback:** `GROUP 7b HANDBACK pass|fail`, obligation
  lines, `gen_backend_docs.py --check` result, acceptance tail, deviations, NO-GO,
  `git status`.
- **Acceptance (once, verbatim, final action):**
  ```bash
  python3 scripts/gen_backend_docs.py && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check && git diff --stat
  ```

## Run ledger

One row per attempt. Metrics the harness did not report are marked `—`
(Phase 7 precedent). Tokens are not re-estimated.

| Attempt | Group | Role | Model | Turns | Wall clock | First-pass acceptance | Interventions | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 (packet) | Planner | muse-spark-1.3 | — | — | n/a (writes packet) | 0 | Packet delivered, committed here |
| 2 | 1 (implement) | Implementer | GLM-5.3-Flash | 12 (budget exhausted in investigation) | — | n/a (blocked pre-implementation) | 1 (orchestrator ruled on SetupTests fork, amended packet + plan.md) | Blocked, no files changed |
| 3 | 1 (implement) | Implementer | GLM-5.3-Flash | — | — | Pass (self-reported slip: acceptance-before-regen, re-ran verbatim) | 0 code; orchestrator re-scored (77/77 + NATIVE build + docs/format) | Pass → commit `f9f48eb`, 9 files, +134/−1 |
| 4 | 0 (T-3 probe) | Implementer | GLM-5.3-Flash | — | — | Pass with finding (opset 18 vs pinned 12) | 1 (maintainer decision [D-8]) | Evidence recorded in [A-1]; [R-5]/[V-4a]/plan updated |
| 5 | 2 (implement) | Implementer | GLM-5.3-Flash | budget exhausted in reads | — | n/a (blocked pre-implementation) | 0 | Fail — no files changed; packet too large for worker window; split into 2a/2b below |
| 6 | 2a (implement) | Implementer | GLM-5.3-Flash | budget exhausted after writing 4 files | — | Self-FAIL (runtime check open); orchestrator closed it: standalone functional test (varint/fixed32/64/tags/LD/skip/truncation/group) ALL PASS under -Wall -Wextra -Werror | 1 (orchestrator functional verification) | Pass → committed as part 1 (no CMake wiring yet; zero build impact) |
| 7 | 2b (implement) | Implementer | GLM-5.3-Flash | budget exhausted mid-write | — | Fail — partial ModelLoader.cpp (stub decode_node, no LoadGraphFromFile), no tests/wiring/acceptance | 0 | Blocked; findings preserved (field numbers, external-data) in packet notes; split into 2c/2d |
| 8 | 2c/2d (complete) | Session driver | deepseek-flash | — | — | Pass: `ctest -R engine_loader` 14/14, default `OPENCV_DNN` 77/77, format clean | 1 (maintainer decision [D-9] on int64 shape constants) | Group 2b completed: dtype-tagged initializers, int64 support, test registration + encoder fix |
| 9 | 3a (implement, attempt 1) | Implementer | Kilo `general` subagent | — | — | Scoreboard pass (4/4 engine ctests, 77/77 default, format clean) but **orchestrator-rejected** | 1 (planner wrote corrected rework packet) | Rejected: non-standard Reshape "batch carry" (ShapeInference.cpp ~451-464) plus a fixture batch-2 test that depends on it; rework dispatched as attempt 2 |
| 10 | 3a (rework, attempt 2) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 4/4 engine ctests, 77/77 default, docs/format clean) | 0 | Strict Reshape, hermetic per-shape seam ([V-13] shape half), fixture batch-2 rejection; committed with Group 3a |
| 11 | 3b (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 6/6 engine ctests, 77/77 default, docs/format clean) | 0 | Liveness + 64-byte-aligned arena plan; reuse, alignment, sequential-shrink and per-shape-seam tests; committed with Group 3b |
| 12 | 3c (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 7/7 engine ctests, 77/77 default, docs/format clean) | 0 | Device seam (allocator/transfer/kernel table) + CPU implementation, 64-byte aligned; fixes the kernel calling convention for Group 4; committed with Group 3c |
| 13 | 4a (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 9/9 engine ctests, 77/77 default, docs/format clean) | 0 | Kernel plumbing + `InferenceException` + `Relu`/`Add`/`Reshape`/`ReduceMean` with hand-computed tests; committed with Group 4a |
| 14 | 4b (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 9/9 engine ctests, 77/77 default, docs/format clean) | 0 | `Gemm` (transposes, alpha/beta, bias) and `MatMul` (batch broadcast, 1-D promotion) with hand-computed tests; committed with Group 4b |
| 15 | 4c (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 9/9 engine ctests incl. 31 kernel cases, 77/77 default, docs/format clean) | 0 | `Conv` (group/stride/pad/dilation/auto_pad/bias) and `MaxPool` (ceil/pad) complete the [R-5] kernel set; committed with Group 4c |
| 16 | 5 (implement, attempt 1) | Implementer | Kilo `general` subagent | — | — | Scoreboard pass but **orchestrator-rejected** | 1 (planner wrote corrected rework packet) | Rejected: executor masked a planner defect with a transient heap copy of same-node aliased inputs; the planner's half-open lifetimes `[def, last_use)` are unsafe for read-then-write kernels. Rework fixes inclusive liveness and removes the copy |
| 17 | 5 (rework, attempt 2) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 11/11 engine ctests, 77/77 default, docs/format clean) | 0 | Inclusive liveness `[def, last_use]` in the planner (no output aliases its defining node's inputs), alias workaround removed, executor end-to-end on the fixture; committed with Group 5 |
| 18 | 6a (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 12/12 native ctests incl. 6 NativeInfer cases, 77/77 default, docs/format clean) | 1 (worker noted PIC + engine-link placement deviations, both accepted) | `NativeInfer` + `NativeRuntimeFactory`, CMake/registry wiring, adapter tests (metadata, typed + raw paths, GPU-rejection throw); committed with Group 6a |
| 19 | 6b (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: 12/12 native ctests, 77/77 default, docs/format clean) | 0 | Public `setup_inference_engine` GPU-request boundary: both overloads return `nullptr`, CPU control returns a Ready backend; committed with Group 6b |
| 20 | 7a (implement) | Implementer | Kilo `general` subagent | — | — | Pass (orchestrator re-scored: `native_parity` 27.4 s, observed max abs diff 3.14713e-05) | 0 | Deterministic opset-18 fixture provisioning + NATIVE↔ONNX_RUNTIME parity; committed with Group 7a |

## Open questions

- [Q-1] Resolved → [D-7] (hand-written reader), 2026-09-30.
- [T-3]/[A-1] Resolved → [D-8] (explicit opset-18 pin), 2026-09-30.
- [Q-2..Q-4] Open, non-blocking for N0 (owners: maintainer).
