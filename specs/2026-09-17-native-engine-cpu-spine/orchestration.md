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

## Open questions

- [Q-1] Resolved → [D-7] (hand-written reader), 2026-09-30.
- [T-3]/[A-1] Resolved → [D-8] (explicit opset-18 pin), 2026-09-30.
- [Q-2..Q-4] Open, non-blocking for N0 (owners: maintainer).
