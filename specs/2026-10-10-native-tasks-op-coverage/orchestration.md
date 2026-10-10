# Orchestration — neuriplo-tasks Operator Coverage on the NERT

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
cmake -S . -B build-ne -DDEFAULT_BACKEND=NERT -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  && nice -n 10 cmake --build build-ne --parallel 2 \
  && NERT_FIXTURE=$HOME/.cache/neuriplo-op-inventory/fixtures/resnet18.onnx \
     ctest --test-dir build-ne -R "engine_" --output-on-failure \
  && PATH=$HOME/.cache/neuriplo-op-inventory/cf18-shim:$PATH ./scripts/quality/format.sh --check \
  && git status --short
```

The format check runs clang-format 18.1.3, CI's exact version (`cf18-shim`
wraps the `clang-format==18.1.3` wheel); the machine's clang-format 21, and
even 18.1.8, accept lines CI rejects. Corrected after attempt 7.

Integration gate (planner, after merging a packet): the same build in
`build-parity` (NERT + ONNX_RUNTIME), `NertInferTest`, ResNet-18 and
YOLO26 parity, then `nert_model_zoo_parity_test` on every manifest whose
families the packet touches, plus the YOLO and Tier 2 manifests as regression.

## Packet index

- T1 — tier 1 (YOLO) — orchestrator-implemented, see ledger
- T2 — tier 2 (transformer encoders) — orchestrator-implemented, see ledger
- P3a — elementwise math: Abs, Neg, Exp, Log, Cos, Sin, Clip
- P3b — comparison and logic: Greater, Less, And, Not
- P3c — ReduceSum, Range, Pad (with load-time evaluation)
- P3d — GridSample, Einsum
- P3e — int64 and bool graph I/O ([D-14]); written at dispatch
- R1 — rename the backend `NATIVE` → `NEURIPLO_ENGINE` (maintainer, 2026-10-10)
- R2 — rename the backend `NEURIPLO_ENGINE` → `NERT`, Neuriplo Engine Runtime (maintainer, 2026-10-10)
- D1 — main README: NERT row and status (maintainer, 2026-10-10); after R2

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

### Packet P3e — int64 and bool graph I/O ([D-14])

Routed to the strongest worker tier (`opus`): it changes an engine API and
the adapter's marshalling contract.

- **Starting revision:** the branch head after P3d integrates.
- **Writable:** `engine/include/engine/Executor.hpp`, `engine/src/Executor.cpp`,
  `engine/src/ModelLoader.cpp` (graph input/output element types only),
  `engine/src/ShapeInference.cpp` (`InferShapes` input dtypes only),
  `backends/nert/src/NertInfer.{hpp,cpp}`,
  `engine/test/{ExecutorTest,LoaderTest}.cpp`,
  `backends/nert/test/NertInferTest.cpp`,
  `backends/nert/test/ModelZooParityTest.cpp` (input generation
  and comparison for non-float inputs/outputs). Nothing else.
- **Required final state:**
  1. The loader accepts FLOAT, INT64 and BOOL graph inputs and outputs;
     other element types stay rejected with the input's name.
  2. `InferShapes` types each graph input by its declared dtype (it forces
     Float32 today).
  3. `engine::Model` gains a typed entry point: a host tensor type carrying
     dtype, dims and bytes, `Run` over a name → host tensor map returning
     every graph output in its own dtype. The existing float-only `Run` and
     `InferenceResult` keep compiling and behaving as before (callers and
     tests depend on them); it may delegate to the typed path.
  4. `NertInfer` marshals each input buffer by the graph input's
     dtype (float32 4 bytes, int64 8, bool 1 per element), reports input and
     output dtypes in its metadata (`TensorDataType::Int64` / `Bool`), and
     returns int64 outputs as `int64_t` and bool outputs as `uint8_t`
     `TensorElement`s (raw path: `TensorDtype::INT64` / `UINT8`), matching
     `HostTensorConverter`'s convention.
  5. `ModelZooParityTest` fills int64 inputs with a value per input given in
     the manifest (new optional token `name=value` style, or a documented
     default), and compares int64/bool outputs exactly.
  6. Tests: loader positives for INT64/BOOL graph I/O replacing the three
     negative tests that asserted rejection (keep one rejection for an
     unsupported type such as INT32); an executor test running a graph with
     an int64 input and a bool output; an adapter test with an int64 input.
- **Acceptance:** the R1 scoreboard (NERT + ONNX_RUNTIME build,
  `ctest -R "^engine_|NertInferTest"`, docs and format checks).
- **Budget:** 60 turns.

### Packet F1 — formatting under CI's clang-format

- **Writable:** `backends/nert/test/YoloParityTest.cpp` only.
- **Required final state:** the format check above (clang-format 18.1.3)
  passes; the diff is whitespace-only.
- **Acceptance:** `PATH=$HOME/.cache/neuriplo-op-inventory/cf18-shim:$PATH ./scripts/quality/format.sh --check && git diff --stat`.
- **Budget:** 10 turns.

### Packet D1 — main README

Maintainer request 2026-10-10 ("update also the main readme"). Runs after
R2 integrates, against the README that PR #45 introduced.

- **Writable:** `Readme.md` only.
- **Read-only:** `engine/README.md`, `specs/roadmap.md`,
  `specs/2026-10-10-native-tasks-op-coverage/validation.md` (the evidence the
  status note summarizes), `docs/backends.yaml`.
- **Required final state:**
  1. The intro says, in one sentence, that Neuriplo also ships its own
     runtime, NERT (Neuriplo Engine Runtime), with no vendor SDK underneath.
  2. "Choosing a Backend" table: a `NERT` row (runtime "NERT (Neuriplo
     Engine Runtime, first-party)", model format "`.onnx` (opset 18)", best
     for "Self-contained CPU inference with no vendor SDK; in development").
  3. Directly under the table, a short NERT status note (at most 6 lines):
     CPU only; opset-18 models (others convert with `onnx.version_converter`);
     validated against `ONNX_RUNTIME` on the YOLO families (v5-v12, YOLO11,
     YOLO26 detection/segmentation/pose) and ViT, ViTPose, Depth Anything V2,
     VideoMAE, ViViT, TimeSformer; DETR-family, RAFT and open-vocabulary
     coverage in progress; links to `engine/README.md` and the roadmap's
     `#nert-track` heading.
  4. Documentation table: an `engine/README.md` row ("NERT engine: layout,
     build and parity tests").
  5. Nothing else in the README changes; the rf-detr layout and tone stay.
- **Acceptance (once, final action):**
  `grep -c "NERT" Readme.md && grep -q "specs/roadmap.md#nert-track" Readme.md && grep -q "## NERT Track" specs/roadmap.md && git diff --stat`
- **Budget:** 15 turns. Tier: `haiku`.

### Packet R2 — rename `NEURIPLO_ENGINE` to `NERT`

Maintainer decision 2026-10-10: `NERT`, Neuriplo Engine Runtime (`NRT` was
the alternative; rejected because it is AWS Neuron's runtime, `libnrt`). New
identity: ID `NERT`, display name "NERT (Neuriplo Engine Runtime)",
directory `backends/nert/`, adapter `NertInfer`. The engine library keeps
its names: directory `engine/`, CMake target `neuriplo_engine`, namespace
`engine`. Spec files are planner-owned and already renamed.

- **Writable:** everything under `backends/neuriplo-engine/` (moved with
  `git mv` to `backends/nert/`), `backends/src/BackendRuntimeRegistry.cpp`,
  `cmake/NeuriploEngine.cmake` (moved to `cmake/Nert.cmake`),
  `cmake/BackendRegistry.cmake`, `cmake/LinkBackend.cmake`,
  `cmake/versions.cmake` (comment), `docs/backends.yaml`,
  `docs/DEPENDENCY_MANAGEMENT.md` (regenerated only), `engine/README.md`,
  `engine/test/*.cpp` and `engine/test/CMakeLists.txt` (fixture variable
  only), `.github/workflows/ci.yml` (the `neuriplo-engine` job only),
  `CHANGELOG.md` (`[Unreleased]` only). Nothing else.
- **Mapping (exact):** `NEURIPLO_ENGINE` → `NERT` (backend ID,
  `USE_NEURIPLO_ENGINE` → `USE_NERT`, `NEURIPLO_BACKEND_NEURIPLO_ENGINE_*` →
  `NEURIPLO_BACKEND_NERT_*`, module `NeuriploEngine` → `Nert`); display name
  "Neuriplo Engine" → "NERT (Neuriplo Engine Runtime)";
  `NeuriploEngineInfer` → `NertInfer`; `NeuriploEngineRuntimeFactory` →
  `NertRuntimeFactory`; `NeuriploEngineInferTest` → `NertInferTest`; gtest
  names `NeuriploEngineAndOnnxRuntime*` → `NertAndOnnxRuntime*`;
  `neuriplo_engine_parity*`, `neuriplo_engine_yolo_parity*`,
  `neuriplo_engine_model_zoo_parity*` → `nert_parity*`, `nert_yolo_parity*`,
  `nert_model_zoo_parity*`; `NEURIPLO_ENGINE_FIXTURE` → `NERT_FIXTURE`; CI job
  `neuriplo-engine` → `nert`, `build-neuriplo-engine` → `build-nert`,
  `/tmp/neuriplo-engine-fixture` → `/tmp/nert-fixture`, cache scope
  `docker-NEURIPLO_ENGINE` → `docker-NERT`; prose naming this backend →
  "NERT".
- **Must not change:** the CMake library target `neuriplo_engine` (every
  `target_link_libraries(... neuriplo_engine)` and
  `set_target_properties(neuriplo_engine ...)` stays), `engine/`, namespace
  `engine`, `specs/**`, released CHANGELOG sections, and ordinary uses of
  "native".
- **Acceptance (once, final action):**
  ```bash
  cmake -S . -B build-nert -DDEFAULT_BACKEND=NERT -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
    && nice -n 10 cmake --build build-nert --parallel 2 \
    && NERT_FIXTURE=$HOME/.cache/neuriplo-op-inventory/fixtures/resnet18.onnx \
       ctest --test-dir build-nert -R "^engine_|NertInferTest" --output-on-failure \
    && python3 scripts/gen_backend_docs.py --check \
    && PATH=$HOME/.cache/neuriplo-op-inventory/cf18-shim:$PATH ./scripts/quality/format.sh --check \
    && ! git grep -nIE "NEURIPLO_ENGINE|NeuriploEngine|neuriplo-engine|neuriplo_engine_(parity|yolo|model_zoo)" -- ':!specs' ':!CHANGELOG.md' \
    && git status --short
  ```
- **Budget:** 60 turns.

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
  `-DDEFAULT_BACKEND=NERT`, build dir `build-ne`, and ctest filter
  `^engine_`.
- **Budget:** 40 turns.

### Packet P3c — ReduceSum, Range, Pad

- **Starting revision:** `8dc80eb`.
- **Writable:** new `engine/src/kernels/{ReduceSum,Range,Pad}.cpp`; modify
  `Kernels.hpp`, `CpuKernels.cpp`, `engine/CMakeLists.txt`,
  `engine/src/ModelLoader.cpp` (allowlist, attribute maps, attribute
  policy, accepted-ops message), `engine/src/ShapeInference.cpp` (shape
  rules, `compute_outputs`, the static evaluator), `engine/test/KernelsTest.cpp`
  and `engine/test/ShapeInferenceTest.cpp` (append only).
- **Survey facts the packet is written against:** DEIMv2 and RF-DETR compute
  a Range `limit` as `Cast(to FLOAT)` of an int64 shape expression
  (`Shape(images)[2] / 16`), with float32 constant start and delta; other
  Range sites take int64 operands. RT-DETR v4 and D-FINE compute Pad `pads`
  as int64 from `Cast`/`Reshape` chains; `mode` is `constant`, the
  constant-value input omitted. ReduceSum takes axes as an input
  (`keepdims` 0 or 1) on float32 data, and on int64 data inside a shape
  subgraph.
- **Required final state:**
  1. ReduceSum (opset 18): data float32 or int64; optional int64 axes input
     resolved at load; `keepdims` (default 1); `noop_with_empty_axes`
     (default 0: no axes reduces everything; 1: identity).
  2. Range: scalar start, limit, delta of one dtype (float32 or int64);
     output 1-D of length max(ceil((limit - start) / delta), 0); values
     start + i*delta. Shape inference resolves the three operands at load.
  3. Pad (opset 18): data float32 or int64; int64 `pads` input (2 x number
     of padded axes), optional scalar `constant_value` (default 0), optional
     int64 `axes` input; attribute `mode` in constant, reflect, edge (`wrap`
     rejected at load). Negative pads crop. Pads resolved at load.
  4. Evaluator: a float lane in `EvalTensor` (values as double) fed by
     float32 initializers and `Cast` to FLOAT of an int64 value, so Range's
     float operands resolve; Range and ReduceSum evaluate over int64; Cast
     to INT64 of a float lane value truncates toward zero as ONNX does.
  5. Tests: hand-computed kernel cases (Pad in each mode plus a negative
     pad, Range float and int64 including an empty result, ReduceSum with
     and without keepdims and with noop_with_empty_axes); `EngineShapes`
     tests rebuilding the two surveyed chains (float Range limit from a
     Shape-derived int64 through Cast, Pad amounts from a Shape-derived
     chain) and asserting the resolved output dims; a negative case for
     Pad `mode` `wrap` at load.
  6. clang-format only on new kernel files.
- **Acceptance:** the scoreboard at the top of this file.
- **Budget:** 60 turns.

### Packet P3d — GridSample, Einsum

- **Starting revision:** the branch head after P3c integrates.
- **Writable:** new `engine/src/kernels/{GridSample,Einsum}.cpp`; modify
  `Kernels.hpp`, `CpuKernels.cpp`, `engine/CMakeLists.txt`,
  `engine/src/ModelLoader.cpp` (allowlist, attribute maps, policies,
  accepted-ops message), `engine/src/ShapeInference.cpp` (shape rules and
  `compute_outputs` only), `engine/test/KernelsTest.cpp` and
  `engine/test/ShapeInferenceTest.cpp` (append only).
- **Survey facts:** every GridSample site (RT-DETR v4, D-FINE, DEIMv2,
  RF-DETR, RF-DETR-Seg) is 4-D, `mode` bilinear, `padding_mode` zeros,
  `align_corners` 0; the one Einsum is `bchw,bnc->bnhw` (RF-DETR-Seg).
- **Required final state:**
  1. GridSample (opset 16 attribute names, as opset 18 exports carry): X
     `[N,C,H,W]`, grid `[N,Ho,Wo,2]` holding (x, y) in [-1, 1]; output
     `[N,C,Ho,Wo]`. `mode` bilinear (default) or nearest; `padding_mode`
     zeros (default), border, or reflection; `align_corners` 0 (default) or
     1, with the ONNX unnormalization (`align_corners` 1: -1 and 1 are the
     centres of the corner pixels; 0: the outer edges). Nearest rounds half
     to even, as the ONNX reference does. bicubic is rejected at load.
  2. Einsum: one or two float32 operands; explicit output (`->`) or the
     implicit form (output = the labels that appear once, alphabetical);
     summation over the labels absent from the output; a label repeated
     inside one operand means its diagonal. Ellipsis (`...`) is rejected at
     load. Shape inference derives the output dims from the labels and
     fails on a label bound to two different sizes.
  3. Tests: hand-computed GridSample cases for bilinear and nearest, each
     padding mode on an out-of-range sample, and both `align_corners`
     values; Einsum cases for `bchw,bnc->bnhw`, a matrix product `ij,jk->ik`,
     a trace `ii->` (or `ii` implicit), and a transpose `ij->ji`; negative
     cases for an ellipsis equation and a size mismatch.
  4. clang-format only on new kernel files.
- **Acceptance:** the scoreboard at the top of this file.
- **Budget:** 60 turns.

## Run ledger

| Attempt | Packet | Implementer | Acceptance (first pass) | Integration gate | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | T1 | orchestrator (not delegated) | pass | pass: 13 YOLO exports | Implemented before the orchestration rule; commit `fbefc95` |
| 2 | T2 | orchestrator (not delegated) | pass after two orchestrator fixes | pass: 6 transformer exports + YOLO regression | Implemented before the rule; the YOLO regression rerun caught two defects (empty initializers, empty Resize scales), fixed by the orchestrator |
| 3 | P3a | worker (`sonnet`, worktree `wip/native-p3a`) | pass on the worker's second run (the first build failed on a typo in its own generated `Abs.cpp`; deviation: acceptance run twice) | pass: engine 12/12, NeuriploEngineInferTest, ResNet-18 and YOLO26 parity, format | Diff inside the writable list; planner review found no defects |
| 4 | R1 | worker (`sonnet`, worktree `wip/native-r1`) | fail: scoreboard defect (planner-owned filter matched fixture-dependent parity tests) | planner re-score with corrected scoreboard: tests 13/13, docs in sync, no leftover names; format check fails on two moved test files | Review also found: released 0.10.0 CHANGELOG entry renamed; README anchor and two gtest names left. Sent to R1-repair |
| 5 | R1-repair | worker (`sonnet`, same worktree) | pass | pass: engine 12/12 + NeuriploEngineInferTest, ResNet-18 and YOLO26 parity, docs, format; `act --dryrun` of the renamed CI job green (full local act still blocked by the registry DNS fault noted in Y12; GitHub CI is the real run) | Clean scope; commit `8d6dece` |
| 6 | P3b | worker (`sonnet`, worktree `wip/native-p3b`) | pass | pass: engine 12/12 + NeuriploEngineInferTest, ResNet-18 and YOLO26 parity, format | Clean scope; review checked the generated Greater/Less comparison lines and the evaluator branches. Deviation: the shape test also routes through And (extra coverage) |
| 7 | P3c | worker (`sonnet`, worktree `wip/native-p3c`) | pass | pass: engine 12/12 + NeuriploEngineInferTest, both parities; P3c files clean under clang-format 18.1.3 | Review checked the evaluator float lane and Cast branch. Known limit: an empty Range infers dim 0, which planning will reject later |
| 8 | (CI) | — | — | CI Format Check failed on `YoloParityTest.cpp:228-229` (from R1, attempt 4-5) | Scoreboard defect: local clang-format 21 vs CI 18.1.3. Scoreboard corrected; repair is F1 |
| 9 | F1 | worker (`haiku`, worktree `wip/native-f1`) | pass | pass: format clean under clang-format 18.1.3 and 21 | Whitespace-only diff in `YoloParityTest.cpp` |
| 10 | P3d | worker (`sonnet`) | aborted: session usage limit hit before any edit (worktree clean) | — | Re-dispatch after the limit resets, after R2 |
| 11 | R2 | worker (`sonnet`, worktree `wip/native-r2`) | fail: environment, planner-owned (after the develop merge `versions.env` pins ONNX Runtime 1.28.0, not installed; planner installed it with `scripts/setup_onnx_runtime.sh`) | planner re-score pass: 13/13, docs, format (18.1.3), no leftover names; integration gate on ORT 1.28.0 pass with ResNet-18 2.24e-05 and YOLO26 IoU 0.999999; `act --dryrun` of job `nert` green | Library target `neuriplo_engine` and the C API's `neuriplo_engine_*` symbols untouched, as required. Deviation: worker edited with `sed -i` |
| 12 | D1 | worker (`haiku`, worktree `wip/native-d1`) | pass | pass: links resolve (`engine/README.md`, `#nert-track`) | Accepted deviation: the status note names YOLO11 segmentation and pose only, matching the evidence (no YOLO11 detection row), instead of the packet's broader wording |
