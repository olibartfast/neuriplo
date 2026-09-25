# Orchestration — Stable Consumer C ABI

Same workflow as Phase 2 (`specs/2026-09-22-plugin-abi-loader-hardening/
orchestration.md`), which is the reference for the harness inventory and the
enforced-vs-advisory table; only what differs is stated here.

## Roles and routing

| Role | Tier | Owns |
| --- | --- | --- |
| Specifier | Strongest | This packet, Group 0 (header contract, acceptance suite, symbol list), [Q-n] decisions, accepting or rejecting output, Group 6 docs |
| Planner | Mid (may be the specifier's session) | Sequencing, packets, routing defects to fresh workers |
| Implementer | Cheap | Groups 1, 2, 4, 5; Group 3 after [Q-1] |
| Reviewer | Strongest, read-only | Diff review against the packet; may reject |

Group 0 is not delegable: it writes the header every group implements against
and the suite that scores them. Lesson carried from Phase 2: the suite must
include the conforming edge cases (empty outputs, trailing `struct_size`
fields), not only the malformed ones — a missing conforming case is how a
worker's over-strict guard went unnoticed there.

## Acceptance

The command in `validation.md`, run once per attempt as the worker's final
action. Implementation groups score a subset (`ctest -R` limited to their
suites) until Group 5 lands the symbol check; the full command then applies.

Read-only to every worker: `include/neuriplo/neuriplo_c.h` (the contract —
a worker that needs it changed stops and reports), `CApiContractTest.c`,
`CApiWrapperTest.cpp`, `CApiHeaderCheck.c`, `CApiHeaderCheck.cpp`,
`CheckHeaderIncludes.cmake`, `backends/src/test/CMakeLists.txt`,
`scripts/abi/neuriplo_c.symbols`, the fixtures
(`backends/src/test/plugin_fixtures/`), `plugin_abi.h`, and the existing C++
API headers. The public declarations in `include/neuriplo/neuriplo.hpp` are
read-only too; Group 4 writes only their bodies and private parts.

Targeted checks per group are the anchored `ctest -R` filters in `plan.md`
(Group 0 section). Per [O-1], Groups 1–2 run their filter in both a gcc
(`build-capi`) and a clang (`build-capi-clang`, `CC=clang-18
CXX=clang++-18`, same flags) tree, both with `-DWERROR=ON`; the acceptance
command itself stays gcc.

## Handoff packets

Written by the planner, 2026-09-26, from the Phase 2 template (§4 of its
orchestration file). A packet states obligations and boundaries; it does not
paste the spec or finished code. Corrected in Group 0 (2026-09-26): the stub
file and its `SOURCES` entry already exist, so no group but 3 and 5 touches
the root `CMakeLists.txt`.

Sequence and where each packet runs:

| Packet | Runs in | Starts from | New cases green | Acceptance filter total |
| --- | --- | --- | --- | --- |
| 1a | main checkout | `86098cb` (Group 0) | 7 | 39 (7 + 32 `PluginAbi*`) |
| 1b | main checkout | 1a committed | 8 (Group 1 = 15) | 47 |
| 2 | main checkout | 1b committed | 11 | 58 |
| 3 | worktree A | 2 committed | 7 | TSan 7; gcc 65 |
| 4 | worktree B | 2 committed | 9 | 67 |
| 5 | main checkout | 3 and 4 merged | symbol check + consumer | 74 + symbols + consumer |
| 5b | main checkout | 5 committed and [T-16] written | CI jobs | `act` dry run + local run per new job |

The orchestrator commits between packets (workers never commit). Worktrees A
and B are made from the head of `feature/consumer-c-abi` after Group 2 is
committed (that branch descends from `develop`, per `AGENTS.md`). They are
fresh checkouts with no build trees, so their packets configure from scratch.

**Why Group 1 is split (1a / 1b), by function.** Group 1 as one packet is
about 300 lines plus the error machinery every later group reuses, which
leaves a 12-turn worker no room to debug. The cut is by function, so each
half ends at a checkable boundary. 1a builds everything that does not need a
live backend: the error model, the infallible functions, and
`neuriplo_engine_create` up to backend resolution (steps 1-3 of [D-11]), plus
`_destroy` / `_backend_id`. 1b adds the backend list and the actual engine
construction (steps 4-5 of [D-11]). This differs from the Notes sketch, which
put the list in 1a and `struct_size` in 1b; moving the list to 1b balances
the halves (about 180 and 120 lines), and `struct_size` validation is part of
argument checking, so it belongs in 1a.

**Disjoint writable paths for 3 and 4.** Group 3 writes only
`src/neuriplo_c.cpp`, `src/neuriplo_c_log.cpp` (new), `src/neuriplo_c_internal.hpp`
(only if it must), and `CMakeLists.txt`. Group 4 writes only
`include/neuriplo/neuriplo.hpp`. The overlap would be the spec files: both
workers might want to note a deviation in `plan.md` or evidence in
`validation.md`. Neither may edit `specs/`; they report deviations in the
handback, and the specifier records them.

### Rules common to every packet

- **Working directory.** Every shell command starts with `cd <root> &&`, where
  `<root>` is the checkout the packet runs in (main checkout:
  `/home/oli/repos/neuriplo`). Tool calls reset cwd.
- **Read-only.** You may not edit these: `include/neuriplo/neuriplo_c.h`,
  `include/neuriplo/plugin_abi.h`, `include/InferenceBackendSetup.hpp`,
  `include/common.hpp`, `backends/src/test/CApiContractTest.c`,
  `backends/src/test/CApiWrapperTest.cpp`, `backends/src/test/CApiHeaderCheck.c`,
  `backends/src/test/CApiHeaderCheck.cpp`, `backends/src/test/CheckHeaderIncludes.cmake`,
  `backends/src/test/CMakeLists.txt`, `backends/src/test/plugin_fixtures/**`,
  `scripts/abi/neuriplo_c.symbols`, `specs/**`, and the public declarations in
  `include/neuriplo/neuriplo.hpp`. The same goes for every path not on your
  packet's writable list. If you need one of them changed, stop and report
  it; do not work around it.
- **Permitted commands.** `cmake` (configure / `--build` / `--install`),
  `ctest`, `./scripts/quality/format.sh [--check|--fix]`,
  `./scripts/quality/cppcheck.sh`, `git diff`, `git status`, plus read
  commands (`cat`, `sed -n`, `grep`, `nm`). Group 5 also needs `pkg-config`,
  `cc`/`c++`, `mktemp`, `chmod`. No network, no package installs, no
  `git commit` / `push` / `stash` / `checkout`.
- **Rebuilding.** After editing any `CMakeLists.txt`, reconfigure with
  `cmake --fresh` (a plain reconfigure of a `-DWERROR=ON` tree breaks an
  unrelated test; see `validation.md`). Build with `--parallel` during
  targeted checks. A full build from scratch takes about 25 s with
  `--parallel` on this machine.
- **Contract rules every C function body must follow** (from the header's
  "General contract", [D-15]):
  - Each definition repeats `NEURIPLO_CALL` and `NEURIPLO_NOEXCEPT` exactly as
    the stub does and does not repeat `NEURIPLO_C_API`. Keep the stub
    signatures byte-for-byte.
  - A status-returning function sets every non-NULL out-pointer to NULL/0
    first. It sets the thread's last error to "" on success and to a
    non-empty message on failure. An `INVALID_ARGUMENT` message starts with
    the function's own name (e.g. `"neuriplo_engine_create: config is NULL"`).
  - Infallible functions (`neuriplo_api_version`, `_status_string`,
    `_last_error`, `_engine_destroy`, `_backend_list_release`,
    `_result_release`) never touch the last error, and NULL is a no-op where
    the header says so.
  - No exception leaves a function. Any body that can throw runs inside the
    shared guard (see 1a).
- **Score once.** Run targeted checks as often as you like. Run the packet's
  acceptance command exactly once, verbatim, as your final action. If it
  fails, report the failure. Do not edit and re-run.
- **Handback report.** List the files changed, the last result line of each
  required check (gcc, clang, ASan, TSan as applicable), the acceptance
  result (pass/fail plus ctest's "N% tests passed, X failed out of Y" line),
  any deviation or question, and the number of turns used.
- **Turn budget (12).** Spend at most 2 turns reading, using the packet's
  read list with parallel reads in one turn. Write each file whole in one
  turn. Leave 2 turns for the required checks and the acceptance run.

---

### Group 1a — error model, version, create's validation and resolution

- **Writable:** `src/neuriplo_c.cpp`, `src/neuriplo_c_internal.hpp` (new).
- **Read, in this order (2 turns):**
  1. `include/neuriplo/neuriplo_c.h` lines 15-87 (general contract) and
     304-422 (version/status/last-error, logging, backend discovery,
     lifecycle)
  2. `src/neuriplo_c.cpp` (the stubs, all 89 lines)
  3. `src/InferenceBackendSetup.cpp` lines 38-60 and 84-162
     (`load_configured_plugins`, `available_backend_ids`,
     `setup_inference_engine`)
  4. `backends/src/BackendRuntimeRegistry.hpp` (26 lines) and
     `backends/src/plugin/PluginLoader.hpp` lines 18-24 and 78-97
  5. `backends/src/test/CApiContractTest.c` lines 472-503, 591-606,
     694-737, 772-816, 829-888 (the seven cases below)
- **Required final state:**
  1. `src/neuriplo_c_internal.hpp` (new, `#pragma once`, internal to the
     library and never installed) declares, in `namespace neuriplo_capi`,
     the shared error machinery that later groups reuse:
     - `void set_last_error(const char* message) noexcept;`
     - `void clear_last_error() noexcept;`
     - `neuriplo_status_t fail(neuriplo_status_t status, const char* message) noexcept;`
       (sets the message and returns `status`)
     - `neuriplo_status_t invalid_argument(const char* function, const char* detail) noexcept;`
       (message `"<function>: <detail>"`, returns `INVALID_ARGUMENT`)
     - `enum class GuardContext { Create, Infer, Other };`
     - `template <typename Body> neuriplo_status_t guarded(GuardContext context, const char* function, Body&& body) noexcept`,
       which runs `body()` (it returns `neuriplo_status_t`) and maps
       exceptions per [D-11]: `std::bad_alloc` → `OUT_OF_MEMORY`; other
       `std::exception` → `MODEL_LOAD` (Create) / `INFERENCE` (Infer) /
       `INTERNAL` (Other); `catch (...)` → `INTERNAL`. The message is
       `"<function>: <what()>"`, or `"<function>: unknown exception"`.

     The helpers are defined in `neuriplo_c.cpp`. The last error is one
     `thread_local std::string`. The setters must not throw: if storing the
     message fails, `neuriplo_last_error()` must still return a non-empty
     static string for that thread (a `thread_local` flag is enough).
  2. `neuriplo_api_version()` returns `NEURIPLO_C_API_VERSION`.
     `neuriplo_status_string()` returns a distinct, static, non-empty string
     for each of the 8 statuses, and a generic non-empty one for any other
     value. `neuriplo_last_error()` returns the thread's message, never NULL.
  3. `struct neuriplo_engine_t` is defined at global scope in
     `neuriplo_c.cpp` with `std::unique_ptr<InferenceInterface> backend;` and
     `std::string backend_id;` (the resolved id). Groups 2-3 append members.
     `neuriplo_engine_destroy` deletes it (NULL is a no-op).
     `neuriplo_engine_backend_id` validates its arguments and returns
     `backend_id.c_str()`.
  4. `neuriplo_engine_create`, steps 1-3 of [D-11], inside
     `guarded(GuardContext::Create, "neuriplo_engine_create", ...)`:
     - Step 1, validate in this order: `out_engine` NULL; set
       `*out_engine = NULL`; `config` NULL; `struct_size` per [D-12] (less
       than `sizeof(neuriplo_engine_config_t)` → invalid; greater → every
       byte in `[sizeof(neuriplo_engine_config_t), struct_size)` of the
       caller's struct must be 0, else invalid); then `model_path` NULL;
       `input_sizes` NULL with `n_input_sizes > 0`; any entry with
       `dims == NULL && ndim > 0`. Read no field before `struct_size` has
       been checked.
     - Step 2: call `available_backend_ids(plugin_dir)` first (NULL → "").
       This scans the plugin directory, and it must run before resolving
       the default. Keep the returned list. Resolve the id: a non-empty
       `backend_id` as given; otherwise
       `get_compiled_backend_registration()->id`, else
       `get_plugin_backends().front().id`, else none.
     - Step 3: if the resolved id is not in the list (or there is none),
       return `BACKEND_NOT_FOUND` with a message that names the requested id
       (or "default") and lists every available id, comma-separated.
     - Temporary until 1b: when step 3 passes, return
       `fail(NEURIPLO_STATUS_UNIMPLEMENTED, "neuriplo_engine_create: engine construction not implemented yet (Group 1b)")`.
  5. Every other function stays exactly as stubbed. That covers the backend
     list (1b) and metadata, infer, result, and log (Groups 2-3).
- **Exact identifiers:** `available_backend_ids` (`InferenceBackendSetup.hpp`),
  `get_compiled_backend_registration` (`BackendRuntimeRegistry.hpp`),
  `get_plugin_backends` (`plugin/PluginLoader.hpp`), and `InferenceInterface`.
  All of these are on `neuriplo`'s private include path, so include them as
  `"InferenceBackendSetup.hpp"`, `"BackendRuntimeRegistry.hpp"`, and
  `"plugin/PluginLoader.hpp"`.
- **Required checks (all green before acceptance):**
  - gcc: `cmake --build build-capi --parallel && ctest --test-dir build-capi -R '<1a filter>' --output-on-failure`
  - clang ([O-1]): `CC=clang-18 CXX=clang++-18 cmake --fresh -S . -B build-capi-clang -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON && cmake --build build-capi-clang --parallel && ctest --test-dir build-capi-clang -R '<1a filter>' --output-on-failure`
  - `./scripts/quality/format.sh --fix` then `./scripts/quality/cppcheck.sh`

  The 1a filter is:
  `^CApi\.(Lifecycle\.(ApiVersion|StatusStrings|UnknownBackend)|StructSize\.(TooSmall|NonZeroTrailing)|Error\.NullArguments|LastError\.PerThread)$`
  (7 cases).
- **Acceptance (once, final action):**
  ```bash
  cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
    && cmake --build build-capi \
    && ctest --test-dir build-capi -R '^CApi\.(Lifecycle\.(ApiVersion|StatusStrings|UnknownBackend)|StructSize\.(TooSmall|NonZeroTrailing)|Error\.NullArguments|LastError\.PerThread)$|PluginAbi' --output-on-failure \
    && ./scripts/quality/format.sh --check
  ```
  Expected: 39/39 pass. The other 8 Group 1 cases still fail (1b's work).
- **Stop condition:** the acceptance run has been reported. Also stop and
  report, without acceptance, if a case in the 1a filter can only pass by
  changing a read-only file.

### Group 1b — backend list and engine construction

- **Writable:** `src/neuriplo_c.cpp`, `src/neuriplo_c_internal.hpp` (only to
  add a helper; do not change 1a's declarations).
- **Read, in this order (2 turns):**
  1. `src/neuriplo_c.cpp` and `src/neuriplo_c_internal.hpp` as 1a left them
  2. `include/neuriplo/neuriplo_c.h` lines 225-258 (config struct) and
     349-422 (backend discovery, lifecycle)
  3. `include/InferenceBackendSetup.hpp` (`EngineOptions`, 30 lines) and
     `src/InferenceBackendSetup.cpp` lines 106-162
  4. `backends/src/test/CApiContractTest.c` lines 505-590, 608-683, 711-724,
     742-771, 817-828
- **Required final state:**
  1. `struct neuriplo_backend_list_t` is defined at global scope and owns a
     `std::vector<std::string>`. `neuriplo_available_backends(plugin_dir, out_list)`
     works as follows: NULL `out_list` → invalid; otherwise set
     `*out_list = NULL` and, inside `guarded(GuardContext::Other, ...)`,
     take `available_backend_ids(plugin_dir ? plugin_dir : "")` into a new
     list, clear the error, and return OK. `neuriplo_backend_list_count` /
     `_get` perform the header's argument and index checks (`_get` with
     `index >= count` → invalid, message starting
     `"neuriplo_backend_list_get"`). `neuriplo_backend_list_release` deletes
     the list (NULL is a no-op).
  2. `neuriplo_engine_create`: replace 1a's `UNIMPLEMENTED` placeholder with
     steps 4-5 of [D-11]:
     - Build `EngineOptions`: `model_path`; `backend_id` = the **resolved**
       id (always explicit, never ""); `use_gpu = config->use_gpu != 0`;
       `batch_size = config->batch_size == 0 ? 1 : config->batch_size`;
       `input_sizes` from `input_sizes[0..n)` (an entry with `ndim == 0`
       becomes an empty vector, and `dims` is not read); `plugin_dir = ""`
       **always** — step 2 already scanned `config->plugin_dir`, and passing it
       again would re-log and re-`dlopen` every rejected plugin ([D-18];
       amended by the specifier 2026-09-26). The host still scans
       `NEURIPLO_PLUGIN_DIR` either way.
     - Call `setup_inference_engine(options)` inside a local `try`.
       `std::bad_alloc` is rethrown to the guard. Any other
       `std::exception` → `MODEL_LOAD` with the message
       `"neuriplo_engine_create: backend '<id>' failed to load model '<path>': <what()>"`.
     - A `nullptr` result → `MODEL_LOAD` with the message
       `"neuriplo_engine_create: backend '<id>' failed to load model '<path>'"`.
       Both the id and the path are required: the suite checks them.
     - On success, allocate the engine, move the backend in, store the
       resolved id, set `*out_engine`, clear the error, and return OK. Leak
       nothing on any failure path.
  3. Nothing else changes. Metadata, infer, result, and log stay stubs.
- **Exact identifiers:** `EngineOptions`, `setup_inference_engine(const EngineOptions&)`,
  `available_backend_ids`, `neuriplo_capi::guarded`, `GuardContext::Create`,
  `GuardContext::Other`.
- **Required checks:** as 1a (gcc, clang, format, cppcheck), using the
  Group 1 filter `^CApi\.(Lifecycle|StructSize|Error|LastError)\.` (15
  cases).
- **Acceptance (once, final action):**
  ```bash
  cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
    && cmake --build build-capi \
    && ctest --test-dir build-capi -R '^CApi\.(Lifecycle|StructSize|Error|LastError)\.|PluginAbi' --output-on-failure \
    && ./scripts/quality/format.sh --check
  ```
  Expected: 47/47 pass.
- **Stop condition:** as 1a.

### Group 2 — metadata views and inference results

- **Writable:** `src/neuriplo_c.cpp`, `src/neuriplo_c_internal.hpp` (only to
  add a helper).
- **Read, in this order (2 turns):**
  1. `src/neuriplo_c.cpp` as 1b left it
  2. `include/neuriplo/neuriplo_c.h` lines 260-302 (info/view structs) and
     424-492 (metadata, inference, result)
  3. `backends/src/InferenceInterface.hpp` lines 1-20 (`RawOutputTensor`) and
     53-65; `backends/src/InferenceMetadata.hpp`; `backends/src/TensorDtype.hpp`;
     `backends/src/TensorDataType.hpp`
  4. `backends/src/test/CApiContractTest.c` lines 890-1220
- **Required final state:**
  1. **Metadata ([T-8], [R-3]).** In `neuriplo_engine_create`, right after
     `setup_inference_engine` succeeds and still inside the Create guard (so
     a throw here maps to `MODEL_LOAD`), the engine takes one copy of
     `backend->get_inference_metadata()` (`getInputs()` / `getOutputs()`)
     into its own storage. Then it builds one `neuriplo_tensor_info_t` per
     layer: `struct_size = sizeof(neuriplo_tensor_info_t)`, `name` =
     `c_str()` of the stored name, `shape` = `data()` of the stored shape
     (NULL when empty), `ndim`, `batch_size`, and `dtype` mapped from
     `TensorDataType` (Float32 0, Int32 1, Int64 2, UInt8 3, Int8 4,
     Bool 5; use an explicit switch with a fallback return after it).
     **Take the pointers only after every container is final.** No
     `push_back` afterwards, and no `std::string` object moved after its
     `c_str()` is taken: short-string storage lives inside the object. The
     simplest way is to fill the members of the already-allocated engine in
     place.
  2. `neuriplo_engine_input_count` / `_output_count` / `_input` / `_output`
     perform the header's argument and index checks, with outs nulled and
     messages prefixed by the function name. They return pointers into the
     engine, are lock-free, and never call the backend.
  3. **Inference ([T-9], [R-4]).** `struct neuriplo_result_t` owns
     `std::vector<RawOutputTensor> outputs` and
     `std::vector<neuriplo_tensor_view_t> views`. `neuriplo_infer` checks
     only what the header lists: engine NULL, `out_result` NULL, `inputs`
     NULL with `n_inputs > 0`, or an input with `data` NULL and
     `size_bytes > 0` → `INVALID_ARGUMENT` "neuriplo_infer: ...". `(NULL, 0)`
     and `{NULL, 0}` inputs are valid, and the backend decides on them.
     It copies each input into a `std::vector<uint8_t>` (the C++ interface
     takes owned bytes) and calls `backend->get_infer_results_raw(...)`
     inside `guarded(GuardContext::Infer, "neuriplo_infer", ...)`, so the
     plugin's message lands in the last error. It moves the returned vector
     into a heap `neuriplo_result_t`, then builds `views` from the moved
     storage: `struct_size`, `dtype` (the `TensorDtype` value, 0-3), `data`
     (NULL when `bytes` is empty), `size_bytes`,
     `element_count = RawOutputTensor::element_count()`, and `shape`
     (NULL when empty) and `ndim`. The result never references the engine
     ([D-13]).
  4. `neuriplo_result_output_count` / `_output` perform the header's checks.
     `neuriplo_result_release` deletes the result (NULL is a no-op).
  5. **Do not add the per-engine mutex.** Group 3 owns [D-7], so the lock
     and the Thread cases are scored in one place.
- **Exact identifiers:** `InferenceInterface::get_inference_metadata`,
  `InferenceMetadata::getInputs` / `getOutputs`, `LayerInfo` (`name`,
  `shape`, `batch_size`, `datatype`),
  `InferenceInterface::get_infer_results_raw`, `RawOutputTensor` (`dtype`,
  `bytes`, `shape`, `element_count()`), `TensorDtype`, `TensorDataType`,
  `neuriplo_capi::guarded`, `GuardContext::Infer`.
- **Required checks:**
  - gcc and clang as in 1a, with the Group 2 filter
    `^CApi\.(Metadata|Infer|InferError)\.` (11 cases)
  - **ASan/LSan/UBSan ([V-5])**, in tree `build-capi-asan` (Debug, no
    `WERROR`). Building only the suite's target takes about 7 s:
    ```bash
    cmake --fresh -S . -B build-capi-asan -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON \
          -DSANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug \
      && cmake --build build-capi-asan --target CApiContractTest --parallel \
      && ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
         LSAN_OPTIONS=suppressions=$PWD/scripts/quality/lsan-suppressions.txt \
         ctest --test-dir build-capi-asan -R '^CApi\.(Metadata|Infer|InferError)\.' --output-on-failure
    ```
    Expected: 11/11 pass with no sanitizer report. `$PWD` must be the repo
    root. Do not widen the filter: the Group 1 cases load OpenCV, whose
    leaks are not this group's to judge.
  - format `--fix` and cppcheck, as in 1a
- **Acceptance (once, final action):**
  ```bash
  cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
    && cmake --build build-capi \
    && ctest --test-dir build-capi -R '^CApi\.(Lifecycle|StructSize|Error|LastError|Metadata|Infer|InferError)\.|PluginAbi' --output-on-failure \
    && ./scripts/quality/format.sh --check
  ```
  Expected: 58/58 pass.
- **Stop condition:** as 1a. Report the last ASan ctest summary line too.

### Group 3 — same-engine lock and log callback (worktree A)

- **Writable:** `src/neuriplo_c.cpp`, `src/neuriplo_c_log.cpp` (new),
  `CMakeLists.txt` (root; only to add `${CMAKE_CURRENT_LIST_DIR}/src/neuriplo_c_log.cpp`
  to `SOURCES`, directly after the `neuriplo_c.cpp` line), and
  `src/neuriplo_c_internal.hpp` (only if a declaration must be shared).
  **Disjoint from Group 4**, which runs at the same time and writes only
  `include/neuriplo/neuriplo.hpp`.
- **Read, in this order (2 turns):**
  1. `include/neuriplo/neuriplo_c.h` lines 72-80 (threads), 186-206 (log
     types), and 324-347 (`neuriplo_set_log_callback`)
  2. `src/neuriplo_c_internal.hpp` and `src/neuriplo_c.cpp` (the
     `neuriplo_infer` body and the `neuriplo_set_log_callback` stub)
  3. `/usr/include/glog/logging.h` lines 1740-1785 (`LogSink`,
     `AddLogSink`)
  4. `backends/src/test/CApiContractTest.c` lines 1221-1485
- **Required final state:**
  1. **[D-7] / [T-10].** `neuriplo_engine_t` gains a `std::mutex`, held with
     `std::lock_guard` around the `get_infer_results_raw` call in
     `neuriplo_infer` only. It does not cover the input copy, result
     building, metadata getters, or `_backend_id`. Different engines never
     share a lock. This lock also keeps `InferenceInterface`'s own counters
     race-free under TSan.
  2. **[D-16] / [T-11].** Delete the `neuriplo_set_log_callback` stub from
     `neuriplo_c.cpp` and define the function in `src/neuriplo_c_log.cpp`
     (same signature rules; include `"neuriplo_c_internal.hpp"` for the
     error helpers):
     - Validation: a non-NULL `callback` with `min_level` not one of the three
       defined levels → `INVALID_ARGUMENT` "neuriplo_set_log_callback: ...".
       A NULL `callback` removes the current one and ignores `min_level`.
       Success clears the error.
     - One sink class derived from `google::LogSink`, overriding **only** the
       `const google::LogMessageTime&` overload of `send` with `override`.
       No using-declaration is needed: the planner checked that this is
       warning-clean under gcc 13 and clang 18 with
       `-Wall -Wextra -Wpedantic -Werror`. Register one heap instance, never
       deleted, with `google::AddLogSink` under `std::call_once` the first
       time a non-NULL callback is installed, and never remove it. Do not
       call glog while holding your mutex, and do not log from
       `neuriplo_set_log_callback`. Do not call `InitGoogleLogging`.
     - Shared state {callback, min level, user_data} is guarded by one
       `std::mutex`. `send` holds it while invoking the callback, and
       `set_log_callback` takes it to swap the state. That gives the
       completion guarantee: when set returns, no old invocation is running.
     - In `send`, `message` is **not NUL-terminated at `message_len`**. Copy
       `std::string(message, message_len)` and pass `.c_str()`. Map
       `GLOG_INFO` → INFO, `GLOG_WARNING` → WARNING, and `GLOG_ERROR` /
       `GLOG_FATAL` → ERROR. Deliver only when the level is `>= min_level`.
       `send` must not throw: wrap its body in `try { } catch (...) { }`.
- **Exact identifiers:** `google::LogSink`, `google::LogMessageTime`,
  `google::LogSeverity`, `google::AddLogSink`, `google::GLOG_INFO` /
  `GLOG_WARNING` / `GLOG_ERROR` / `GLOG_FATAL`, `neuriplo_log_callback_t`,
  `neuriplo_log_level_t`.
- **Setup (fresh worktree, first turn after reading):** configure and build
  both trees once with `--parallel`, using the two commands in Acceptance
  below with `--parallel` added to each build step, and skip ctest.
- **Required checks:**
  - TSan tree `build-capi-tsan` (clang 18, Debug, no suppressions, [D-17]):
    ctest with `-R '^CApi\.(Thread|Log)\.'` (7 cases), no TSan report
  - gcc tree `build-capi`: `-R '^CApi\.'` (33 cases)
  - format `--fix` and cppcheck
- **Acceptance (once, final action):**
  ```bash
  CC=clang-18 CXX=clang++-18 cmake --fresh -S . -B build-capi-tsan -DDEFAULT_BACKEND=OPENCV_DNN \
        -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug \
        "-DCMAKE_C_FLAGS=-fsanitize=thread -g" "-DCMAKE_CXX_FLAGS=-fsanitize=thread -g" \
        -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread \
        -DCMAKE_MODULE_LINKER_FLAGS=-fsanitize=thread \
    && cmake --build build-capi-tsan --target CApiContractTest \
    && TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-capi-tsan -R '^CApi\.(Thread|Log)\.' --output-on-failure \
    && cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
    && cmake --build build-capi \
    && ctest --test-dir build-capi -R '^CApi\.|PluginAbi' --output-on-failure \
    && ./scripts/quality/format.sh --check
  ```
  Expected: TSan 7/7, then gcc 65/65 (all 33 `CApi.*` + 32 `PluginAbi*`).
  `^CApi\.` excludes `CApiWrapper.*`, which is Group 4's.
- **Stop condition:** as 1a. If TSan reports a race inside glog or the
  plugin loader rather than in `src/neuriplo_c*.cpp`, stop and report the
  trace. Do not add a suppression ([D-17]).

### Group 4 — C++ wrapper bodies (worktree B)

- **Writable:** `include/neuriplo/neuriplo.hpp`. Only the bodies, private
  members, and private helpers, plus removal of `detail::unimplemented`.
  Every public declaration, including its `const`/`noexcept`, stays exactly
  as in the Group 0 skeleton. **Disjoint from Group 3.**
- **Read, in this order (2 turns):**
  1. `include/neuriplo/neuriplo.hpp` (the skeleton, 201 lines)
  2. `specs/2026-09-25-consumer-c-abi/plan.md`, the [T-12] section
     (lines 140-171): the exact required behaviour, method by method
  3. `include/neuriplo/neuriplo_c.h` lines 225-302 (structs) and 304-492
     (functions)
  4. `backends/src/test/CApiWrapperTest.cpp` lines 95-266
- **Required final state:** every bullet of [T-12] holds. The points a worker
  most often misses:
  - Include only `"neuriplo_c.h"` and standard headers (no `/` or `.` in an
    angle include). The build's include audit fails otherwise.
  - A moved-from `Engine` or `Result` gets `INVALID_ARGUMENT` by passing
    its NULL handle to the C function, not by a wrapper-side check.
  - `Engine::operator=(Engine&&)` and `Result::operator=(Result&&)` stay
    `noexcept`. They destroy/release the target's handle first and are
    self-move safe. The destructors call `neuriplo_engine_destroy` /
    `neuriplo_result_release`.
  - `backends()` releases the C list on every path, including when a later
    call throws, via a small RAII holder or try/catch.
  - `TensorView::data_as<T>()` must not throw for a matching type on a
    zero-element tensor (it may return nullptr). T outside the four
    supported types is a `static_assert` with a dependent-false condition.
  - `check()`: when `neuriplo_last_error()` is empty, fall back to
    `neuriplo_status_string(status)`.
- **Exact identifiers:** as declared in the skeleton; every C function in
  `neuriplo_c.h` except `neuriplo_set_log_callback` (not wrapped in v1,
  [D-16]).
- **Setup (fresh worktree):** run the gcc configure + build from Acceptance
  once with `--parallel`, and skip ctest.
- **Required checks:**
  - gcc `build-capi`: `-R '^CApiWrapper\.'` (9 cases). The build also runs
    `neuriplo_capi_include_audit` and compiles `CApiHeaderCheck.cpp`, which
    asserts the move traits.
  - clang: `CC=clang-18 CXX=clang++-18 cmake --fresh -S . -B build-capi-clang -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON && cmake --build build-capi-clang --parallel && ctest --test-dir build-capi-clang -R '^CApiWrapper\.' --output-on-failure`
  - format `--fix` and cppcheck
- **Acceptance (once, final action):**
  ```bash
  cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
    && cmake --build build-capi \
    && ctest --test-dir build-capi -R '^CApiWrapper\.|^CApi\.(Lifecycle|StructSize|Error|LastError|Metadata|Infer|InferError)\.|PluginAbi' --output-on-failure \
    && ./scripts/quality/format.sh --check
  ```
  Expected: 67/67 pass (9 wrapper + 26 C + 32 plugin). Thread/Log cases are
  excluded because Group 3's work is not in this worktree.
- **Stop condition:** as 1a. If a wrapper case cannot pass without changing
  a public declaration, stop and report it.

### Group 5 — packaging, consumer build, symbol check

- **Writable:** `CMakeLists.txt` (root; append a delimited install section
  and change nothing that existing targets build or link),
  `cmake/neuriplo-config.cmake.in` (new), `neuriplo.pc.in` (new, repo root),
  `test/consumer/**` (new), `scripts/abi/check_symbols.sh` (new, executable).
- **Read, in this order (2 turns):**
  1. `CMakeLists.txt` lines 85-159 (library target, include dirs, link
     libraries)
  2. `test/public-headers/CMakeLists.txt` lines 1-20 (the standalone-project
     pattern) and `scripts/abi/neuriplo_c.symbols`
  3. `include/neuriplo/neuriplo_c.h` lines 385-492 and `include/neuriplo/neuriplo.hpp`
     lines 59-201 (what the consumer programs call)
  4. `specs/2026-09-25-consumer-c-abi/requirements.md` [R-9]-[R-11]
     (lines 101-117)
- **Required final state:**
  1. **Install ([T-13], [R-10])**, guarded by
     `option(NEURIPLO_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})`
     so a parent project's `add_subdirectory` gains no install rules, using
     `GNUInstallDirs`:
     - **Amended by the specifier ([D-20], 2026-09-26):** do not use
       `PROJECT_IS_TOP_LEVEL` for the default (CMake 3.21+; the project
       declares 3.10, and on older CMake the default would silently be OFF
       for a top-level build). Compute it first —
       `if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)` sets a local
       variable ON, else OFF — and pass that variable to
       `option(NEURIPLO_INSTALL ...)`.
     - `install(TARGETS neuriplo EXPORT neuriplo-targets ...)` with
       LIBRARY/ARCHIVE to `${CMAKE_INSTALL_LIBDIR}` and RUNTIME to
       `${CMAKE_INSTALL_BINDIR}`.
     - Headers: `include/neuriplo/{neuriplo_c.h,neuriplo.hpp,plugin_abi.h}`
       go to `${CMAKE_INSTALL_INCLUDEDIR}/neuriplo`. The existing C++ API
       and its include closure, `include/{InferenceBackendSetup.hpp,common.hpp}`
       plus `backends/src/{InferenceInterface.hpp,BackendState.hpp,InferenceMetadata.hpp,TensorDtype.hpp,TensorDataType.hpp}`,
       go flat to `${CMAKE_INSTALL_INCLUDEDIR}`.
     - `install(EXPORT neuriplo-targets NAMESPACE neuriplo:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuriplo)`.
       The config comes from `configure_package_config_file` on
       `cmake/neuriplo-config.cmake.in`, and a version file comes from
       `write_basic_package_version_file(... COMPATIBILITY SameMajorVersion)`.
       Do not add `find_dependency(glog)` or any other dependency: glog and
       the backends are PRIVATE to a shared library, and a C / wrapper
       consumer must not need them.
     - `neuriplo.pc` is configured from `neuriplo.pc.in` into
       `${CMAKE_INSTALL_LIBDIR}/pkgconfig`. **It must be relocatable**,
       because `cmake --install --prefix X` does not re-run
       `configure_file`. Use `prefix=${pcfiledir}/<rel>`, where `<rel>` comes
       from `file(RELATIVE_PATH ... "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")`.
       The file has `Libs: -L${libdir} -lneuriplo`,
       `Cflags: -I${includedir}`, and `Version: @PROJECT_VERSION@`.
  2. **Consumer proof ([T-14], [R-11]).** `test/consumer/` is a standalone
     CMake project that the root never configures:
     `find_package(neuriplo CONFIG REQUIRED)`, a C99 program `consumer.c`
     (only `neuriplo/neuriplo_c.h`), and a C++17 program `consumer.cpp`
     (only `neuriplo/neuriplo.hpp`), both linking `neuriplo::neuriplo`.
     Each takes `<plugin_dir>` as `argv[1]`, creates `FIXTURE_GOOD` with
     model `"ok"`, infers `[1,2,3,4]`, and checks for `[2,4,6,8]`. On
     success it prints exactly one line, `OK FIXTURE_GOOD 2 4 6 8`, and
     exits 0; on any failure it prints the status and last error and exits
     non-zero. `test/consumer/run.sh <build-dir>` (bash,
     `set -euo pipefail`, temp prefix from `mktemp -d` removed by `trap`)
     does the following:
     - `cmake --install <build-dir> --prefix <tmp>`.
     - Configures `test/consumer` with **only** `-DCMAKE_PREFIX_PATH=<tmp>`,
       builds it, and runs both programs with `<abs build-dir>/plugin_fixtures`,
       requiring the exact line from each.
     - Finds `neuriplo.pc` under `<tmp>`, then uses
       `PKG_CONFIG_PATH=<its dir> pkg-config --cflags --libs neuriplo` to
       compile and link `consumer.c` with `cc -std=c99`, and runs it with
       `LD_LIBRARY_PATH=<tmp libdir>`, requiring the same line.
     - Prints `consumer check: PASS` at the end.
  3. **Symbol check ([T-15], [R-9]).** Usage:
     `scripts/abi/check_symbols.sh <build-dir> [<symbols-file>]`. The
     symbols file defaults to `scripts/abi/neuriplo_c.symbols`; ignore `#`
     lines and blank lines. On Linux, take `<build-dir>/libneuriplo.so`,
     run `nm -D --defined-only` on it, and keep the names starting with
     `neuriplo_`. On macOS, use `nm -gU` on `libneuriplo.dylib`. For
     Windows, leave a comment pointing at `dumpbin /exports bin/neuriplo.dll`.
     Compare the two sorted sets with `LC_ALL=C`. Print each listed symbol
     that is missing and each exported symbol that is unlisted; exit 1 if
     either set is non-empty, and exit 0 printing `symbols: OK (19)` otherwise.
     The optional second argument makes the negative checks possible
     without editing sources.
- **Required checks:** `./test/consumer/run.sh build-capi` and
  `./scripts/abi/check_symbols.sh build-capi`, after
  `cmake --fresh` + build of `build-capi`. Confirm with `git status` that
  `test/consumer/` has no build output inside the source tree.
- **Acceptance (once, final action):** the full `validation.md` command,
  followed by the consumer run and the two negative symbol checks:
  ```bash
  cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
    && cmake --build build-capi \
    && ctest --test-dir build-capi -R "CApi|PluginAbi" --output-on-failure \
    && ./scripts/abi/check_symbols.sh build-capi \
    && ./scripts/quality/format.sh --check \
    && ./test/consumer/run.sh build-capi \
    && ! ./scripts/abi/check_symbols.sh build-capi <(grep -v '^neuriplo_infer$' scripts/abi/neuriplo_c.symbols) \
    && ! ./scripts/abi/check_symbols.sh build-capi <(cat scripts/abi/neuriplo_c.symbols; echo neuriplo_not_exported)
  ```
  Expected: 74/74 ctest, `symbols: OK (19)`, `consumer check: PASS`, and
  both negative runs exit non-zero (the first reports `neuriplo_infer` as
  unlisted, the second reports `neuriplo_not_exported` as missing).
- **Stop condition:** as 1a. Wiring these checks into CI
  (`.github/workflows/`) is outside this packet; Group 5b ([T-19], [D-21])
  owns it. Do not do it.

### Group 5b — CI wiring (stub; planner completes once [T-16] exists)

Added by the specifier 2026-09-26 ([D-21]).

- **Writable:** `.github/workflows/ci.yml` only.
- **Required final state:** plan.md [T-19] — one job (Docker
  `opencvdnn` builder image, fresh `-DWERROR=ON` configure) running
  `check_symbols.sh` with its two negatives, `test/consumer/run.sh`, and
  `smoke_ctypes.py`; one TSan job running the `validation.md` TSan command
  (`^CApi\.(Thread|Log)\.`, no suppressions). No ASan change: the existing
  `sanitizers` job's full `ctest` already covers every `CApi*` case.
- **Permitted commands:** as the common rules, plus `act`.
- **Acceptance:** `act push --job <job> --dryrun`, then a full local `act`
  run of each new job (`docs/LOCAL_CI.md`); both results in the handback.

## Measurement

Run ledger in the same format as Phase 2 §7, one row per attempt, planner and
worker split. Record when the harness does not report a metric rather than
leaving the cell looking measured.

| Attempt | Group | Role | Model / tier | Total ctx | Active ctx | Tool calls | Turns | Wall clock | Cost | First-pass acceptance | Interventions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | Specifier / architect | strongest | 284k | — | 67 | — | 1889 s | — | n/a (writes the suite); [M-4]: 42/42 `CApi*` cases fail against the stubs, 32/32 `PluginAbi*` pass | 0 |
| 2 | 1a–5 | Planner (packets) | strongest | pending (orchestrator fills from the harness) | — | pending | — | pending | — | n/a (writes packets) | 0 |

Totals are the subagent token counts the harness reported; "—" means the
harness did not report it.

## Open questions

- [O-1] **Resolved (2026-09-26): yes.** Implementers of Groups 1–2 run their
  targeted filter in a gcc 13 tree and a clang 18 tree, both
  `-DWERROR=ON` (commands in Acceptance above). Both compilers are installed
  locally, the Group 0 tree builds warning-free under both, and Phase 2's
  Werror failure was compiler-specific. Groups 3–5 may do the same; it is
  required only where the most new C/C++ lands.
