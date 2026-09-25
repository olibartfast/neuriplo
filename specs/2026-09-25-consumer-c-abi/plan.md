# Plan — Stable Consumer C ABI

Thin groups, each ending in something runnable and reviewable. The header
contract and the acceptance suite come first and are specifier-owned; the
implementation groups fill them in one slice at a time.

**Start condition:** Phase 2 (PR #42) merged into `develop`, and
`feature/consumer-c-abi` updated from `develop`. Group 0 needs the fixture
plugins ([A-1]).

`develop` stays green throughout: every group leaves the default `OPENCV_DNN`
configure/build/test path passing.

## Group 0 — Contract and acceptance suite (specifier-owned)

The group that must not be delegated: it fixes the interface and writes the
tests everything else is scored against.

- [T-1] Decide [Q-1], [Q-2], [Q-3] and record them as decisions in
  `requirements.md`. [Q-1] blocks Group 3.
- [T-2] Update `specs/mission.md` ("usable by other C++ consumers" → C++ and
  non-C++ consumers through the stable C ABI) and `specs/tech-stack.md` (the
  consumer boundary is `include/neuriplo/neuriplo_c.h`; list it beside the
  plugin ABI).
- [T-3] Write `include/neuriplo/neuriplo_c.h` in full: types, status enum,
  dtype enum, config/view structs with `struct_size`, every function
  declaration with its ownership, lifetime, thread-safety, and error contract
  in the doc comment. Export and calling-convention macros declared once, in
  the header.
- [T-4] Write the acceptance suite:
  `backends/src/test/CApiContractTest.c` (plain C, built as C, links only
  `neuriplo`) and `backends/src/test/CApiWrapperTest.cpp` (GoogleTest, uses
  only `neuriplo.hpp`), both against the Phase 2 fixtures. Every [V-n] in
  `validation.md` lands as a case. Add `scripts/abi/neuriplo_c.symbols` (the
  committed export list) and the layout-assertion TU.
- [T-5] Stub every declared function to return `NEURIPLO_STATUS_UNIMPLEMENTED`
  so the suite builds and fails loudly. Capture the pre-implementation run —
  the evidence that each check can fail.

**Group 0 delivered (2026-09-26):**

| Artifact | Path | Owner after Group 0 |
| --- | --- | --- |
| C contract (19 functions) | `include/neuriplo/neuriplo_c.h` | specifier (read-only) |
| Stubs | `src/neuriplo_c.cpp` (in `SOURCES`; `NEURIPLO_C_BUILDING` defined PRIVATE on `neuriplo`) | Groups 1–3 |
| Wrapper skeleton | `include/neuriplo/neuriplo.hpp` — exact public API, bodies throw `UNIMPLEMENTED` | Group 4 (bodies only) |
| C suite, 33 cases | `backends/src/test/CApiContractTest.c` | specifier |
| Wrapper suite, 9 cases | `backends/src/test/CApiWrapperTest.cpp` | specifier |
| Build-time checks | `CApiHeaderCheck.c` (C99 layout/enum/signature asserts), `CApiHeaderCheck.cpp` (C++17, wrapper traits), `CheckHeaderIncludes.cmake` (include audit) | specifier |
| Export list | `scripts/abi/neuriplo_c.symbols` | specifier |
| Fixture amendment | `plugin_fixtures/fixture_backend.c` (atomic counters, `slow` mode, `neuriplo_fixture_overlaps`) | specifier |

Test names. Every `CAPI_CASE(Group, Name)` line in the C suite's case table
is registered by CMake as the ctest test `CApi.Group.Name` (one process per
case); the wrapper suite's cases are `CApiWrapper.<Name>` via
`gtest_discover_tests`. The filters below are exact and anchored, so no
group's filter picks up another's cases.

```bash
# Group 1 -- 15 cases
ctest --test-dir build-capi -R "^CApi\.(Lifecycle|StructSize|Error|LastError)\." --output-on-failure
# Group 2 -- 11 cases
ctest --test-dir build-capi -R "^CApi\.(Metadata|Infer|InferError)\." --output-on-failure
# Group 3 -- 7 cases (TSan tree, see validation.md [V-8]; also run in build-capi)
ctest --test-dir build-capi-tsan -R "^CApi\.(Thread|Log)\." --output-on-failure
# Group 4 -- 9 cases
ctest --test-dir build-capi -R "^CApiWrapper\." --output-on-failure
```

## Group 1 — Lifecycle, versioning, errors ([R-1], [R-2], [R-5])

- [T-6] `neuriplo_api_version`, `neuriplo_engine_create` /
  `neuriplo_engine_destroy` over `setup_inference_engine`,
  `neuriplo_available_backends`.
- [T-7] The error model: a single entry-point guard that catches every
  exception and maps it to a status by context per [D-11]
  (`std::bad_alloc` → `OUT_OF_MEMORY`; any `std::exception` → `MODEL_LOAD`
  in create, `INFERENCE` in infer; `catch (...)` → `INTERNAL`) and stores the
  message thread-locally per [D-15]. `struct_size` handling per [D-12]: reject
  smaller-than-v1, accept larger only when the extra bytes are all zero.
  Create resolves the backend and chooses `BACKEND_NOT_FOUND` vs
  `MODEL_LOAD` per [D-11], and hands the host `plugin_dir = ""` after its own
  scan per [D-18].
  - Checks: the Group 1 filter above (15 cases), under gcc and clang with
    `-DWERROR=ON` ([O-1]).
  - Scope: `neuriplo_api_version`, `neuriplo_status_string`,
    `neuriplo_last_error`, the backend list (4 functions),
    `neuriplo_engine_create` / `_destroy` / `_backend_id`. The metadata,
    infer, result, and log functions stay stubs.
  - Shape the planner should pass on: `struct neuriplo_engine_t` in
    `src/neuriplo_c.cpp` owns `std::unique_ptr<InferenceInterface>`, the
    resolved backend id as `std::string`, and (Groups 2–3) the metadata store
    and `std::mutex`; `struct neuriplo_backend_list_t` owns a
    `std::vector<std::string>`. Resolving the default id uses
    `get_compiled_backend_registration()` (`BackendRuntimeRegistry.hpp`) and,
    failing that, `get_plugin_backends().front()` (`plugin/PluginLoader.hpp`)
    — both already on the library's private include path. The last error is a
    `thread_local std::string`. Definitions repeat `NEURIPLO_CALL` and
    `NEURIPLO_NOEXCEPT` and do not repeat `NEURIPLO_C_API`.
  - Sizing: the largest group (roughly 250–300 lines in one file). If a
    12-turn implementer cannot finish it, split it by function as described
    in Notes (1a / 1b).

## Group 2 — Metadata and inference ([R-3], [R-4])

- [T-8] Metadata views built once at engine creation and owned by the engine.
  `TensorDataType` → `neuriplo_tensor_dtype_t`: Float32 0, Int32 1, Int64 2,
  UInt8 3, Int8 4, Bool 5. Every `neuriplo_tensor_info_t` pointer (name,
  shape) must point into storage that never moves after create returns —
  build the containers completely, then take pointers (or `reserve` first).
- [T-9] `neuriplo_infer` returning a result handle that owns the moved
  `std::vector<RawOutputTensor>`; accessors for count, dtype, shape, data;
  `neuriplo_result_release`. The result builds its `neuriplo_tensor_view_t`
  array after the move and never touches the engine again ([D-13]).
  `element_count` = `bytes.size() / tensor_dtype_size(dtype)`. Argument
  checks are only those the header lists; whether inputs fit the model is the
  backend's decision (`INFERENCE`), never the C layer's.
  - Checks: the Group 2 filter above (11 cases) under gcc and clang with
    `-DWERROR=ON`; `Infer.*` clean under ASan/LSan (validation [V-5]).
  - The per-engine lock of [D-7] does not land here: Group 3 owns it, so
    the lock and the `Thread` cases are scored in one place (planner's
    sequencing, accepted 2026-09-26).

## Group 3 — Threading and logging ([R-6], [R-7])

- [T-10] Same-engine policy per [D-7]: a `std::mutex` per engine held across
  the backend call in `neuriplo_infer`, nothing else locked.
- [T-11] Log callback via a glog `LogSink` shim ([A-2]), installed and removed
  thread-safely, per [D-16]. Suggested shape: one sink object registered with
  `google::AddLogSink` the first time a callback is installed and left
  registered; a mutex guarding {callback, min level, user data}; `send`
  (override the `const LogMessageTime&` overload) takes that mutex and
  invokes the callback, so `neuriplo_set_log_callback` (which takes the same
  mutex) returns only when no invocation is in flight. Severity map: INFO →
  INFO, WARNING → WARNING, ERROR and FATAL → ERROR. `set_log_callback` must
  not itself log.
  - Checks: the Group 3 filter above (7 cases) in the TSan tree
    (validation.md, [V-8]), plus the same filter in `build-capi`.

## Group 4 — C++ wrapper ([R-8])

- [T-12] `include/neuriplo/neuriplo.hpp`: `Engine`, `Result`, `Error`,
  `TensorView`, `backends()`; move-only RAII; no neuriplo C++ headers
  included. The public API is fixed by the Group 0 skeleton; Group 4 writes
  the bodies (and may add private members/helpers), removes
  `detail::unimplemented`, and changes no public declaration. Exact
  behaviour required:
  - `check(status)`: no-op on OK; otherwise throw
    `Error(status, neuriplo_last_error())`, falling back to
    `neuriplo_status_string(status)` when the message is empty.
  - `api_version()` → `neuriplo_api_version()`.
  - `backends(plugin_dir)` → the list in library order; the C list is
    released on every path, including when a later call throws.
  - `Engine(const EngineConfig&)`: memset a `neuriplo_engine_config_t`, set
    `struct_size`, point at the config's strings, pass `batch_size` as is,
    build a `std::vector<neuriplo_dims_t>` for `input_sizes`, `check()` the
    create. `~Engine` destroys; move-assign destroys the target's engine
    first and is self-move safe; a moved-from Engine has `handle() ==
    nullptr` and every call on it throws `Error(INVALID_ARGUMENT)` (by
    passing NULL to the C function, not by a wrapper-side check).
  - `backend_id()`, `inputs()`, `outputs()` copy out of the C views.
  - `infer(inputs)`: build a `std::vector<neuriplo_input_view_t>`, call
    `neuriplo_infer`, return `Result(owned)`.
  - `Result`: `size()` / `output(i)` via the C accessors (so out-of-range
    and moved-from throw `INVALID_ARGUMENT`); `~Result` releases; move-assign
    releases the target's result first.
  - `TensorView`: accessors read the wrapped `neuriplo_tensor_view_t`;
    `shape()` copies; `data_as<T>()` returns `static_cast<const T*>(data())`
    when T matches dtype (float/FLOAT32, int32_t/INT32, int64_t/INT64,
    uint8_t/UINT8) and throws `Error(INVALID_ARGUMENT, ...)` otherwise; a T
    outside those four is a compile error (`static_assert`).
  - Checks: the Group 4 filter above (9 cases), gcc and clang; the build
    itself runs the include audit and `CApiHeaderCheck.cpp`.

## Group 5 — Packaging and ABI checks ([R-9], [R-10], [R-11])

- [T-13] `install()` rules, `neuriplo-config.cmake` + targets export,
  `neuriplo.pc`.
- [T-14] `test/consumer/` standalone project (C program + C++ wrapper
  program) built against a `cmake --install` prefix; a script that installs to
  a temporary prefix, builds, and runs both.
- [T-15] `scripts/abi/check_symbols.sh`: exported `neuriplo_*` symbols vs
  `scripts/abi/neuriplo_c.symbols` (`nm -D` on Linux; `dumpbin /exports`
  noted for Windows).
  - Checks: consumer script green; symbol check green, and red when a symbol
    is removed.

## Group 5b — CI wiring ([D-21])

Delegable; runs after [T-16] exists (so after Group 6's smoke script, before
[T-18]'s evidence pass).

- [T-19] Wire the checks CI does not run yet into
  `.github/workflows/ci.yml` (the only writable path):
  - one job, based on the existing `docker/Dockerfile.opencvdnn` builder
    image like `build-warnings`/`sanitizers`, that configures fresh with
    `-DWERROR=ON`, builds, and runs `./scripts/abi/check_symbols.sh` plus its
    two negatives ([V-11]), `./test/consumer/run.sh` ([V-12]), and
    `python3 test/consumer/python/smoke_ctypes.py <prefix>` ([V-13]);
  - one TSan job running the [V-8]/[V-9] command from `validation.md`
    (clang 18, no suppressions). If the image lacks clang-18, installing it
    in the job is CI tooling, not a project dependency. GitHub runners'
    kernels may need `sudo sysctl vm.mmap_rnd_bits=28` on the host before
    `docker run` for TSan to start; do that rather than disabling the job.
  - ASan needs nothing: the `sanitizers` job's full `ctest` already covers
    every `CApi*` case ([D-21]).
  - Checks: `act push --job <job> --dryrun`, then a full local `act` run of
    each new job (`docs/LOCAL_CI.md`), both recorded in the handback.

## Group 6 — Foreign-language proof and documentation ([R-12], [R-13])

- [T-16] `test/consumer/python/smoke_ctypes.py` against the installed library
  and the good fixture plugin.
- [T-17] `docs/C_API.md`: direction diagram, lifecycle, ownership, errors,
  threading, logging, versioning, examples for C, C++, Python, C#/Unity (Unity
  plugin folder layout, `DllImport`, keeping the log delegate alive, running
  inference off the main thread). Link from `Readme.md` and
  `docs/PLUGIN_BACKENDS.md`. States [D-19]: the package needs nothing but
  neuriplo for the C ABI and the wrapper; a consumer of the installed C++ API
  headers adds `find_package(glog)` / `glog::glog`. Also states that the
  `sanitizers` CI job runs the `CApi*` cases on five backends.
- [T-18] Execute everything in `validation.md` and record evidence with
  dates; `CHANGELOG.md` under `[Unreleased]`; roadmap Phase 7 status updated
  only once the evidence is in.

## Notes

- Delegation, routing, permissions, and the acceptance command are in
  `orchestration.md`. Groups 1, 2, 4, and 5 are delegable; Group 0 is
  specifier-owned; Group 3 is delegable only after [Q-1] is decided; Group 6's
  documentation is judgement work.
- Record in this section anything that deviated from the plan and why.
- Group 0 (2026-09-26), deviations and additions:
  - [T-7] amended by [D-11] (exception mapping by context rather than
    "anything else → INTERNAL") and [D-12] (non-zero trailing `struct_size`
    bytes are rejected, not ignored). Both are stricter than the sketch and
    both are covered by cases (`StructSize.NonZeroTrailing`,
    `Lifecycle.DefaultBackendBadModel`).
  - The API gained `neuriplo_engine_backend_id` (backend selection must be
    observable, `specs/mission.md`), `neuriplo_status_string`, and an opaque
    backend list (`neuriplo_backend_list_t` + count/get/release) for
    `neuriplo_available_backends`. 19 exported functions in total.
  - [V-2]'s "create on the compiled-in default backend" can only run as a
    failure path offline: `OPENCV_DNN` needs a real model and none is
    available. `Lifecycle.DefaultBackendBadModel` creates with NULL, "" and
    the explicit default id against a nonexistent model and requires
    `MODEL_LOAD` (not `BACKEND_NOT_FOUND`) with the id and path in the
    message — which proves the default is resolved and reached.
  - The fixture (specifier-owned since Phase 2) was amended: atomic counters
    ([D-17]); a `slow` mode that holds `infer` open 2 ms and counts calls
    overlapping on one instance, exported as `neuriplo_fixture_overlaps`, so
    [D-7] is observable rather than asserted. All 32 `PluginAbi` cases still
    pass.
  - The C suite is one executable taking `<Group>.<Name>`; CMake parses the
    case table, so the table is the only list of cases. Header, layout, and
    include checks are build-time targets (`neuriplo_capi_header_check_c`,
    `neuriplo_capi_header_check_cxx`, `neuriplo_capi_include_audit`), not
    ctest cases: they are properties of the Group 0 header and pass from
    Group 0 on, so as ctest cases they would have been cases that pass
    against the stubs ([M-4]).
  - Until Groups 1–4 land, the 42 `CApi*` cases fail on this branch, so a
    full `ctest` here (and CI on `feature/consumer-c-abi`) is red. This is
    the Phase 2 precedent ("fail loudly is the starting state"); the branch
    must not merge to `develop` before they pass. Every other test stays
    green (35/35 non-`CApi` in `build-capi`).
  - The acceptance command now configures with `cmake --fresh`: re-configuring
    an existing `-DWERROR=ON` tree trips a pre-existing gtest include-path bug
    in `OCVDNNInferTest.cpp` (details in `validation.md`). Implementers
    re-configure with `--fresh` after any `CMakeLists.txt` edit.
  - Post-Group-0 amendments (specifier, 2026-09-26, planner questions):
    [D-18] create passes `plugin_dir = ""` to the host after its own scan
    (changes packet 1b); [D-19] no `find_dependency(glog)`, C++ API
    consumers supply glog, documented in [T-17]; [D-20] install layout
    accepted, `NEURIPLO_INSTALL` default computed without
    `PROJECT_IS_TOP_LEVEL` (changes packet 5); [D-21] + [T-19] new Group 5b
    owns CI wiring; [R-4] wording now states the single input copy; [V-5]'s
    ASan evidence is `^CApi\.(Metadata|Infer|InferError)\.`; [V-11]'s
    negatives run against edited lists (planner's `check_symbols.sh`
    override argument, accepted). Also accepted: the lock ([D-7]) lands in
    Group 3 only, and the private `src/neuriplo_c_internal.hpp` (never
    installed, never included by a public header).
  - Group 1 split, if the planner needs one: 1a = `Lifecycle.ApiVersion`,
    `Lifecycle.StatusStrings` (needs `engine_create(NULL, …)` →
    `INVALID_ARGUMENT` only), `Lifecycle.AvailableBackends`,
    `Error.SuccessClearsLastError` (needs create's `BACKEND_NOT_FOUND`
    path) — so 1a still includes create's argument checks and backend
    resolution; 1b = the rest of create (`MODEL_LOAD`, `struct_size`,
    destroy, backend id). The cleaner cut is by function, not by case.
