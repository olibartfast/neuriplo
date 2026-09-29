# Requirements — Stable Consumer C ABI

Spec: `specs/2026-09-25-consumer-c-abi/requirements.md` ·
Branch: `feature/consumer-c-abi` · Roadmap: Phase 7

## Goal

Let **any third-party application — C++ or not — run inference through
neuriplo** in a way that survives compiler, standard-library, and language
differences between the application and the library.

Today the only way in is the C++ API (`setup_inference_engine`,
`InferenceInterface`). It passes `std::vector`, `std::unique_ptr`,
`std::variant`, and exceptions across the shared-library boundary, so it is
reliable only for applications built with the same toolchain as `libneuriplo`
(as `neuriplo-infer` is). A C++ application built with a different compiler or
MSVC runtime can fail to link or corrupt memory, and C#/Unity, Python, Rust, Go,
or plain C applications cannot call it at all.

This phase adds one stable, versioned C ABI in front of the existing host —
`include/neuriplo/neuriplo_c.h` — plus a header-only C++ wrapper over it, and
makes the library installable so an outside project can find and link it.

## Direction, stated once

Two ABIs, opposite directions, both C:

| ABI | Header | Direction | Who implements | Who calls |
| --- | --- | --- | --- | --- |
| Plugin ABI (Phase 2) | `plugin_abi.h` | inbound — backends plug into neuriplo | backend plugins | the neuriplo host |
| Consumer ABI (this phase) | `neuriplo_c.h` | outbound — applications call neuriplo | `libneuriplo` | third-party applications |

A consumer calls `neuriplo_c.h`; neuriplo selects a compiled-in backend or
loads a plugin, applies the Phase 2 host validation, and runs inference. The
consumer never sees the plugin ABI.

## Context — what exists today

Verified on `origin/develop` @ `4e4e237`:

- `libneuriplo` is built `SHARED`. On Windows it uses
  `WINDOWS_EXPORT_ALL_SYMBOLS`; on POSIX no visibility is set, so every C++
  symbol is exported. Neither is a stable ABI.
- The C++ entry points are `include/InferenceBackendSetup.hpp`
  (`EngineOptions`, `setup_inference_engine`, `available_backend_ids`) and
  `backends/src/InferenceInterface.hpp` (`get_infer_results`,
  `get_infer_results_raw`, `get_inference_metadata`).
- Inputs are raw bytes per input tensor; the backend interprets layout from
  model metadata. Outputs are `RawOutputTensor` (`TensorDtype` + bytes +
  shape). Metadata dtypes are `TensorDataType` (six values, a superset of the
  four output dtypes).
- **There are no `install()` rules.** An outside project can only consume
  neuriplo by `add_subdirectory` or by pointing at the build tree
  (`Readme.md` → the `neuriplo-infer` example).
- `test/public-headers/` already guards that public headers compile without
  OpenCV; it is the model for the consumer-build check here.
- Phase 2 (PR #42, `feature/plugin-abi-loader-hardening`) adds first-party
  fixture plugins that serve deterministic inference with no vendor SDK. This
  phase's acceptance suite runs against them ([A-1]).

## In Scope

- [R-1] **The C header.** `include/neuriplo/neuriplo_c.h` compiles as C99 and
  as C++ with no other neuriplo header, no STL, no OpenCV, no glog.
  `extern "C"`, opaque handle types, fixed-width integer types, an explicit
  calling-convention macro, and an export macro. `NEURIPLO_C_API_VERSION` in
  the header plus `neuriplo_api_version()` at run time.
- [R-2] **Engine lifecycle.** Create an engine from a config struct
  (`backend_id`, `model_path`, `use_gpu`, `batch_size`, `input_sizes`,
  `plugin_dir`) and destroy it. Every config struct carries `struct_size` as
  its first member so fields can be appended without breaking older callers.
  List available backend ids (compiled-in plus loaded plugins).
- [R-3] **Metadata.** Query inputs and outputs: name, shape, batch size, dtype.
  Views stay valid until the engine is destroyed. The dtype enum covers every
  `TensorDataType` value.
- [R-4] **Inference.** Run inference from an array of input views (pointer +
  byte size per input, the existing raw-bytes contract). Success returns a
  library-owned result handle exposing per-output dtype, shape, and data
  views; the caller releases it exactly once. No additional copy beyond what
  `get_infer_results_raw` already makes ([D-4]). Precisely: the C layer
  copies each input view once into the `std::vector<uint8_t>` that
  `get_infer_results_raw` takes — the same copy a same-toolchain C++ caller
  makes to call it — and moves the returned outputs into the result without
  copying them. No other copy of inputs or outputs is allowed.
- [R-5] **Error model.** Every function returns a `neuriplo_status_t` (or is
  documented as infallible). No C++ exception ever crosses the boundary: every
  entry point catches everything and maps it to a status. A thread-local
  `neuriplo_last_error()` returns a human-readable message for the calling
  thread's last failure. Null or malformed arguments return
  `NEURIPLO_STATUS_INVALID_ARGUMENT`, never crash.
- [R-6] **Thread-safety contract.** Different engines are usable concurrently
  from different threads. Engine creation, backend listing, and plugin loading
  are thread-safe. Behaviour of concurrent calls on the *same* engine is
  decided by [Q-1] and documented in the header.
- [R-7] **Logging.** The application can install a log callback that receives
  neuriplo's (and its plugins') log messages with a severity. Without a
  callback, behaviour is unchanged. The application never has to initialise
  glog.
- [R-8] **Header-only C++ wrapper.** `include/neuriplo/neuriplo.hpp`: RAII
  `neuriplo::Engine` and `neuriplo::Result`, errors thrown as
  `neuriplo::Error` *on the application's side*, lightweight view types for
  inputs and outputs. It calls only `neuriplo_c.h`, so it compiles into the
  application and no C++ type crosses the library boundary. This is the
  recommended path for third-party C++ applications.
- [R-9] **ABI stability checks.** The exported `neuriplo_*` symbol set is
  compared against a committed list; struct layouts are pinned with
  `offsetof`/`sizeof` assertions compiled as C. An accidental removal,
  rename, or layout change fails the build.
- [R-10] **Installable package.** `cmake --install` installs the library, the
  public headers (C, C++ wrapper, and the existing C++ API), and a CMake
  package so an outside project can `find_package(neuriplo CONFIG REQUIRED)`
  and link `neuriplo::neuriplo`. A `neuriplo.pc` pkg-config file for non-CMake
  builds. The package itself requires no other dependency; a consumer of
  the installed C++ API headers additionally provides glog ([D-19]).
- [R-11] **Consumer-build proof.** A standalone project under
  `test/consumer/` — never configured by the root `CMakeLists.txt` — builds
  against the *installed* package: one plain-C program and one C++ program
  using the wrapper, each running inference end to end.
- [R-12] **Foreign-language smoke test.** A Python `ctypes` script (standard
  library only) loads the installed shared library, creates an engine on the
  good fixture plugin, runs inference, and checks the result — proving a
  non-C/C++ runtime can use the ABI with nothing but the header's contract.
- [R-13] **Documentation.** `docs/C_API.md`: the direction diagram, lifecycle,
  ownership, errors, threading, logging, versioning policy, and worked
  examples for C, C++ (wrapper), Python (`ctypes`), and C#/Unity
  (`DllImport`), including where to place plugin and framework libraries in a
  Unity project.

## Out of Scope

- `infer_into` / caller-owned output buffers and any zero-copy path. Backends
  materialise outputs in host memory today; writing into caller memory without
  an extra copy needs backend changes. See [D-4], [Q-5].
- Device (GPU) buffers, I/O binding, asynchronous or streaming inference.
- Published language packages (Python wheel, NuGet, Unity `.unitypackage`,
  crates). The ABI makes them possible; shipping them is later work.
- Hiding the existing C++ symbols or changing the C++ API. `neuriplo-infer`
  and other same-toolchain consumers keep working unchanged ([D-1]).
- Any change to backend inference logic, backend selection, device placement,
  or fallback behaviour (`specs/tech-stack.md` non-choices). `force_gpu`
  semantics stay a Phase 3/4 question.
- Plugin ABI changes. `plugin_abi.h` and its version are untouched.

## Decisions

- [D-1] **The C ABI is the stable boundary for every third-party consumer, C++
  included.** The C++ wrapper compiles into the consumer, so the library never
  exchanges C++ types with code it was not built with. The existing C++ API
  stays as-is for same-toolchain consumers; no migration is forced.
- [D-2] **Thin wrapper over the existing host.** The C API calls
  `setup_inference_engine`, `available_backend_ids`, and `InferenceInterface`.
  It adds no inference logic; Phase 2's plugin validation applies to every
  plugin-backed engine automatically.
- [D-3] **Status codes plus thread-local message.** Chosen over error-buffer
  out-parameters (noisy at every call site) and over callbacks (hard to use
  from FFI). Matches common C ABIs (SQLite, ONNX Runtime status objects,
  libcurl).
- [D-4] **Library-owned results in v1.** The result handle owns the moved
  `std::vector<RawOutputTensor>`; views point into it, so the C API adds no
  copy over the C++ raw path. An `infer_into` that writes to caller buffers
  would still copy once today and adds sizing complexity for dynamic shapes;
  it waits until a backend can write into caller memory directly.
- [D-5] **Independent versioning.** `NEURIPLO_C_API_VERSION` starts at 1 and
  is independent of `NEURIPLO_PLUGIN_ABI_VERSION`. Appending a field behind
  `struct_size` or adding a function is a minor, compatible change; removing
  or changing anything existing bumps the version.
- [D-6] **The C API lives in `libneuriplo`** unless [Q-3] decides otherwise:
  one library to ship, and the plugin loader's process-global state is not
  duplicated across two libraries. (Confirmed by [D-9].)

Decisions taken in Group 0 (2026-09-26). The header
`include/neuriplo/neuriplo_c.h` is the authoritative statement of each; the
entries below record the choice and why.

- [D-7] **Same-engine calls are serialised by a per-engine mutex** (resolves
  [Q-1]). `neuriplo_infer` on one engine takes that engine's lock; different
  engines never share a lock. Metadata queries and `neuriplo_engine_backend_id`
  are lock-free because the engine's metadata is immutable after create.
  `neuriplo_engine_destroy` concurrent with any other call on the same engine
  stays undefined — a lock cannot make use-after-free safe. Why: FFI callers
  (Unity jobs, Python threads, C# tasks) share handles by accident, backends
  are not documented as re-entrant, and the lock is negligible next to
  inference. The alternative ("one thread at a time per engine") pushes a
  correctness rule onto every binding author. Observable: the `slow` fixture
  mode counts overlapping calls on one instance (`CApi.Thread.SameEngineSerialised`).
- [D-8] **Eager load; no separate load call** (resolves [Q-2]).
  `neuriplo_engine_create` returns a ready engine or a failure status, the
  existing "constructed == ready" contract of `setup_inference_engine`. A
  separate load would add a state (created-but-not-loaded) every function must
  check, for no caller that needs it today.
- [D-9] **The C API is compiled into `libneuriplo`** (resolves [Q-3], confirms
  [D-6]): `src/neuriplo_c.cpp` in the library sources, exports declared in the
  header. Why: one binary to ship and find; the plugin table and log sink stay
  single process-global objects; the C++ API is untouched. The export check
  ([R-9]) stays unambiguous because every C++ symbol is mangled, so the
  exported `neuriplo_*` names are exactly the C API.
- [D-10] **C#/Unity is a manual check this phase** (resolves [Q-4]): [M-3] only;
  no .NET SDK in CI until a Unity package actually ships.
- [D-11] **How `neuriplo_engine_create` chooses a status.** `setup_inference_engine`
  returns `nullptr` both for an unknown backend and for a load failure, and its
  plugin path can throw, so the C layer does not infer the status from
  `nullptr`. It (1) validates arguments; (2) resolves the id itself — the
  given `backend_id`, or for NULL/"" the compiled-in default registration, else
  the first loaded plugin; (3) checks that id against
  `available_backend_ids(plugin_dir)` and returns `BACKEND_NOT_FOUND` (message:
  the requested id and the available list) when absent; (4) calls
  `setup_inference_engine` with the explicit id — a `nullptr` now means
  `MODEL_LOAD` (message: resolved id and model path); (5) guards the whole call.
  Exception mapping is by context, amending [T-7]: `std::bad_alloc` →
  `OUT_OF_MEMORY` anywhere; during create any `std::exception` → `MODEL_LOAD`;
  during infer any `std::exception` → `INFERENCE`; `catch (...)` →
  `INTERNAL`. Vendor exceptions (`cv::Exception`, `Ort::Exception`) are
  `std::exception`-derived, and a failed inference is not a library defect, so
  mapping them to `INTERNAL` as [T-7] first said would mislabel them.
- [D-12] **`struct_size` follows the extensible-struct rule of Linux's
  `copy_struct_from_user`.** Smaller than v1: `INVALID_ARGUMENT`. Equal: read.
  Larger: accepted only if every byte past the fields this library knows is
  zero, otherwise `INVALID_ARGUMENT`. This amends [T-7]'s "ignore trailing
  fields": silently ignoring a field a newer caller set (a future fallback or
  device policy, say) is exactly the silent substitution `specs/mission.md`
  forbids. Zero-initialised newer structs still work against older libraries.
- [D-13] **Library-produced structs are returned by `const` pointer, never
  copied into caller memory.** `neuriplo_tensor_info_t` (owned by the engine)
  and `neuriplo_tensor_view_t` (owned by the result) carry `struct_size` set by
  the library; an old caller simply never reads appended fields. Array-element
  types (`neuriplo_dims_t`, `neuriplo_input_view_t`) have no `struct_size` and
  are frozen. Results are independent of their engine and may outlive it.
- [D-14] **Consumer names never collide with `plugin_abi.h`.** New
  `neuriplo_tensor_dtype_t` (`NEURIPLO_TENSOR_DTYPE_*`, six values covering
  every `TensorDataType`, 0–3 equal to the plugin/`TensorDtype` values) and
  `neuriplo_log_level_t` (`NEURIPLO_LOG_LEVEL_*`); both headers compile in one
  translation unit (build-time check). Every enum carries a `*_MAX_ENUM_ =
  0x7FFFFFFF` sentinel so it is 32 bits wide under any compiler setting, and
  layouts are pinned with `offsetof`/`sizeof` in C.
- [D-15] **Error-reporting details.** Every status-returning call overwrites the
  thread's last error (empty string on success); infallible functions never
  touch it; an `INVALID_ARGUMENT` message starts with the rejecting function's
  name; non-NULL out-pointers are set to NULL/0 on failure. In C++ the
  declarations are `noexcept` (`NEURIPLO_NOEXCEPT`), and the definitions must
  repeat it, so a missing guard terminates instead of unwinding into C.
- [D-16] **One process-wide log callback, additive, with a completion
  guarantee.** `neuriplo_set_log_callback(callback, min_level, user_data)`
  replaces the previous callback; NULL removes it. It does not silence glog's
  default output (the host application may use glog itself). When the call
  returns, no invocation of the old callback is running or will start, so a
  C# delegate or a `user_data` block can be freed right after. Callbacks must
  not call into neuriplo. The C++ wrapper does not wrap logging in v1: a
  `std::function` owned across a C callback adds a lifetime problem for no
  functional gain.
- [D-17] **[V-8] runs under TSan with no suppressions.** The fixture's
  bookkeeping counters were plain globals and would race across engines; they
  are made atomic in the fixture (specifier-owned) rather than suppressed, so
  no suppression can ever hide a race in the host.

Decisions taken after Group 0, on questions the planner raised (2026-09-26):

- [D-18] **Create scans `plugin_dir` once** (amends [D-11] step 4). The C layer
  scans `config->plugin_dir` itself (step 2, `available_backend_ids`) and then
  calls `setup_inference_engine` with `EngineOptions::plugin_dir = ""` and the
  explicit resolved id. Equivalence: the host's `load_configured_plugins("")`
  still scans `NEURIPLO_PLUGIN_DIR`, exactly as it would with the directory
  passed through (that scan never depended on `plugin_dir`); plugins are never
  unloaded, so every plugin step 2 saw is still loaded; and with an explicit id
  the host resolves through the process-global tables
  (`find_backend_registration`, `find_plugin_backend`), not through the
  directory. The only observable difference is that a rejected plugin in
  `plugin_dir` is logged, and re-`dlopen`ed, once per create instead of twice.
  The `NEURIPLO_PLUGIN_DIR` directory is still scanned twice per create; that
  is the host's existing behaviour and out of scope.
- [D-19] **The package promises the C ABI and the wrapper without glog; the
  installed C++ API needs glog, supplied by the consumer.** `neuriplo-config.cmake`
  has no `find_dependency(glog)`: glog is PRIVATE to the shared library, and
  the consumers [R-10]/[R-11] target (C, wrapper, FFI) must configure with
  nothing but neuriplo. The C++ API headers are still installed ([R-10]) for
  same-toolchain consumers; because `common.hpp` includes `<glog/logging.h>`,
  such a consumer also runs `find_package(glog)` and links `glog::glog`
  itself. glog is already a required dependency of the project, so this adds
  none. `docs/C_API.md` ([T-17]) states it. Adding `find_dependency` later,
  behind an opt-in component, would be a compatible change.
- [D-20] **Install layout** (planner's proposal, accepted with one change).
  `include/neuriplo/{neuriplo_c.h,neuriplo.hpp,plugin_abi.h}` install to
  `<includedir>/neuriplo`; the C++ API and its closure
  (`InferenceBackendSetup.hpp`, `common.hpp`, `InferenceInterface.hpp`,
  `BackendState.hpp`, `InferenceMetadata.hpp`, `TensorDtype.hpp`,
  `TensorDataType.hpp`) install flat to `<includedir>`, mirroring the in-tree
  include path so existing `#include "InferenceBackendSetup.hpp"` code resolves
  unchanged. Nothing else is installed (not `src/neuriplo_c_internal.hpp`, not
  `PluginLoader.hpp`). `neuriplo.pc` is relocatable via `${pcfiledir}`.
  **Change:** the `NEURIPLO_INSTALL` default must not use
  `PROJECT_IS_TOP_LEVEL`, which only exists from CMake 3.21 while the project
  declares 3.10 — on older CMake the default would silently be OFF for a
  top-level build. Compute it as `CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR`.
- [D-21] **CI ownership.** The existing `sanitizers` job in
  `.github/workflows/ci.yml` already runs the full `ctest` under
  ASan/LSan/UBSan for five backends, so every `CApi*` case gets ASan coverage
  in CI with no change. Not covered anywhere: the TSan run ([V-8], [V-9]),
  `check_symbols.sh` and its negatives ([V-11]), `test/consumer/run.sh`
  ([V-12]), and `smoke_ctypes.py` ([V-13]). They are wired by a new
  Group 5b ([T-19], plan.md), delegable, writable
  `.github/workflows/ci.yml` only, validated with `act` per `AGENTS.md`, and
  run after [T-16] exists so all four land in one workflow change.

## Constraints

- No new dependency (`specs/tech-stack.md`). The Python smoke test uses the
  standard library only; `python3` is already required by the quality scripts.
- C++17 for the implementation and wrapper; the C header is C99-clean.
- The default `OPENCV_DNN` build and all existing tests stay green; the
  existing C++ API compiles and behaves identically.
- Windows stays buildable: the export macro and calling convention are
  correct under MSVC (Phase 2 showed MSVC rejects a `dllexport` added on
  redeclaration — declare exports in the header, once).
- Documentation-only commits carry `[skip ci]`.

## Dependencies

- Phase 2 (PR #42) merged into `develop`: the fixture plugins, host
  validation, and thread-safe descriptor access this phase builds on.
- `include/InferenceBackendSetup.hpp`, `backends/src/InferenceInterface.hpp`,
  `backends/src/TensorDataType.hpp`, `backends/src/TensorDtype.hpp`.
- `CMakeLists.txt` (library target, new install/export rules),
  `test/public-headers/` (pattern for the standalone consumer project).
- `specs/mission.md` — its "usable by other C++ consumers" line widens to C
  ABI consumers; that constitution change lands in this branch ([T-2]).

## Assumptions & Open Questions

- [A-1] The Phase 2 fixture plugins (`FIXTURE_GOOD`, `FIXTURE_SCRIPTED`) give
  the C API suite deterministic, SDK-free inference and every failure mode it
  needs to map to statuses. Confirm in Group 0 once PR #42 has merged.
  **Confirmed 2026-09-26**, with one amendment: the fixture gained atomic
  counters, a `slow` mode, and a `neuriplo_fixture_overlaps` export so [D-7]
  and [D-17] are observable (plan.md Notes).
- [A-2] glog's `LogSink` can forward messages to a C callback on every glog
  version the project supports (the LogSink `send` signature changed in glog
  0.7). Confirm in Group 3; if not, [R-7] narrows to plugin-host messages and
  the change is recorded. Group 0 finding: the installed glog 0.6.0
  (Ubuntu 24.04, the Linux CI base) declares both `send(..., const
  LogMessageTime&, ...)` and a deprecated `send(..., const std::tm*, ...)`;
  glog 0.7 (vcpkg, Windows CI) has the `LogMessageTime` form. Overriding only
  the `LogMessageTime` overload should cover both; Group 3 confirms on Windows.
- [A-3] Exceptions thrown by backends are all `std::exception`-derived, so a
  `catch (const std::exception&)` plus `catch (...)` at each entry point is
  sufficient to keep them inside the library.
- [Q-1] **Resolved → [D-7].** Concurrent calls on the *same* engine: serialise them with a
  per-engine mutex, or document "one thread at a time per engine"?
  Recommendation: serialise. FFI callers (Unity jobs, Python threads) will
  share handles by accident, and a mutex costs nothing next to inference.
  Decide before Group 3.
- [Q-2] **Resolved → [D-8].** Does `neuriplo_engine_create` load eagerly (current
  `setup_inference_engine` behaviour) with no separate `load` call?
  Recommendation: yes — keep the "constructed == ready" contract.
- [Q-3] **Resolved → [D-9].** C API inside `libneuriplo` ([D-6]) or a separate `libneuriplo_c`?
  Recommendation: inside, per [D-6].
- [Q-4] **Resolved → [D-10].** Unity/C# coverage: a CI job needs the .NET SDK, which is a new CI
  dependency. Recommendation: manual check [M-3] this phase; revisit when a
  Unity package is actually shipped.
- [Q-5] When does `infer_into` come back? Recommendation: with the first
  backend that can write outputs directly into caller memory (likely ONNX
  Runtime I/O binding); record it in the roadmap, not here.

## Definition of Done (requirements level)

- [ ] Every [R-n] implemented or explicitly deferred with a tracked location
- [x] [Q-1] decided and recorded before Group 3 starts ([D-7], 2026-09-26)
- [ ] A plain-C program, a C++ program via the wrapper, and a Python `ctypes`
      script each run inference against the installed package
- [ ] The existing C++ API and `neuriplo-infer`'s build path are unchanged
- [ ] `plugin_abi.h` unchanged; `NEURIPLO_C_API_VERSION` is 1
