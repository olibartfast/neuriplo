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

## Group 1 — Lifecycle, versioning, errors ([R-1], [R-2], [R-5])

- [T-6] `neuriplo_api_version`, `neuriplo_engine_create` /
  `neuriplo_engine_destroy` over `setup_inference_engine`,
  `neuriplo_available_backends`.
- [T-7] The error model: a single entry-point guard that catches every
  exception and maps it to a status (`ModelLoadException` →
  `NEURIPLO_STATUS_MODEL_LOAD`, `InferenceExecutionException` →
  `NEURIPLO_STATUS_INFERENCE`, `std::bad_alloc` →
  `NEURIPLO_STATUS_OUT_OF_MEMORY`, anything else → `NEURIPLO_STATUS_INTERNAL`)
  and stores the message thread-locally. `struct_size` handling: reject
  smaller-than-v1 structs, ignore trailing fields from newer callers.
  - Checks: `ctest -R "CApi.*Lifecycle|CApi.*Error"`.

## Group 2 — Metadata and inference ([R-3], [R-4])

- [T-8] Metadata views built once at engine creation and owned by the engine.
- [T-9] `neuriplo_infer` returning a result handle that owns the moved
  `std::vector<RawOutputTensor>`; accessors for count, dtype, shape, data;
  `neuriplo_result_release`.
  - Checks: `ctest -R "CApi.*Metadata|CApi.*Infer"`; ASan/LSan clean.

## Group 3 — Threading and logging ([R-6], [R-7])

- [T-10] Per-[Q-1] same-engine policy; concurrency test with several engines
  and several threads per engine.
- [T-11] Log callback via a glog `LogSink` shim ([A-2]), installed and removed
  thread-safely.
  - Checks: `ctest -R "CApi.*Thread|CApi.*Log"` clean under TSan.

## Group 4 — C++ wrapper ([R-8])

- [T-12] `include/neuriplo/neuriplo.hpp`: `Engine`, `Result`, `Error`,
  `TensorView`, `backends()`; move-only RAII; no neuriplo C++ headers
  included.
  - Checks: `ctest -R CApiWrapper`.

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

## Group 6 — Foreign-language proof and documentation ([R-12], [R-13])

- [T-16] `test/consumer/python/smoke_ctypes.py` against the installed library
  and the good fixture plugin.
- [T-17] `docs/C_API.md`: direction diagram, lifecycle, ownership, errors,
  threading, logging, versioning, examples for C, C++, Python, C#/Unity (Unity
  plugin folder layout, `DllImport`, keeping the log delegate alive, running
  inference off the main thread). Link from `Readme.md` and
  `docs/PLUGIN_BACKENDS.md`.
- [T-18] Execute everything in `validation.md` and record evidence with
  dates; `CHANGELOG.md` under `[Unreleased]`; roadmap Phase 7 status updated
  only once the evidence is in.

## Notes

- Delegation, routing, permissions, and the acceptance command are in
  `orchestration.md`. Groups 1, 2, 4, and 5 are delegable; Group 0 is
  specifier-owned; Group 3 is delegable only after [Q-1] is decided; Group 6's
  documentation is judgement work.
- Record in this section anything that deviated from the plan and why.
