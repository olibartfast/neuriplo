# Validation — Stable Consumer C ABI

Written before implementation. Every check traces to a requirement in
`requirements.md`; nothing is marked checked unless the command was actually
run and its result recorded in the evidence log.

## Acceptance command

Owned by the specifier, read-only to anything it scores:

```bash
cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
  && cmake --build build-capi \
  && ctest --test-dir build-capi -R "CApi|PluginAbi" --output-on-failure \
  && ./scripts/abi/check_symbols.sh build-capi \
  && ./scripts/quality/format.sh --check
```

`--fresh` (CMake 3.24+; 3.28 is installed) was added in Group 0. Without it,
re-running the command on an existing tree fails in pre-existing code: on a
re-configure the root `include_directories(... ${gtest_SOURCE_DIR}/include)`
sees the now-cached `gtest_SOURCE_DIR` (set when gmock is built from
`/usr/src/googletest`), gtest lands on a plain `-I`, and
`backends/opencv-dnn/test/OCVDNNInferTest.cpp` fails `-Wsign-compare` under
`-Werror`. A first configure is unaffected (CI always is one). Same rule for
targeted checks: after editing any `CMakeLists.txt`, re-configure with
`--fresh` rather than letting the build re-run CMake. The root cause is
outside this phase and is reported, not fixed, here.

No vendor SDK, no GPU, no network. `-DWERROR=ON` is part of the command on
purpose: Phase 2 passed locally and failed the CI Werror job.

`-R "CApi|PluginAbi"` selects the 33 `CApi.<Group>.<Name>` cases, the 9
`CApiWrapper.<Name>` cases, and the 32 Phase 2 `PluginAbi*` cases. The build
step itself runs the header, layout, and include checks
(`neuriplo_capi_header_check_c`, `neuriplo_capi_header_check_cxx`,
`neuriplo_capi_include_audit`), so a header regression fails the command
before ctest starts. `./scripts/abi/check_symbols.sh` arrives in Group 5;
until then groups score the subset in `plan.md` (per-group filters) and the
command minus that step.

Sanitizer trees used by individual checks (not part of the single command,
which stays fast):

```bash
# ASan + LSan + UBSan ([V-5], and any case on demand)
cmake -S . -B build-capi-asan -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON \
      -DSANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug \
  && cmake --build build-capi-asan \
  && ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
     LSAN_OPTIONS=suppressions=$PWD/scripts/quality/lsan-suppressions.txt \
     ctest --test-dir build-capi-asan -R "^CApi" --output-on-failure

# TSan ([V-8]; clang 18, no suppressions -- [D-17])
CC=clang-18 CXX=clang++-18 cmake -S . -B build-capi-tsan -DDEFAULT_BACKEND=OPENCV_DNN \
      -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug \
      "-DCMAKE_C_FLAGS=-fsanitize=thread -g" "-DCMAKE_CXX_FLAGS=-fsanitize=thread -g" \
      -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread \
      -DCMAKE_MODULE_LINKER_FLAGS=-fsanitize=thread \
  && cmake --build build-capi-tsan \
  && TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-capi-tsan -R "^CApi\.(Thread|Log)\." --output-on-failure
```

## Automated Checks

Case names are the ctest names; `CApi.X.*` means every case of group X.

- [ ] [V-1] → [R-1]: `neuriplo_c.h` compiles as strict C99
      (`C_STANDARD 99`, no extensions, `-Wall -Wextra -Wpedantic -Werror`) and
      as strict C++17, in each case in the same translation unit as
      `plugin_abi.h` (build targets `neuriplo_capi_header_check_c` / `_cxx`);
      the include audit allows it only `<stddef.h>` and `<stdint.h>`.
      `NEURIPLO_C_API_VERSION == 1` and `neuriplo_api_version()` returns it
      (`CApi.Lifecycle.ApiVersion`); status names are non-empty and distinct
      (`CApi.Lifecycle.StatusStrings`).
- [ ] [V-2] → [R-2], [Q-2]/[D-8], [D-11]: create → backend id → destroy on
      `FIXTURE_GOOD` via `plugin_dir`, the fixture instance released on
      destroy (`CApi.Lifecycle.CreateGoodFixture`); conforming config edges
      accepted (`.ConformingConfigVariants`); the compiled-in default backend,
      selected by NULL, "" and its explicit id, is reached and fails with
      `MODEL_LOAD` naming the id and model path — no model is available
      offline, so this is the default-backend check (`.DefaultBackendBadModel`);
      an unknown `backend_id`, and a rejected plugin's id, return
      `BACKEND_NOT_FOUND` with the requested and available ids in
      `neuriplo_last_error()` (`.UnknownBackend`);
      `neuriplo_available_backends` lists the default backend first and the
      loaded fixtures once each, but no rejected fixture (`.AvailableBackends`).
- [ ] [V-3] → [R-2], [D-12]: `struct_size` of 0, 4, one field short, and one
      byte short is `INVALID_ARGUMENT` (`CApi.StructSize.TooSmall`); a larger
      `struct_size` with zeroed trailing fields is accepted
      (`.LargerZeroedTrailing`); a non-zero trailing field is
      `INVALID_ARGUMENT` (`.NonZeroTrailing`). Layout of every struct pinned
      in C ([V-11]).
- [ ] [V-4] → [R-3]: metadata names, shapes, batch size, and dtypes match the
      fixture, `struct_size` set by the library (`CApi.Metadata.Fixture`);
      the same view addresses and contents before and after three inference
      calls, until destroy (`.ViewsStableAcrossInfer`); NULL handle / NULL
      out / out-of-range index are `INVALID_ARGUMENT` with outs nulled
      (`.InvalidArguments`).
- [ ] [V-5] → [R-4], [D-4], [D-13]: inference on `FIXTURE_GOOD` returns
      `[2,4,6,8]` for input `[1,2,3,4]`, dtype FLOAT32, shape `[1,4]`,
      `element_count` 4 (`CApi.Infer.DoublesInput`); a zero-element output
      (`FIXTURE_SCRIPTED` `out_empty`) has count 0, size 0, shape `[0,4]`
      (`.EmptyOutput`); a result stays readable after its engine is destroyed
      (`.ResultOutlivesEngine`); 1000 iterations with every plugin output
      released exactly once (`.Repeated1000`, fixture counters). `CApi.Infer.*`
      clean under ASan/LSan/UBSan in `build-capi-asan`.
- [ ] [V-6] → [R-5], [A-3], [D-11], [D-15]: every failure mode maps to its
      status with a non-empty `neuriplo_last_error()` and nulled outs:
      create failure and all seven metadata rejections → `MODEL_LOAD`
      (`CApi.Error.CreateFailure`, `.MetadataRejected`); NULL config / out /
      model path / input sizes → `INVALID_ARGUMENT` whose message starts with
      the function name (`.NullArguments`); success clears the message
      (`.SuccessClearsLastError`); infer failure → `INFERENCE` carrying the
      plugin's message (`CApi.InferError.BackendFailure`); all eight
      malformed plugin outputs → `INFERENCE`, released exactly once
      (`.MalformedOutputs`); inputs the backend rejects → `INFERENCE` and the
      engine stays usable, while `(NULL, 0)` inputs are not over-rejected
      (`.WrongInputs`); NULL engine / result / out, NULL inputs with
      `n_inputs > 0`, NULL data with non-zero size → `INVALID_ARGUMENT`
      (`.InvalidArguments`). No case ever observes a C++ exception (the suite
      is C and would terminate).
- [ ] [V-7] → [R-5], [D-15]: `neuriplo_last_error()` is per-thread: a fresh
      thread reads ""; four threads failing concurrently (50 times each) each
      read only their own message; the main thread's message is unchanged
      afterwards (`CApi.LastError.PerThread`).
- [ ] [V-8] → [R-6], [Q-1]/[D-7], [D-17]: 4 engines × 4 threads × 250
      inferences, all results correct and every output released
      (`CApi.Thread.EnginesTimesThreads`); 8 threads on one `slow` engine
      never overlap inside the backend (`.SameEngineSerialised`); 8 threads
      concurrently listing, creating, inferring, destroying
      (`.ConcurrentCreateAndList`). Clean under TSan in `build-capi-tsan`,
      no suppressions.
- [ ] [V-9] → [R-7], [A-2], [D-16]: with a callback installed, loader
      rejections arrive at WARNING+ naming the plugin, with the given
      `user_data`, no trailing newline, and stderr still receives them
      (`CApi.Log.CallbackReceivesRejection`); after removal nothing arrives
      although stderr shows messages were produced (`.RemovedCallbackSilent`);
      no callback means glog's stderr output as before
      (`.NoCallbackStderrUnchanged`); `min_level` ERROR filters out warnings
      but delivers a plugin create error, and an undefined level is
      `INVALID_ARGUMENT` (`.MinLevelFilters`).
- [ ] [V-10] → [R-8]: the 9 `CApiWrapper.*` cases exercise version,
      backend listing, create, metadata, infer (vector and pointer inputs),
      empty output, dtype mismatch, error-as-exception with status and
      message for each failure class, move construction and assignment with
      exactly-once release, and result-outlives-engine. The wrapper test
      links no glog. The include audit allows `neuriplo.hpp` only
      `"neuriplo_c.h"` and standard headers; `CApiHeaderCheck.cpp` asserts
      the move-only / nothrow-move traits under strict C++17.
- [ ] [V-11] → [R-9]: `check_symbols.sh` matches `scripts/abi/neuriplo_c.symbols`
      (19 names); deleting one function from the build makes it fail, and so
      does exporting an unlisted `neuriplo_*` symbol. Layout, enum-width,
      enumerator-value, and function-type assertions compile as C99
      (`CApiHeaderCheck.c`, every build).
- [ ] [V-12] → [R-10], [R-11]: install to a temporary prefix;
      `test/consumer/` configures with only `CMAKE_PREFIX_PATH` set to that
      prefix, builds the C and C++ programs, and both print the expected
      output. `pkg-config --cflags --libs neuriplo` works against the prefix.
- [ ] [V-13] → [R-12]: `python3 test/consumer/python/smoke_ctypes.py <prefix>`
      exits 0 after checking the inference result.
- [ ] [V-14] → constraints: default `OPENCV_DNN` build, full `ctest`,
      `run.sh`, `cppcheck.sh`, `gen_backend_docs.py --check` green; the
      existing C++ API headers are unchanged (`git diff origin/develop --
      include/InferenceBackendSetup.hpp include/common.hpp` empty);
      `plugin_abi.h` unchanged.
- [ ] [V-15] → constraints: no new dependency (`git diff origin/develop --
      versions.env docs/backends.yaml`, no new `find_package` beyond what the
      project already requires).

## Manual Checks

- [ ] [M-1] → [R-13]: someone new to the repository follows `docs/C_API.md`
      and gets the C and Python examples running from an installed package.
- [ ] [M-2] → [R-13], [D-5]: the page states the versioning policy and the
      ownership rules clearly enough to write a binding from them alone.
- [ ] [M-3] → [R-13], [Q-4]: the C#/Unity example is run once in a Unity
      project on Windows (plugin folder layout, `DllImport`, log callback,
      inference off the main thread), with the Unity and OS versions recorded.
- [x] [M-4] → [T-5]: the pre-implementation run shows every C API case
      failing against the stubs — a case that passes against a stub is not
      testing anything.
- [ ] [M-5] → ownership: the acceptance files are changed after Group 0 only
      by the specifier, with the reason recorded in `plan.md`.

## Evidence Log

| ID | Command/Check | Pre-implementation | Result | Date | Notes |
|----|---------------|--------------------|--------|------|-------|
| V-1 | build targets `neuriplo_capi_header_check_{c,cxx}` + audit; `ctest -R "^CApi\.Lifecycle\.(ApiVersion\|StatusStrings)$"` | build: pass (header is Group 0's); ctest: 2/2 fail | | 2026-09-26 | build-time part passes by construction |
| V-2 | `ctest -R "^CApi\.Lifecycle\."` | 7/7 fail | | 2026-09-26 | |
| V-3 | `ctest -R "^CApi\.StructSize\."` | 3/3 fail | | 2026-09-26 | |
| V-4 | `ctest -R "^CApi\.Metadata\."` | 3/3 fail | | 2026-09-26 | |
| V-5 | `ctest -R "^CApi\.Infer\."` + ASan/LSan tree | 4/4 fail | | 2026-09-26 | |
| V-6 | `ctest -R "^CApi\.(Error\|InferError)\."` | 8/8 fail | | 2026-09-26 | |
| V-7 | `ctest -R "^CApi\.LastError\."` | 1/1 fail | | 2026-09-26 | |
| V-8 | `ctest -R "^CApi\.Thread\."` in TSan tree | 3/3 fail (gcc, clang, TSan) | | 2026-09-26 | TSan tree builds and runs; failures are stub statuses, no TSan report |
| V-9 | `ctest -R "^CApi\.Log\."` | 4/4 fail | | 2026-09-26 | |
| V-10 | `ctest -R "^CApiWrapper\."` + include audit | 9/9 fail | | 2026-09-26 | skeleton throws `UNIMPLEMENTED` |
| V-11 | `check_symbols.sh` + negative; `CApiHeaderCheck.c` | list committed (19); script is Group 5 | | 2026-09-26 | |
| V-12 | install + `test/consumer` + pkg-config | | | | |
| V-13 | `smoke_ctypes.py` | | | | |
| V-14 | default path + quality gates | | | | |
| V-15 | dependency diff | | | | |
| M-1 | docs walkthrough | | | | |
| M-2 | contract read | | | | |
| M-3 | Unity run | | | | |
| M-4 | pre-implementation capture | 42/42 `CApi*` fail; 32/32 `PluginAbi*` pass | pass | 2026-09-26 | see Pre-implementation run below |
| M-5 | acceptance ownership | | | | |

## Pre-implementation run ([M-4], 2026-09-26)

Tree: `feature/consumer-c-abi` working tree after Group 0 (stubs in
`src/neuriplo_c.cpp`, wrapper skeleton in `neuriplo.hpp`). Command: the
acceptance command without `check_symbols.sh` (Group 5):

```bash
cmake --fresh -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
  && cmake --build build-capi \
  && ctest --test-dir build-capi -R "CApi|PluginAbi" --output-on-failure \
  && ./scripts/quality/format.sh --check
```

- Configure and build: clean under gcc 13.3 with `-DWERROR=ON`, including
  the three build-time check targets. The same with clang 18.1
  (`build-capi-clang`): clean.
- ctest: **42 of 42 C API cases fail** (33 `CApi.*`, 9 `CApiWrapper.*`), each
  by a failed check against a stub status (`UNIMPLEMENTED`), an inert value
  (`neuriplo_api_version() == 0`, empty status strings), or the wrapper's
  `UNIMPLEMENTED` exception — no crash, abort, or sanitizer report. **32 of
  32 `PluginAbi*` cases pass** with the amended fixture. Result: "43% tests
  passed, 42 tests failed out of 74"; identical under clang.
- No case passed against a stub, so none needed rework.
- Remaining acceptance steps: `format.sh --check` passes;
  `cppcheck.sh` passes; `check_includes.py` reports no missing includes; all
  35 non-`CApi` tests in `build-capi` pass.
- TSan tree (`build-capi-tsan`, clang 18): builds; `CApi.Thread.*` fail on
  stub statuses with no TSan report.
- Found while re-running: a plain re-configure of an existing `-DWERROR=ON`
  tree breaks `OCVDNNInferTest.cpp` (pre-existing; see the note under the
  acceptance command). Both trees were then rebuilt from scratch and the
  results above re-confirmed; the command now uses `cmake --fresh`.
- Negative control for the include audit: a header including
  `"../InferenceBackendSetup.hpp"` and `<glog/logging.h>`, and a C header
  including `<stdlib.h>`, are both rejected.

## Deviations

- Record here any criterion not fully met, why, and how it is tracked.

## Definition of Done (integration)

- [ ] Every automated and manual check executed, with evidence and dates
- [ ] Spec, code, docs, changelog, and roadmap agree
- [ ] Roadmap Phase 7 status updated only after this section is satisfied
