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
# ASan + LSan + UBSan -- [V-5] evidence is exactly this filter (11 cases)
cmake --fresh -S . -B build-capi-asan -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON \
      -DSANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug \
  && cmake --build build-capi-asan --target CApiContractTest \
  && ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
     LSAN_OPTIONS=suppressions=$PWD/scripts/quality/lsan-suppressions.txt \
     ctest --test-dir build-capi-asan -R "^CApi\.(Metadata|Infer|InferError)\." --output-on-failure

# TSan ([V-8], [V-9]; clang 18, no suppressions -- [D-17])
CC=clang-18 CXX=clang++-18 cmake --fresh -S . -B build-capi-tsan -DDEFAULT_BACKEND=OPENCV_DNN \
      -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug \
      "-DCMAKE_C_FLAGS=-fsanitize=thread -g" "-DCMAKE_CXX_FLAGS=-fsanitize=thread -g" \
      -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread \
      -DCMAKE_MODULE_LINKER_FLAGS=-fsanitize=thread \
  && cmake --build build-capi-tsan \
  && TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-capi-tsan -R "^CApi\.(Thread|Log)\." --output-on-failure
```

The [V-5] filter is deliberately narrow: it covers every case that creates
metadata views and results, and nothing else. Wider ASan runs are not [V-5]
evidence: `CApiWrapper.*` is [V-10]'s, `Thread`/`Log` are TSan's, and the
Group 1 cases load the OpenCV DNN backend, whose leaks are not this phase's
to judge. In CI the existing `sanitizers` job already runs the full `ctest`
under ASan/LSan/UBSan for five backends ([D-21]); a `CApi*` failure there is
still a finding, just not [V-5]'s evidence.

## Automated Checks

Case names are the ctest names; `CApi.X.*` means every case of group X.

- [x] [V-1] → [R-1]: `neuriplo_c.h` compiles as strict C99
      (`C_STANDARD 99`, no extensions, `-Wall -Wextra -Wpedantic -Werror`) and
      as strict C++17, in each case in the same translation unit as
      `plugin_abi.h` (build targets `neuriplo_capi_header_check_c` / `_cxx`);
      the include audit allows it only `<stddef.h>` and `<stdint.h>`.
      `NEURIPLO_C_API_VERSION == 1` and `neuriplo_api_version()` returns it
      (`CApi.Lifecycle.ApiVersion`); status names are non-empty and distinct
      (`CApi.Lifecycle.StatusStrings`).
- [x] [V-2] → [R-2], [Q-2]/[D-8], [D-11]: create → backend id → destroy on
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
- [x] [V-3] → [R-2], [D-12]: `struct_size` of 0, 4, one field short, and one
      byte short is `INVALID_ARGUMENT` (`CApi.StructSize.TooSmall`); a larger
      `struct_size` with zeroed trailing fields is accepted
      (`.LargerZeroedTrailing`); a non-zero trailing field is
      `INVALID_ARGUMENT` (`.NonZeroTrailing`). Layout of every struct pinned
      in C ([V-11]).
- [x] [V-4] → [R-3]: metadata names, shapes, batch size, and dtypes match the
      fixture, `struct_size` set by the library (`CApi.Metadata.Fixture`);
      the same view addresses and contents before and after three inference
      calls, until destroy (`.ViewsStableAcrossInfer`); NULL handle / NULL
      out / out-of-range index are `INVALID_ARGUMENT` with outs nulled
      (`.InvalidArguments`).
- [x] [V-5] → [R-4], [D-4], [D-13]: inference on `FIXTURE_GOOD` returns
      `[2,4,6,8]` for input `[1,2,3,4]`, dtype FLOAT32, shape `[1,4]`,
      `element_count` 4 (`CApi.Infer.DoublesInput`); a zero-element output
      (`FIXTURE_SCRIPTED` `out_empty`) has count 0, size 0, shape `[0,4]`
      (`.EmptyOutput`); a result stays readable after its engine is destroyed
      (`.ResultOutlivesEngine`); 1000 iterations with every plugin output
      released exactly once (`.Repeated1000`, fixture counters). Clean under
      ASan/LSan/UBSan: `ctest --test-dir build-capi-asan -R
      "^CApi\.(Metadata|Infer|InferError)\."`, 11/11, no sanitizer report.
      The input copy of [R-4] is the only copy; outputs are moved (reviewer
      obligation on the Group 2 diff).
- [x] [V-6] → [R-5], [A-3], [D-11], [D-15]: every failure mode maps to its
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
- [x] [V-7] → [R-5], [D-15]: `neuriplo_last_error()` is per-thread: a fresh
      thread reads ""; four threads failing concurrently (50 times each) each
      read only their own message; the main thread's message is unchanged
      afterwards (`CApi.LastError.PerThread`).
- [x] [V-8] → [R-6], [Q-1]/[D-7], [D-17]: 4 engines × 4 threads × 250
      inferences, all results correct and every output released
      (`CApi.Thread.EnginesTimesThreads`); 8 threads on one `slow` engine
      never overlap inside the backend (`.SameEngineSerialised`); 8 threads
      concurrently listing, creating, inferring, destroying
      (`.ConcurrentCreateAndList`). Clean under TSan in `build-capi-tsan`,
      no suppressions.
- [x] [V-9] → [R-7], [A-2], [D-16]: with a callback installed, loader
      rejections arrive at WARNING+ naming the plugin, with the given
      `user_data`, no trailing newline, and stderr still receives them
      (`CApi.Log.CallbackReceivesRejection`); after removal nothing arrives
      although stderr shows messages were produced (`.RemovedCallbackSilent`);
      no callback means glog's stderr output as before
      (`.NoCallbackStderrUnchanged`); `min_level` ERROR filters out warnings
      but delivers a plugin create error, and an undefined level is
      `INVALID_ARGUMENT` (`.MinLevelFilters`).
- [x] [V-10] → [R-8]: the 9 `CApiWrapper.*` cases exercise version,
      backend listing, create, metadata, infer (vector and pointer inputs),
      empty output, dtype mismatch, error-as-exception with status and
      message for each failure class, move construction and assignment with
      exactly-once release, and result-outlives-engine. The wrapper test
      links no glog. The include audit allows `neuriplo.hpp` only
      `"neuriplo_c.h"` and standard headers; `CApiHeaderCheck.cpp` asserts
      the move-only / nothrow-move traits under strict C++17.
- [x] [V-11] → [R-9]: `scripts/abi/check_symbols.sh <build-dir> [<symbols-file>]`
      reports `symbols: OK (19)` against `scripts/abi/neuriplo_c.symbols`. It
      compares the symmetric difference, so both directions of drift fail,
      shown without editing sources by passing an edited list: a list without
      `neuriplo_infer` fails with `neuriplo_infer` exported-but-unlisted (the
      case "a new or renamed export"), and a list with an extra
      `neuriplo_not_exported` fails with it listed-but-missing (the case "a
      function deleted from the build" — a removed definition is exactly a
      listed symbol the library no longer exports). Layout, enum-width,
      enumerator-value, and function-type assertions compile as C99
      (`CApiHeaderCheck.c`, every build).
- [x] [V-12] → [R-10], [R-11]: install to a temporary prefix;
      `test/consumer/` configures with only `CMAKE_PREFIX_PATH` set to that
      prefix, builds the C and C++ programs, and both print the expected
      output. `pkg-config --cflags --libs neuriplo` works against the prefix.
      Neither program nor `test/consumer/CMakeLists.txt` references glog
      ([D-19]).
- [x] [V-13] → [R-12]: `python3 test/consumer/python/smoke_ctypes.py <prefix>`
      exits 0 after checking the inference result.
- [x] [V-14] → constraints: default `OPENCV_DNN` build, full `ctest`,
      `run.sh`, `cppcheck.sh`, `gen_backend_docs.py --check` green; the
      existing C++ API headers are unchanged (`git diff origin/develop --
      include/InferenceBackendSetup.hpp include/common.hpp` empty);
      `plugin_abi.h` unchanged.
- [x] [V-15] → constraints: no new dependency (`git diff origin/develop --
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
- [x] [M-5] → ownership: the acceptance files are changed after Group 0 only
      by the specifier, with the reason recorded in `plan.md`.

## Evidence Log

| ID | Command/Check | Pre-implementation | Result | Date | Notes |
|----|---------------|--------------------|--------|------|-------|
| V-1 | build targets `neuriplo_capi_header_check_{c,cxx}` + audit; `ctest -R "^CApi\.Lifecycle\.(ApiVersion\|StatusStrings)$"` | build: pass (header is Group 0's); ctest: 2/2 fail | pass: builds clean gcc 13.3 + clang 18.1 `-DWERROR=ON`; 2/2 | 2026-09-27 | |
| V-2 | `ctest -R "^CApi\.Lifecycle\."` | 7/7 fail | pass 7/7 (gcc, clang) | 2026-09-27 | |
| V-3 | `ctest -R "^CApi\.StructSize\."` | 3/3 fail | pass 3/3 (gcc, clang) | 2026-09-27 | |
| V-4 | `ctest -R "^CApi\.Metadata\."` | 3/3 fail | pass 3/3 (gcc, clang) | 2026-09-27 | |
| V-5 | `ctest -R "^CApi\.Infer\."`; ASan: `-R "^CApi\.(Metadata\|Infer\|InferError)\."` in `build-capi-asan` | 4/4 fail | pass 4/4; ASan/LSan/UBSan 11/11, no report | 2026-09-27 | single input copy / moved outputs: `neuriplo_infer` in `src/neuriplo_c.cpp` |
| V-6 | `ctest -R "^CApi\.(Error\|InferError)\."` | 8/8 fail | pass 8/8 (gcc, clang) | 2026-09-27 | |
| V-7 | `ctest -R "^CApi\.LastError\."` | 1/1 fail | pass 1/1 (gcc, clang) | 2026-09-27 | |
| V-8 | `ctest -R "^CApi\.Thread\."` in TSan tree | 3/3 fail (gcc, clang, TSan) | pass 3/3 TSan (clang 18, no suppressions, `halt_on_error=1`), gcc, clang | 2026-09-27 | |
| V-9 | `ctest -R "^CApi\.Log\."` | 4/4 fail | pass 4/4 TSan, gcc, clang | 2026-09-27 | |
| V-10 | `ctest -R "^CApiWrapper\."` + include audit | 9/9 fail | pass 9/9 gcc, clang, and ASan tree; audit + `CApiHeaderCheck.cpp` build clean | 2026-09-27 | unsupported `data_as<double>` rejected by `static_assert` (negative compile checked by hand) |
| V-11 | `check_symbols.sh` + negative; `CApiHeaderCheck.c` | list committed (19); script is Group 5 | `symbols: OK (19)` (gcc and clang trees); list without `neuriplo_infer` → exit 1 naming it unlisted; list plus `neuriplo_not_exported` → exit 1 naming it missing | 2026-09-27 | |
| V-12 | install + `test/consumer` + pkg-config | | pass: `consumer check: PASS`; cmake C, cmake C++, pkg-config C each print `OK FIXTURE_GOOD 2 4 6 8`; no glog reference under `test/consumer/` | 2026-09-27 | configured with only `CMAKE_PREFIX_PATH` |
| V-13 | `smoke_ctypes.py` | | pass: `OK FIXTURE_GOOD 2 4 6 8` (run by `run.sh` against the same prefix) | 2026-09-27 | takes `<prefix> <plugin_dir>`; see Deviations |
| V-14 | default path + quality gates | | pass: full `ctest` in `build-capi` 77/77; `scripts/quality/run.sh` pass; `cppcheck.sh` pass; `gen_backend_docs.py --check` pass; `git diff origin/develop` of `InferenceBackendSetup.hpp`, `common.hpp`, `plugin_abi.h` empty | 2026-09-27 | |
| V-15 | dependency diff | | pass: `versions.env`, `docs/backends.yaml` unchanged; no `find_package` added | 2026-09-27 | the package config has no `find_dependency` ([D-19]) |
| M-1 | docs walkthrough | | open | 2026-09-27 | author run only: the page's C example compiled with `-std=c99 -Wpedantic -Werror` and printed `2 4 6 8`; needs a newcomer run |
| M-2 | contract read | | open | 2026-09-27 | the ctypes binding was written from the header's declarations; an independent read is still needed |
| M-3 | Unity run | | open | | no Windows/Unity environment here; `docs/C_API.md` carries the example |
| M-4 | pre-implementation capture | 42/42 `CApi*` fail; 32/32 `PluginAbi*` pass | pass | 2026-09-26 | see Pre-implementation run below |
| M-5 | acceptance ownership | | pass | 2026-09-27 | `git log 86098cb..HEAD` over every read-only acceptance file is empty; wrapper public declarations unchanged |

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

## Implementation run (2026-09-27, Groups 3–6)

Tree: `feature/consumer-c-abi` after Group 5 and [T-16] (`27f7d7e`), then the
docs commit. The acceptance command above, run verbatim: configure and build
clean under gcc 13.3 `-DWERROR=ON`; ctest "100% tests passed, 0 tests failed
out of 74"; `symbols: OK (19)`; format check passed. Also:

- clang 18.1 `-DWERROR=ON`, fresh tree: builds with no diagnostics; 74/74;
  `symbols: OK (19)`.
- TSan (`build-capi-tsan`, [V-8]/[V-9] command): 7/7, no report.
- ASan/LSan/UBSan (`build-capi-asan`, [V-5] command): 11/11, no report;
  `CApiWrapper.*` also 9/9 there.
- `test/consumer/run.sh build-capi`: `consumer check: PASS` (four consumers).
- CI wiring ([T-19]): `act push --job capi-consumer|capi-tsan --dryrun` both
  succeed. Full local `act` runs: `capi-consumer` succeeded (74/74,
  `symbols: OK (19)`, both negatives reported, `consumer check: PASS`).
  `capi-tsan` first failed before any test ran: under Docker's default
  seccomp profile, TSan could not re-exec with `personality(ADDR_NO_RANDOMIZE)`.
  With `--security-opt seccomp=unconfined` on that container it succeeded,
  7/7 with no report.

## Deviations

- [V-13]: `smoke_ctypes.py` takes `<prefix> <plugin_dir>`, not `<prefix>`
  alone. The fixture plugins live in the build tree, not the install prefix,
  and the script must not guess a build directory. `test/consumer/run.sh`
  passes both and runs it after the C/C++ consumers, so the `capi-consumer`
  CI job covers [V-12] and [V-13] with one call.
- [M-1], [M-2], [M-3] remain open (manual; see the evidence log). Phase 7
  stays short of Complete until they are run.

## Definition of Done (integration)

- [ ] Every automated and manual check executed, with evidence and dates
- [ ] Spec, code, docs, changelog, and roadmap agree
- [ ] Roadmap Phase 7 status updated only after this section is satisfied
