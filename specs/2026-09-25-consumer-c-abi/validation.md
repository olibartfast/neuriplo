# Validation — Stable Consumer C ABI

Written before implementation. Every check traces to a requirement in
`requirements.md`; nothing is marked checked unless the command was actually
run and its result recorded in the evidence log.

## Acceptance command

Owned by the specifier, read-only to anything it scores:

```bash
cmake -S . -B build-capi -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON -DWERROR=ON \
  && cmake --build build-capi \
  && ctest --test-dir build-capi -R "CApi|PluginAbi" --output-on-failure \
  && ./scripts/abi/check_symbols.sh build-capi \
  && ./scripts/quality/format.sh --check
```

No vendor SDK, no GPU, no network. `-DWERROR=ON` is part of the command on
purpose: Phase 2 passed locally and failed the CI Werror job.

## Automated Checks

- [ ] [V-1] → [R-1]: `neuriplo_c.h` compiles standalone as C99
      (`-std=c99 -Wall -Wextra -Wpedantic -Werror`) and as C++17, with only
      `include/neuriplo` on the include path. `NEURIPLO_C_API_VERSION == 1`
      and `neuriplo_api_version()` returns it.
- [ ] [V-2] → [R-2], [Q-2]: create → metadata → destroy on `FIXTURE_GOOD` via
      `plugin_dir`; create on the compiled-in default backend; unknown
      `backend_id` returns `NEURIPLO_STATUS_BACKEND_NOT_FOUND` with the
      available ids in `neuriplo_last_error()`;
      `neuriplo_available_backends` lists the default backend and loaded
      fixtures but no rejected fixture.
- [ ] [V-3] → [R-2]: `struct_size` smaller than the v1 config is
      `INVALID_ARGUMENT`; a larger `struct_size` with trailing zeroed fields is
      accepted (forward-compatible caller).
- [ ] [V-4] → [R-3]: metadata names, shapes, batch size, and dtypes match the
      fixture; views stay valid across later inference calls until destroy.
- [ ] [V-5] → [R-4]: inference on `FIXTURE_GOOD` returns `[2,4,6,8]` for
      input `[1,2,3,4]`, dtype FP32, shape `[1,4]`; a zero-element output
      (`FIXTURE_SCRIPTED` `out_empty`) has a count of 0 and a NULL-safe view.
      Result released exactly once; ASan/LSan clean over 1000 iterations.
- [ ] [V-6] → [R-5], [A-3]: every failure mode maps to its status with a
      non-empty `neuriplo_last_error()`: create failure, metadata rejection,
      infer failure, malformed plugin outputs (all via `FIXTURE_SCRIPTED`),
      NULL handle, NULL out-pointer, NULL input data with non-zero size. No
      test ever observes a C++ exception (the C suite is compiled as C and
      would terminate).
- [ ] [V-7] → [R-5]: `neuriplo_last_error()` is per-thread: a failure on
      thread A does not change the message thread B reads.
- [ ] [V-8] → [R-6], [Q-1]: 4 engines × 4 threads each running inference for a
      fixed count; all results correct; TSan clean.
- [ ] [V-9] → [R-7], [A-2]: with a callback installed, a plugin rejection
      message arrives with severity WARNING or ERROR; after removal, nothing
      arrives; no callback means unchanged stderr behaviour.
- [ ] [V-10] → [R-8]: the wrapper test uses only `neuriplo.hpp` and exercises
      create, metadata, infer, error-as-exception, and move semantics;
      `neuriplo.hpp` includes no neuriplo header other than `neuriplo_c.h`.
- [ ] [V-11] → [R-9]: `check_symbols.sh` matches the committed list; deleting
      one function from the build makes it fail. Layout assertions compile as
      C.
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
- [ ] [M-4] → [T-5]: the pre-implementation run shows every C API case
      failing against the stubs — a case that passes against a stub is not
      testing anything.
- [ ] [M-5] → ownership: the acceptance files are changed after Group 0 only
      by the specifier, with the reason recorded in `plan.md`.

## Evidence Log

| ID | Command/Check | Pre-implementation | Result | Date | Notes |
|----|---------------|--------------------|--------|------|-------|
| V-1 | header standalone C99/C++17 | | | | |
| V-2 | `ctest -R CApi.*Lifecycle` | | | | |
| V-3 | `ctest -R CApi.*StructSize` | | | | |
| V-4 | `ctest -R CApi.*Metadata` | | | | |
| V-5 | `ctest -R CApi.*Infer` + LSan | | | | |
| V-6 | `ctest -R CApi.*Error` | | | | |
| V-7 | `ctest -R CApi.*LastError` | | | | |
| V-8 | `ctest -R CApi.*Thread` under TSan | | | | |
| V-9 | `ctest -R CApi.*Log` | | | | |
| V-10 | `ctest -R CApiWrapper` | | | | |
| V-11 | `check_symbols.sh` + negative | | | | |
| V-12 | install + `test/consumer` + pkg-config | | | | |
| V-13 | `smoke_ctypes.py` | | | | |
| V-14 | default path + quality gates | | | | |
| V-15 | dependency diff | | | | |
| M-1 | docs walkthrough | | | | |
| M-2 | contract read | | | | |
| M-3 | Unity run | | | | |
| M-4 | pre-implementation capture | | | | |
| M-5 | acceptance ownership | | | | |

## Deviations

- Record here any criterion not fully met, why, and how it is tracked.

## Definition of Done (integration)

- [ ] Every automated and manual check executed, with evidence and dates
- [ ] Spec, code, docs, changelog, and roadmap agree
- [ ] Roadmap Phase 7 status updated only after this section is satisfied
