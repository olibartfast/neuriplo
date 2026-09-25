# Validation — Plugin ABI and Loader Hardening

Written before implementation. Every check below traces to a requirement in
`requirements.md`; nothing is marked checked unless the command was actually
run and its result recorded in the evidence log.

## Acceptance command

One command settles whether a group is done. It is owned by the specifier and
is read-only to anything it scores ([D-6], and see `orchestration.md`):

```bash
cmake -S . -B build-plugin -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON \
  && cmake --build build-plugin \
  && ctest --test-dir build-plugin -R "PluginAbi|PluginLoader" --output-on-failure \
  && ./scripts/quality/format.sh --check
```

No vendor SDK, no GPU, no network. If this needs anything else installed,
[R-9] has failed.

## Automated Checks

```bash
# Sanitizer pass — the tool that actually catches R-1/R-2 regressions
./scripts/quality/sanitizers.sh
# or: cmake -S . -B build-asan -DSANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug \
#       -DBUILD_INFERENCE_ENGINE_TESTS=ON && ctest --test-dir build-asan -R PluginAbi
```

- [ ] [V-1] → [R-9]: the acceptance command above passes in a tree with no
      backend SDK installed and `NEURIPLO_PLUGIN_BACKENDS` unset. Evidence
      records that the `PluginAbi` tests actually ran — a suite that silently
      registered zero tests is a failure, not a pass.
- [ ] [V-2] → [R-6], [D-2]: every fixture in [R-6] exists, builds from plain C
      against `plugin_abi.h`, links no vendor library, and is named so the
      loader's directory scan finds it. Asserted:
      `ldd` on each fixture shows no backend SDK.
- [ ] [V-3] → [R-1]: each malformed-metadata fixture (null `shape` with
      `ndim > 0`, out-of-bound `ndim`, null `name`) is rejected with
      `ModelLoadException`, and the message names the plugin, the layer index,
      and the bad field. Asserted on message content, not just the exception
      type. Clean under ASan and UBSan — no read of the bad pointer.
- [ ] [V-4] → [R-2]: each malformed-output fixture is rejected rather than
      dereferenced: success return with null `tensors`, null `data`, null
      `shape` with `ndim > 0`, and `size_bytes` inconsistent with
      dtype × Π shape per [Q-3]. Clean under ASan and UBSan.
- [ ] [V-5] → [R-3]: **release ownership.** For every fixture whose `infer`
      succeeds, `release_outputs` is called exactly once — including when the
      host then rejects the outputs under [V-4], and when the copy throws. The
      fixture counts its own `release_outputs` calls and the test asserts the
      count; ASan with `detect_leaks=1` is the backstop, not the primary
      assertion. A count of 0 or 2 fails.
- [ ] [V-6] → [R-4], [Q-2]: a fixture returning a dtype outside
      `neuriplo_dtype_t` is rejected per [Q-2]'s decision, and the two code
      paths agree — no silent `Float32` substitution in metadata and no silently
      empty element vector in `to_elements`.
- [ ] [V-7] → [R-5]: loading plugins on one thread while
      `find_plugin_backend` / `get_plugin_backends` run on another is clean
      under TSan where available, ASan otherwise. Asserted: no crash, no
      torn read, and every successfully loaded id is findable afterwards.
- [ ] [V-8] → [R-7]: with one conforming and several broken fixtures in the same
      directory, the conforming plugin loads, serves inference, and returns the
      expected tensor; `load_backend_plugins` returns the count of *accepted*
      plugins only; and the compiled-in `OPENCV_DNN` path is unaffected in the
      same process.
- [ ] [V-9] → [R-6] load-time half, [D-3]: the six existing load-time
      rejections each have a test — unopenable library, missing entry symbol,
      entry returning null, ABI mismatch, incomplete api table, duplicate
      `backend_id` — and a rejected plugin leaves the descriptor list unchanged.
- [ ] [V-10] → [D-4]: `plugin_abi.h` is unmodified and
      `NEURIPLO_PLUGIN_ABI_VERSION` is still 2.
      `git diff origin/develop -- include/neuriplo/plugin_abi.h` → expected:
      empty. A conforming v2 plugin built before this change still loads.
- [ ] [V-11] → constraints: the default path is untouched —
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN
      -DBUILD_INFERENCE_ENGINE_TESTS=ON`, build, full `ctest` green, plus
      `./scripts/quality/run.sh` and `python3 scripts/gen_backend_docs.py
      --check` clean.
- [ ] [V-12] → constraints: no new dependency. `git diff origin/develop --
      versions.env docs/backends.yaml cmake/` introduces no SDK or package
      requirement, and the fixtures add no `find_package` for anything not
      already required.

## Manual Checks

- [ ] [M-1] → [R-8]: a reader who knows nothing about this repository can
      follow `docs/PLUGIN_BACKENDS.md` to build, package, deploy, and load one
      plugin, without relying on unwritten environment behavior. Read it as that
      newcomer would, on a clean machine if possible.
- [ ] [M-2] → [R-8], [D-4]: the page states the host-side validation contract —
      what is rejected and with what diagnostic — clearly enough that a plugin
      author can write a conforming plugin from it alone, and states that
      host-side hardening is not an ABI break.
- [ ] [M-3] → [A-2], [T-5]: the pre-hardening run was captured before any guard
      was written, showing the call-time cases failing. Without this, the suite
      might be asserting nothing — a test that passes before *and* after the fix
      is not evidence.
- [ ] [M-4] → [D-6]: no commit in this branch modifies
      `backends/src/test/PluginAbiContractTest.cpp` or the fixtures after
      Group 0, except by the specifier and with the reason recorded here.
      `git log --oneline -- backends/src/test/PluginAbiContractTest.cpp`
      reviewed against `orchestration.md`'s ownership rule.
- [ ] [M-5] → [Q-1]: `force_gpu`'s silent device override is recorded for
      Phase 3/4 and has not been quietly fixed or quietly forgotten here.

## Evidence Log

| ID | Command/Check | Pre-hardening | Result | Date | Notes |
|----|---------------|---------------|--------|------|-------|
| V-1 | acceptance command, no SDK | 31 tests registered; 13 pass, 18 fail/crash (expected pre-hardening) | | | tests registered: |
| V-2 | fixture build + `ldd` | 7 fixtures, plain C, each links only libc/ld-linux/vdso | pass | 2026-09-25 | no vendor SDK |
| V-3 | `ctest -R PluginAbi.*Metadata` | 5 of 6 malformed cases SEGFAULT/bus error or fail; QueryFailure fails on missing plugin path | | | |
| V-4 | `ctest -R PluginAbi.*Output` | null tensors/data/shape/ndim SEGFAULT; size short/long/negative dim silently accepted | | | |
| V-5 | release-count assertions + LSan | release count equal on conforming path; rejection paths untestable (crash first) | | | observed counts: |
| V-6 | `ctest -R PluginAbi.*Dtype` | unknown dtype accepted in outputs, metadata, and legacy view | | | |
| V-7 | concurrency test under TSan/ASan | LoadWhileLookingUp: torn reads (empty ids) with no sanitizer; pointer-stability SEGFAULT | | | |
| V-8 | `ctest -R PluginAbi.*Isolation` | pass (isolation already holds when bad plugins fail at load) | | | |
| V-9 | `ctest -R PluginAbi.*Load` | pass — all six load-time rejections, path + reason in diagnostic | | | |
| V-10 | `git diff` on `plugin_abi.h` | | | | |
| V-11 | default path + quality gates | | | | |
| V-12 | dependency diff | | | | |
| M-1 | docs walkthrough | | | | |
| M-2 | validation contract read | | | | |
| M-3 | pre-hardening capture | captured 2026-09-25 against 4049980 + Group 0 | | | |
| M-4 | acceptance-suite ownership | | | | |
| M-5 | Q-1 deferral recorded | | | | |

The **Pre-hardening** column is filled during [T-5], before any guard exists.
A row where pre-hardening and post-hardening both pass means the check is not
testing what it claims ([M-3]).

## Deviations

- Record here any criterion not fully met, why, and how it is tracked.
- [V-5] "when the copy throws": no fixture can force a throw inside the copy
  once [D-8] bounds it, so this half is verified by review of the RAII guard
  (release on every exit path, including unwinding), not by a test.

## Definition of Done (integration)

- [ ] Every automated and manual check executed, with evidence and dates
- [ ] Evidence traced to [R-n]; deviations documented
- [ ] Spec, code, docs, changelog, and roadmap agree — merged as one coherent
      change into `develop`
- [ ] Durable discoveries propagated to `specs/tech-stack.md` if the plugin
      boundary rules themselves need restating
- [ ] Roadmap Phase 2 moved to Complete only after this section is satisfied
