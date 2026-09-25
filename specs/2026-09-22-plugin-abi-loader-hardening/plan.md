# Plan — Plugin ABI and Loader Hardening

Thin groups, each ending in something runnable and reviewable. Order is
walk-before-run: the harness and the evidence of current behavior come first,
then one guard at a time, each landing with the test that proves it.

`develop` stays green throughout: every group leaves the default
`OPENCV_DNN` configure/build/test path passing.

## Group 0 — Fixture harness and acceptance suite (specifier-owned)

The one group that must not be delegated: it creates the suite everything else
is scored against ([D-6]).

- [T-1] Decide [Q-2] (unknown dtype: reject call vs drop tensor) and [Q-3]
  (size/shape strict equality vs lower bound) and record them as decisions in
  `requirements.md`. Blocks Group 3.
- [T-2] Add `backends/src/test/plugin_fixtures/` — first-party, dependency-free
  malformed plugins per [R-6], one translation unit each, plain C against
  `include/neuriplo/plugin_abi.h`, built into the build tree with the
  `libneuriplo_backend_*` naming the loader's directory scan requires.
  One conforming "good" fixture is part of this set: [R-7] needs something that
  still works, and the positive tests should not depend on a vendor SDK either.
- [T-3] Relax the `NEURIPLO_PLUGIN_BACKENDS` gating in
  `backends/src/test/CMakeLists.txt` so plugin tests build from the fixtures
  alone ([R-9]). The existing `PluginLoaderTest` keeps its dependency on real
  backend plugins where it genuinely needs them.
- [T-4] Add `backends/src/test/PluginAbiContractTest.cpp` — the acceptance
  suite. Every [V-n] in `validation.md` lands here as a test, expressed against
  the fixtures. Read-only to all later groups.
- [T-5] Record the *current* behavior before any guard exists: run the suite and
  capture which cases already pass (load-time rejection, [D-3]) and which crash,
  leak, or read out of bounds. Deliverable: the pre-hardening column of the
  evidence log, which is also the proof that [A-2]'s sanitizer detection works.
  - Checks: `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN
    -DBUILD_INFERENCE_ENGINE_TESTS=ON`, build, and
    `ctest -R PluginAbi --output-on-failure` runs the new suite with no vendor
    SDK installed. Load-time cases pass; call-time cases fail — loudly, and
    that is the expected starting state.

## Group 1 — Load-time rejection, proven

- [T-6] No production code expected. Confirm the six load-time rejections
  ([D-3]) against the fixtures: unopenable library, missing entry symbol, entry
  returning null, ABI mismatch, incomplete api table, duplicate `backend_id`.
- [T-7] Assert the diagnostic, not just the outcome: each rejection names the
  plugin path and the reason. If a message is unclear or a rejection path proves
  missing, fix it here — that is the only production change this group allows.
  - Checks: `ctest -R PluginAbi.*Load --output-on-failure` green; a rejected
    plugin leaves `get_plugin_backends()` unchanged.

## Group 2 — Metadata validation ([R-1])

- [T-8] Validate every `neuriplo_layer_info_t` in `populate_metadata` before
  constructing anything from it: null `shape` with non-zero `ndim`, `ndim`
  beyond a stated bound, null `name`, and a `batch_size`/shape combination that
  cannot be reconciled. Reject with `ModelLoadException` naming the plugin, the
  layer index, and which field was bad.
- [T-9] Apply the same validation to inputs and outputs — both arrays come from
  the same struct and the same untrusted source.
  - Checks: the malformed-metadata fixtures are rejected rather than
    dereferenced; `ctest -R PluginAbi.*Metadata` green under ASan and UBSan.

## Group 3 — Output validation and release ownership ([R-2], [R-3], [R-4])

The group where the real ownership defect lives; it needs [T-1] decided first.

- [T-10] Introduce an RAII guard that owns the `infer` out-parameters and calls
  `release_outputs` exactly once on every exit path — normal return, [R-2]
  rejection, or an exception thrown while copying. This is [R-3], and it must go
  in *before* the validation of [T-11], so the new rejection paths are covered
  by it from the start rather than adding two leaks and then fixing them.
- [T-11] Validate the out-parameters per [R-2]: a success return with null
  `tensors` or a non-zero count it cannot satisfy, null `data`, null `shape`
  with non-zero `ndim`, and `size_bytes` against dtype × shape product per
  [Q-3]'s decision.
- [T-12] Validate dtype per [R-4] and [Q-2]'s decision, and resolve the
  silent-fallback inconsistency: `metadata_dtype_from_abi`'s `Float32` default
  and `to_elements`' defaultless switch must agree on rejecting an unknown
  value rather than each guessing differently.
  - Checks: `ctest -R PluginAbi.*Output` green; the leak case runs clean under
    ASan with `detect_leaks=1`; no fixture can make the host read out of bounds.

## Group 4 — Concurrency ([R-5])

- [T-13] Make descriptor access race-free: `get_plugin_backends()` currently
  hands out a reference to a vector that a concurrent `load_backend_plugin` can
  reallocate, and `find_plugin_backend()` iterates it unlocked. Prefer a
  solution that keeps both signatures usable by existing callers; a snapshot
  copy or a stable-storage container both work, and the choice belongs to
  whoever implements it, recorded here when made.
  - Checks: a test loading plugins on one thread while looking up on another is
    clean under TSan if available, and under ASan otherwise; existing callers in
    `src/InferenceBackendSetup.cpp` keep compiling unchanged.

## Group 5 — Isolation ([R-7])

- [T-14] With a good fixture and several broken ones in one directory, assert
  the good plugin loads, serves inference, and returns correct results, that
  `load_backend_plugins` reports the right count, and that the compiled-in
  default backend is unaffected.
  - Checks: `ctest -R PluginAbi.*Isolation --output-on-failure` green.

## Group 6 — Documentation and integration ([R-8])

- [T-15] Extend `docs/PLUGIN_BACKENDS.md` with packaging layout, dependency
  discovery and what `RTLD_LOCAL` / `LOAD_WITH_ALTERED_SEARCH_PATH` mean for
  shipping framework libraries beside a plugin, the compatibility and version
  policy ([D-4]: host-side hardening is not an ABI break), and one deployment
  example carried end to end.
- [T-16] Document the host's validation contract: what the host now rejects and
  the diagnostic a plugin author should expect. A plugin author who reads only
  this page should be able to write a conforming plugin.
- [T-17] Execute everything in `validation.md` and record evidence with dates.
- [T-18] `CHANGELOG.md` under `[Unreleased]`; set roadmap Phase 2 status to
  Complete only once [T-17]'s evidence is filled in.

## Notes

- Conventions reused: `backends/src/test/` for shared backend tests,
  GoogleTest + CTest, `feature/<slug>` off `develop` merged by PR,
  `[skip ci]` on documentation-only commits.
- Delegation, permissions, model routing, and the single acceptance command are
  specified in `orchestration.md`. Groups 2, 3, 4 and 5 are the delegable ones;
  Group 0 is specifier-owned by [D-6] and Group 6 is judgement work.
- Record in this section anything that deviated from the plan and why.
- Group 0 (2026-09-25): [T-2] ships one fixture source,
  `plugin_fixtures/fixture_backend.c`, compiled into seven modules by
  `FIXTURE_BACKEND_ID` / `FIXTURE_API_DEFECT`, instead of one translation unit
  per condition. Load-time defects need their own module (the defect is in the
  exported table); call-time defects are chosen per instance by `model_path`
  on the `FIXTURE_SCRIPTED` module. Same coverage as [R-6], one file to keep in
  step with `plugin_abi.h`. The duplicate-id module lives in a `duplicate/`
  subdirectory so a scan of the main fixture directory never races it against
  `FIXTURE_GOOD`.
- Group 0: the suite is registered with `gtest_discover_tests`, one process per
  case, because the loader's plugin table is process-global and never unloads.
- Group 0: [V-5]'s "copy throws" case has no fixture — after [D-8] the copy is
  bounded by validated sizes and cannot be made to throw from plugin data. It is
  a reviewer obligation on the RAII guard instead; see validation Deviations.
- Group 1 needs no production change: all six load-time rejections pass with
  path and reason in the diagnostic (pre-hardening capture, validation.md).
