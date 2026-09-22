# Requirements — Plugin ABI and Loader Hardening

Spec: `specs/2026-09-22-plugin-abi-loader-hardening/requirements.md` ·
Branch: `feature/plugin-abi-loader-hardening` · Roadmap: Phase 2

## Goal

Make runtime-loaded backends predictable when a plugin is incompatible, broken,
or lying. Today the host validates a plugin at *load* time and then trusts
everything it says at *call* time: pointers, counts, dtypes, and ownership are
taken at face value. A plugin is a third-party binary loaded with `dlopen`, so
that trust is the wrong default.

This phase adds host-side validation at the ABI boundary, makes output-release
ownership exception-safe, and turns plugin failure behavior into executable
contract tests that run in the default build with no vendor SDK installed.

## Context — what exists today

Verified on `origin/develop` @ `62825dd`:

- The C ABI is `include/neuriplo/plugin_abi.h`, `NEURIPLO_PLUGIN_ABI_VERSION 2`,
  one exported symbol (`neuriplo_plugin_get_api_v1`), plugin-owned memory
  released through `release_outputs` / `destroy`.
- `backends/src/plugin/PluginLoader.cpp` loads with
  `dlopen(RTLD_NOW | RTLD_LOCAL)` (and `LoadLibraryExA` on Windows) and already
  rejects, at load time: an unopenable library, a missing entry symbol, an entry
  returning null, an ABI-version mismatch, an incomplete api table, and a
  duplicate `backend_id`. Each rejection logs and closes the handle. That half
  is in good shape — this phase proves it with tests rather than rewriting it.
- `backends/src/test/PluginLoaderTest.cpp` has seven tests, all positive-path
  (discovery, id merging, idempotent load, identity inference, raw-vs-variant
  equivalence, unknown-backend failure). **There is no negative coverage at all**
  for the conditions Phase 2 names.
- The test target is only built when `NEURIPLO_PLUGIN_BACKENDS` is non-empty
  (`backends/src/test/CMakeLists.txt`), so plugin coverage does not run in the
  default `OPENCV_DNN` configuration.

## In Scope

- [R-1] **Metadata validation.** `PluginBackendAdapter::populate_metadata`
  builds `std::vector<int64_t>(layer.shape, layer.shape + layer.ndim)` directly
  from plugin-supplied pointers. A null `shape` with non-zero `ndim`, an absurd
  `ndim`, or a null `name` is undefined behavior in the host today. Validate
  each `neuriplo_layer_info_t` before use and reject the backend with a
  diagnostic naming the plugin and the offending layer index.
- [R-2] **Output validation.** `get_infer_results_raw` trusts `infer`'s
  out-parameters whenever it returns 0: it reads `tensors[i].data` for
  `size_bytes`, and `tensors[i].shape` for `ndim`. A success return with a null
  `tensors`, a null `data`, a null `shape` with non-zero `ndim`, or a
  `size_bytes` inconsistent with dtype × shape product must be rejected as a
  plugin contract violation, not dereferenced.
- [R-3] **Release-ownership safety.** `release_outputs` is called on the happy
  path only. Any throw between `infer` returning and that call — a `bad_alloc`
  in `bytes.assign`, or an [R-2] rejection — leaks plugin-owned memory. Every
  successful `infer` must release exactly once, on every exit path, including
  exceptions. This is the ownership half of Phase 2's exit criteria.
- [R-4] **Dtype validation.** `static_cast<TensorDtype>(tensors[i].dtype)`
  accepts any integer the plugin puts in the field. Reject unknown dtype values
  explicitly. Note the existing inconsistency this exposes:
  `metadata_dtype_from_abi` silently falls back to `Float32` for an unknown
  value, while `to_elements`' switch has no default and would yield an empty
  element vector — two different silent wrongs for the same bad input.
- [R-5] **Thread-safe descriptor access.** `load_plugin_locked` runs under
  `PluginState::mutex`, but `get_plugin_backends()` returns a reference to the
  descriptor vector and `find_plugin_backend()` iterates it with no lock. A
  load concurrent with a lookup is a data race on a `std::vector` that
  reallocates. Make descriptor access safe without changing the public shape of
  these functions where possible.
- [R-6] **Negative contract-test fixtures.** First-party, dependency-free
  malformed plugins built by the test CMake, one per condition: wrong
  `abi_version`, missing entry symbol, entry returning null, incomplete api
  table, duplicate `backend_id`, failing `create`, failing `get_metadata`,
  failing `infer`, malformed metadata ([R-1]), malformed outputs ([R-2]),
  unknown dtype ([R-4]).
- [R-7] **Isolation.** A broken plugin must not destabilize compiled-in
  backends or other plugins: with a good and a bad fixture in the same
  directory, the good one still loads, serves inference, and returns correct
  results, and the compiled-in default backend is unaffected.
- [R-8] **Documentation.** `docs/PLUGIN_BACKENDS.md` gains packaging layout,
  dependency discovery (`RTLD_LOCAL` on POSIX, `LOAD_WITH_ALTERED_SEARCH_PATH`
  on Windows, and what each implies for shipping framework libraries beside the
  plugin), the compatibility/version policy, and one end-to-end deployment
  example a consumer can follow without relying on unwritten behavior.
- [R-9] **The suite runs by default.** The negative tests must build and run in
  the default `-DDEFAULT_BACKEND=OPENCV_DNN` configuration with no vendor SDK
  and no GPU. Coverage that only runs when someone opts into
  `NEURIPLO_PLUGIN_BACKENDS` is coverage that will rot.

## Out of Scope

- `force_gpu` device semantics. A descriptor's `force_gpu` is OR-ed into
  `options.use_gpu` in `create_plugin_backend`, so a plugin can move a caller
  who asked for CPU onto GPU. That is arguably the "no silent change to device
  placement" non-choice in `specs/tech-stack.md`, but it is a *device contract*
  question and belongs to Phase 3/4, not to loader hardening. See [Q-1].
- Plugin unloading or hot reload. Handles are deliberately never unloaded
  because backend objects and api structs must outlive any call; see [D-5].
- An ABI version bump, or any change to `plugin_abi.h` struct layout — see
  [D-4].
- Process-level sandboxing or fault isolation. A plugin that segfaults takes the
  process with it; this phase hardens against malformed *data*, not against
  hostile code.
- Windows CI. The loader's Windows path must keep compiling and the fixtures
  must be portable, but validating on Windows stays a manual, documented step.
- New compiled-in backends, and any change to compiled-in backend behavior.

## Decisions

- [D-1] **Validate at the host boundary, reject with diagnostics, never abort.**
  The host treats every plugin-supplied pointer, count, and enum as untrusted
  input. Rejection follows the established contract: throw the appropriate
  `InferenceException` subtype from the adapter, which
  `setup_inference_engine` already translates into a logged `nullptr`
  (`src/InferenceBackendSetup.cpp`). No `std::exit`, no `assert` that vanishes
  in release builds.
- [D-2] **Fixtures are first-party and dependency-free.** Each malformed plugin
  is a few dozen lines of C against `plugin_abi.h` with no vendor SDK, built by
  the test CMake into the build tree. This is what makes [R-9] achievable: the
  negative suite becomes part of the default gate rather than an opt-in extra.
  Reusing the real backend plugins would tie plugin coverage to having ORT or
  TensorRT installed, which is why it does not run today.
- [D-3] **Load-time rejection is already correct; this phase proves it.** The
  six load-time paths listed in Context are implemented. Phase 2's value there
  is executable evidence, not new code. Effort goes to the call-time paths
  ([R-1] through [R-4]), which are genuinely unguarded.
- [D-4] **No ABI version bump.** All validation is host-side and the struct
  layout does not change, so existing conforming plugins keep working and
  `NEURIPLO_PLUGIN_ABI_VERSION` stays at 2. Hardening the host is not a
  breaking change to the boundary.
- [D-5] **Handles stay loaded for the process lifetime.** The existing comment
  in `PluginLoader.cpp` states the reason and it still holds; rejected plugins
  are closed, accepted ones are not. Unchanged here.
- [D-6] **The acceptance suite is owned by the specifier, not the implementer.**
  `backends/src/test/PluginAbiContractTest.cpp` and the fixtures are written in
  Group 0 and are read-only to every later task group. A worker hardening the
  loader cannot edit the tests that score it; its own tests are output, not
  acceptance. See `orchestration.md`.

## Constraints

- The C ABI boundary rules in `specs/tech-stack.md` hold: no C++ types, STL
  containers, or exceptions cross it; plugin-allocated memory is released by the
  plugin; the host never frees plugin memory.
- `InferenceInterface` lifecycle, metadata, input validation, and output
  semantics are preserved. Plugin backends keep behaving like every other
  backend from a consumer's point of view.
- No new dependency (`specs/tech-stack.md` non-choices). The fixtures use the
  existing GoogleTest/CTest and CMake machinery only.
- C++17, CMake ≥ 3.10, glog the only required library. The fixtures must not
  make any toolchain or SDK a new hard requirement.
- The default `OPENCV_DNN` configure/build/test path stays green, as do
  `./scripts/quality/run.sh` and `./scripts/quality/format.sh --check`.
- Documentation-only commits carry `[skip ci]` (`AGENTS.md`); code commits do
  not.

## Dependencies

- `include/neuriplo/plugin_abi.h` — the boundary being hardened; read-only in
  this phase per [D-4].
- `backends/src/plugin/PluginLoader.cpp` / `.hpp` — the implementation under
  change.
- `backends/src/test/PluginLoaderTest.cpp` and
  `backends/src/test/CMakeLists.txt` — existing positive suite and the
  `NEURIPLO_PLUGIN_BACKENDS` gating that [R-9] must relax.
- `src/InferenceBackendSetup.cpp` — the `nullptr`-and-log translation [D-1]
  relies on.
- `docs/PLUGIN_BACKENDS.md` — the target for [R-8].
- `scripts/quality/sanitizers.sh` — ASan/UBSan, the tool that catches [R-1] and
  [R-2] regressions directly.

## Assumptions & Open Questions

- [A-1] Malformed fixtures compiled as plain C against `plugin_abi.h` build
  anywhere the host builds, needing only a C compiler already required by the
  project. Confirm in Group 0 — if a platform makes building a test-only shared
  library awkward, [R-9] is the requirement at risk.
- [A-2] ASan/UBSan under the existing `scripts/quality/sanitizers.sh` will flag
  the null-dereference and out-of-bounds reads that [R-1] and [R-2] describe, so
  a regression in the validation shows up as a sanitizer failure rather than a
  silent pass. Confirm by running the suite with a fixture that violates the
  contract *before* the guard is added — the check should fail loudly first.
- [Q-1] Does `force_gpu`'s silent OR into `use_gpu` get fixed here or in
  Phase 3/4? Recommendation: Phase 3/4, because a fix means deciding what a
  device request *means* across all backends, which is exactly Phase 3's
  contract work. It is recorded in Out of Scope so it is not lost.
- [Q-2] Should an unknown dtype reject the whole inference call, or drop the
  offending tensor and continue? Recommendation: reject the call — a plugin
  returning a dtype the host cannot name is broken, and partial results are
  harder to debug than a clear failure. Needs a decision before Group 3.
- [Q-3] Should the size/shape consistency check in [R-2] be strict equality
  (`size_bytes == element_size × Π shape`) or a lower bound? Recommendation:
  strict equality, since every conforming plugin knows both. A lower bound
  would admit the buffer-overread case the check exists to catch.

## Definition of Done (requirements level)

- [ ] Every [R-n] implemented or explicitly deferred with a tracked location
- [ ] [Q-2] and [Q-3] decided and recorded before Group 3 starts
- [ ] Negative suite runs and passes in the default `OPENCV_DNN` build ([R-9])
- [ ] No In Scope behavior silently dropped; no Out of Scope work smuggled in
- [ ] `plugin_abi.h` unchanged and `NEURIPLO_PLUGIN_ABI_VERSION` still 2 ([D-4])
