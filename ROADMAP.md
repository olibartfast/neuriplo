# Neuriplo Library Roadmap

This roadmap is scoped to the `neuriplo` backend-orchestration library. It
prioritizes correctness, backend compatibility, dependency safety, device
placement assumptions, fallback behavior, and performance regressions.

## Roadmap Principles

- Keep `develop` as the integration branch and `master` release-only.
- Preserve the existing `setup_inference_engine(...)` contract unless a
  planned interface migration is explicitly approved.
- Treat runtime version changes as compatibility work, not routine dependency
  bumps.
- Keep backend setup, CMake registration, generated docs, Docker images, and CI
  matrix entries synchronized.
- Prefer opt-in behavior for new runtime paths, decorators, providers, plugins,
  and fallback modes.
- Do not silently fall back across devices, providers, or model formats.

## Phase 0: Stabilize The Current Surface

Goal: make the existing multi-backend platform predictable before expanding the
API or dependency surface.

Work items:

- Close known CI reliability issues around disk pressure, Docker Buildx
  bootstrap flakes, and backend image caching.
- Fix or document backend-specific test gaps, especially the TensorRT metadata
  crash noted in `backends/tensorrt/test/TensorRTInferTest.cpp`.
- Keep generated backend documentation in sync from `docs/backends.yaml` and
  `versions.env`.
- Run the fast local quality gate before branch handoff:
  `./scripts/quality/run.sh` and `./scripts/quality/format.sh --check`.
- Keep `docs/TROUBLESHOOTING.md` current when a failure pattern costs real
  debugging time.

Exit criteria:

- Default `OPENCV_DNN` configure/build/test path is green.
- Whole-tree format, cppcheck, and backend-docs checks are green.
- Known backend failures are either fixed, isolated, or explicitly documented.

## Phase 1: Backend Registry And Dependency Hygiene

Goal: reduce the cost and risk of maintaining thirteen pinned native backends.

Work items:

- Make `docs/backends.yaml` the single operational backend inventory for docs,
  setup scripts, Docker metadata, test executable names, and architecture/GPU
  support.
- Add validation that `BackendRegistry.cmake`, setup scripts, Docker files,
  and generated docs agree on supported backend IDs.
- Tighten dependency validation so selected-backend failures are clear and
  non-selected backends remain ignored.
- Standardize setup-script output, environment exports, and installation
  verification.
- Add a release checklist for dependency bumps: upstream changelog, ABI/API
  changes, Docker rebuild, local backend test, generated docs, and rollback
  notes.

Exit criteria:

- Adding or bumping a backend fails early if CMake, scripts, docs, or CI metadata
  are out of sync.
- Dependency validation errors name the missing header/library/path and the
  backend that required it.

## Phase 2: Runtime Selection And Device Semantics

Goal: move beyond the current `bool use_gpu` limit without breaking existing
consumers.

Work items:

- Implement the focused ONNX Runtime execution-provider plan in `docs/plans/ort-execution-providers.md` as the
  first provider-selection increment.
- Keep Phase 1 ORT provider selection contained to `ONNX_RUNTIME`, using an
  environment/config path before changing shared interfaces.
- Extend the existing `EngineOptions` API (backend, batch, model, plugin
  directory fields already landed with the multi-backend registry) with explicit
  device, provider/delegate, and fallback policy fields.
- Add strict fallback semantics: requested accelerators must either be used,
  fail clearly, or fall back only when the user requested fallback.
- Add test coverage for provider/delegate parsing, unavailable-provider errors,
  CPU default behavior, and fallback logging.

Exit criteria:

- Existing `setup_inference_engine(model_path, use_gpu, batch_size, input_sizes)`
  callers keep working.
- Provider/device selection is observable in logs and tests.
- No backend silently moves from accelerator to CPU.

## Phase 3: Plugin Backend Maturity

Goal: make runtime-loaded backends a reliable platform feature, not a side path.

Work items:

- Expand plugin tests beyond the current ONNX Runtime identity-model path.
- Add ABI compatibility tests for mismatched `NEURIPLO_PLUGIN_ABI_VERSION`,
  missing symbols, broken metadata, inference errors, and output release rules.
- Document plugin packaging layout, version compatibility, dependency discovery,
  and deployment examples.
- Add CI coverage for at least one host-without-backend plus plugin-backend
  combination.
- Define plugin metadata fields needed by serving layers: backend id, version,
  model formats, device capabilities, and supported tensor dtypes.

Exit criteria:

- Broken plugins are skipped or reported without destabilizing compiled-in
  backends.
- Plugin memory ownership and error propagation are covered by tests.
- Plugin docs are sufficient to build and deploy a backend shared library.

## Phase 4: Test Matrix And Model Fixtures

Goal: make backend behavior comparable across vision, graph, and GGUF-native
runtime families.

Work items:

- Split tests into contract tests, backend smoke tests, model-format tests,
  device/provider tests, and performance baselines.
- Standardize small deterministic model fixtures for classification and object
  detection paths where each backend supports them.
- Add quantized fixture coverage for runtimes that require or benefit from
  INT8/INT16 models.
- Keep architecture-specific lanes explicit: x86_64, ARM64, NVIDIA GPU, AMD
  ROCm, and manual hardware-only targets.
- Add regression checks for metadata shape population, batch-size behavior,
  unsupported input counts, and model-format rejection.

Exit criteria:

- Every supported backend has at least one deterministic smoke test.
- Contract-level behavior is checked independently from vendor runtime quirks.
- Manual hardware-only validation has written commands and expected outcomes.

## Phase 5: Serving Integration Readiness

Goal: make Neuriplo safer to use as the backend layer under `neuriplo-infer` and
other serving/runtime projects.

Work items:

- Document the stable public contract between this library and consumers:
  construction, errors, metadata, tensor buffers, batching, and lifecycle.
- Add examples for compiled-in backend selection and plugin-backed runtime
  selection.
- Define thread-safety expectations for `InferenceInterface`, `ModelRunner`,
  decorators, and backend instances.
- Add lifecycle tests for repeated construction/destruction, failed loads, and
  process-scope vendor runtime initialization.
- Add optional performance instrumentation that remains disabled by default.

Exit criteria:

- Consumer projects can choose backend, device/provider, and plugin directory
  without relying on undocumented environment behavior.
- Failure modes return clear errors or `nullptr` according to the documented
  contract.
- Profiling/logging decorators do not change production behavior when disabled.

## Phase 6: Release And Maintenance Discipline

Goal: make releases repeatable and reduce drift between documentation, CI, and
backend implementation.

Work items:

- Define a release branch checklist for `master`: backend matrix status,
  generated docs, dependency versions, known exclusions, and migration notes.
- Add a compatibility table for backend IDs, model formats, architecture support,
  GPU requirements, and tested runtime versions.
- Track deprecations explicitly in `docs/Versioning.md` and link them from
  backend-specific docs.
- Keep README broad and move backend-specific details into dedicated docs.
- Add changelog entries for public behavior changes, dependency bumps, fallback
  semantics, and plugin ABI changes.

Exit criteria:

- A release can be cut from `develop` to `master` with a documented validation
  trail.
- Users can tell which backend/runtime combinations are supported, experimental,
  deprecated, or hardware-only.

## Near-Term Priority Order

1. Resolve the TensorRT metadata test TODO or quarantine it with a precise issue.
2. Add backend metadata consistency checks around `docs/backends.yaml`,
   `BackendRegistry.cmake`, and setup scripts.
3. Implement the ONNX Runtime execution-provider Phase 1 plan from `docs/plans/ort-execution-providers.md`.
4. Expand plugin ABI and loader tests.
5. Draft the `EngineOptions` device/provider/fallback field extension proposal.

(Was item 1: the CI free-disk-space fix — merged as PR #15, develop green.)

## Validation Commands

Use these gates according to the risk of the change:

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
./scripts/quality/run.sh
./scripts/quality/format.sh --check
./scripts/test_backends.sh --backend <BACKEND_NAME>
python3 scripts/gen_backend_docs.py --check
```

For Dockerfile or workflow changes, inspect and run the affected job with
`act` as described in `docs/LOCAL_CI.md`.
