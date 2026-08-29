# Neuriplo Technical Stack

> Status: working brownfield constitution, reconstructed on 2026-08-29. The
> implementation and executable checks take precedence if this file drifts;
> update both in the same branch when a durable technical decision changes.

## Core Stack

| Area | Current choice | Boundary |
| --- | --- | --- |
| Library | C++17 shared library | Do not raise the language standard without an explicit compatibility decision. |
| Build | CMake 3.10 or newer | `CMakeLists.txt` and `cmake/` define backend selection, validation, compilation, linking, tests, and plugins. |
| Required libraries | OpenCV and glog | They are required for the common library; backend SDKs remain selection-dependent. |
| Backends | 14 registered runtime or pipeline backends | `cmake/BackendRegistry.cmake` is the CMake-visible backend-ID authority. |
| Tests | GoogleTest and CTest | Backend-agnostic contract tests and selected-backend tests are enabled by `BUILD_INFERENCE_ENGINE_TESTS`. |
| Automation | Bash, Python 3, Docker, GitHub Actions | Scripts should be reproducible locally; CI supplies backend-specific environments. |
| Quality | clang-format 18, cppcheck, clang-tidy, ASan, UBSan, pre-commit | Use the checked-in scripts rather than ad hoc command variants. |

The registered IDs are `OPENCV_DNN`, `ONNX_RUNTIME`, `LIBTORCH`,
`LIBTENSORFLOW`, `TENSORRT`, `OPENVINO`, `GGML`, `TVM`, `MIGRAPHX`, `CACTUS`,
`LLAMACPP`, `EXECUTORCH`, `LITERT`, and `DALI`. Their tested dependency versions
live in `versions.env`; the human-readable backend inventory lives in
`docs/backends.yaml` and generated sections of `docs/DEPENDENCY_MANAGEMENT.md`.

## Architectural Boundaries

- `InferenceInterface` is the common backend contract. Preserve its lifecycle,
  metadata, input validation, and output semantics across implementations.
- `EngineOptions` is the extensible construction path. The legacy
  `setup_inference_engine(model_path, use_gpu, batch_size, input_sizes)` overload
  remains a compatibility contract until a planned migration says otherwise.
- `BackendRuntimeRegistry` and one `IBackendRuntimeFactory` implementation per
  backend provide runtime lookup and construction.
- `DEFAULT_BACKEND` preserves the default and single-backend path.
  `NEURIPLO_BACKENDS` adds compiled-in backends, while
  `NEURIPLO_PLUGIN_BACKENDS` builds dependency-isolated `dlopen` plugins.
- The plugin boundary is the versioned C ABI in
  `include/neuriplo/plugin_abi.h`. Memory ownership, metadata, error propagation,
  and ABI mismatch behavior must be covered explicitly.
- Decorators are optional and disabled by default. They must not change the
  production path when not enabled.
- `RawOutputTensor` is the typed contiguous-buffer path. Avoid materializing
  per-element variants in performance-critical backend overrides.

## Device and Fallback Assumptions

- CPU remains the safe default except for runtimes that intrinsically require
  another device.
- GPU, execution-provider, delegate, NPU, or offload selection must be opt-in
  and observable.
- A requested accelerator must either be used, fail clearly, or fall back only
  under an explicit caller policy. Never infer success from a runtime that
  silently moved work to CPU.
- Provider and delegate capabilities belong inside the owning backend rather
  than becoming duplicate top-level backend IDs.
- Hardware-only behavior needs written commands, target details, and expected
  results because ordinary CI cannot validate it.

## Durable Sources of Truth

| Concern | Source |
| --- | --- |
| Agent scope, owned paths, and automated-change limits | `REPO_META.yaml` and `AGENTS.md` |
| Mission and product boundaries | `specs/mission.md` |
| Delivery order | `specs/roadmap.md` |
| Backend IDs and CMake properties | `cmake/BackendRegistry.cmake` |
| Tested runtime versions | `versions.env` |
| Backend documentation metadata | `docs/backends.yaml` |
| Public version and release history | `VERSION` and `CHANGELOG.md` |
| Troubleshooting knowledge | `docs/TROUBLESHOOTING.md` |

Do not introduce a second manually maintained backend inventory. When
`versions.env` or `docs/backends.yaml` changes, regenerate the `GEN:` sections
with `python3 scripts/gen_backend_docs.py` in the same change.

## Explicit Non-Choices

- No new dependency without explicit approval and a compatibility review.
- No autonomous inference-logic or performance-critical kernel changes.
- No silent change to backend selection, device placement, or fallback behavior.
- No framework or language migration merely to simplify one feature.
- No backend-specific setup, model-format, Docker, build, or troubleshooting
  expansion in `Readme.md`; keep those details in the appropriate `docs/` guide.
- No feature implementation before consequential scope and validation decisions
  are recorded in a feature packet.

## Branch and Documentation Workflow

- `develop` is the integration branch. All feature branches and worktrees start
  from `develop`; `master` is release-only.
- A selected roadmap phase gets `specs/YYYY-MM-DD-feature-name/` containing
  `requirements.md`, `plan.md`, and `validation.md` before implementation.
- Requirements describe what, the plan describes how, and validation describes
  proof. Change the spec in the same branch whenever implementation changes the
  contract.
- Documentation-only commits include `[skip ci]`.
- Before pushing documentation with links, verify every relative target exists
  and every absolute URL is reachable.

## Validation Entrypoints

Use the smallest relevant checks while implementing, then the declared feature
validation before handoff:

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
./scripts/quality/run.sh
./scripts/quality/format.sh --check
./scripts/test_backends.sh --backend <BACKEND_NAME>
python3 scripts/gen_backend_docs.py --check
```

Dockerfile and workflow changes also require the affected `act` dry run and
full local job described in `docs/LOCAL_CI.md`.

## Open Technical Decisions

- Extend `EngineOptions` with backend-neutral device, provider/delegate, and
  fallback fields while retaining the legacy overload.
- Define repository-wide backend support tiers and the validation required for
  each tier.
- Define representative performance baselines without pretending unlike
  backend, model, device, and architecture combinations are directly
  interchangeable.
