# Neuriplo Technical Stack

> Status: working brownfield constitution, reconstructed on 2026-08-29. The
> implementation and executable checks take precedence if this file drifts;
> update both in the same branch when a durable technical decision changes.

## Core Stack

| Area | Current choice | Boundary |
| --- | --- | --- |
| Library | C++17 shared library | Do not raise the language standard without an explicit compatibility decision. Its public headers are the C++ API (same-toolchain consumers) and the consumer C ABI `include/neuriplo/neuriplo_c.h` (C99, stable, versioned) with the header-only wrapper `include/neuriplo/neuriplo.hpp`. |
| Build | CMake 3.10 or newer | `CMakeLists.txt` and `cmake/` define backend selection, validation, compilation, linking, tests, and plugins. |
| Required libraries | glog | Required by the common library. OpenCV is **not**: it is required only by the `OPENCV_DNN` backend and is linked from there (`cmake/LinkBackend.cmake`), so the other 13 backends build and test with no OpenCV installed. Backend SDKs remain selection-dependent. |
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
- The consumer boundary is the versioned C ABI in
  `include/neuriplo/neuriplo_c.h` (`NEURIPLO_C_API_VERSION`, independent of the
  plugin ABI version) -- the opposite direction: applications call it,
  `libneuriplo` implements it. No C++ type or exception crosses it; its
  exported symbol list (`scripts/abi/neuriplo_c.symbols`) and struct layouts are
  checked in the build. `include/neuriplo/neuriplo.hpp` is a header-only C++
  wrapper over it that includes nothing else from neuriplo. The C++ API
  (`InferenceBackendSetup.hpp`, `InferenceInterface`) remains for consumers
  built with the same toolchain; it is not an ABI.
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
| Automated-change scope and owned paths | `REPO_META.yaml` and `AGENTS.md` |
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
- Non-trivial public-behavior or architecture work records scope and validation
  in a dated feature packet before implementation; small fixes scale the artifact
  to their risk.

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

_Revision: 2026-09-26 - added the consumer C ABI boundary beside the plugin
ABI (Phase 7, `specs/2026-09-25-consumer-c-abi`)._
