# Plan: Configurable ONNX Runtime Execution Providers (Phase 1 — vision)

> **Status (2026-06-12): Phase 1 is implemented** — commit `33bb936`
> (`NEURIPLO_ORT_EP` selector in `ORTInfer.cpp`, `ORT_ENABLE_*_EP` CMake
> options, provider-parsing tests, user guide in
> `docs/ORT_EXECUTION_PROVIDERS.md`). This document is retained for the
> rationale and the still-open follow-ups: ExecuTorch delegate sibling path,
> QNN SDK dependency wiring + quantized test fixtures, on-device NPU
> validation, and the `DeviceType`/`EngineOptions` interface refactor
> (option A).

## Approach
Add a configurable Execution Provider (EP) selection layer inside the existing
`ONNX_RUNTIME` backend, not a new backend ID per provider.

Rationale: ONNX Runtime already models hardware targets as EPs, ordered by priority. A new
neuriplo backend ID per ORT EP would duplicate ORT's abstraction and multiply registry/test
surface area. The selection belongs where CUDA/ROCm selection already lives in
`ORTInfer.cpp:12-50`, but generalized beyond the current hard-coded CUDA-then-ROCm probe.

Default builds stay untouched: `ONNX_RUNTIME` continues to use CPU unless `use_gpu=true` or an
explicit ORT EP selection is provided.

This plan is specifically for ORT. ExecuTorch has the same accelerator-selection need, but the
equivalent concept is a **backend/delegate**, not an Execution Provider. That work should be
tracked as a sibling ExecuTorch delegate plan because ExecuTorch usually requires ahead-of-time
export/lowering into a backend-specific representation, whereas ORT EPs partition ONNX graphs
at runtime.

## Hard constraints this must respect
- **`new-dependency` is forbidden** (`REPO_META.yaml`) — provider-specific SDKs and
  provider-enabled ORT builds are genuinely new deps. This plan **isolates** that:
  provider support is compile/config gated and nothing new is pulled unless explicitly enabled.
  Actual dependency wiring (setup script / Dockerfile / `versions.env`) is a **separate
  sign-off item**, not bundled in.
- **`preserve_device_placement_expectations` / `protect_fallback_behavior`** — QNN selection
  or any other non-CPU EP selection must not change behavior for existing CPU/CUDA users, and
  provider→CPU fallback must be explicit and logged, never silent.
- **`inference-logic-change` is forbidden** — EP *selection* is config, not inference logic;
  but it's adjacent. Flag for the reviewer.

## Design problem to settle first
`InferenceInterface` only has a `bool use_gpu` — no "execution provider" or "device type"
concept. ORT EP selection needs to be selectable without changing other backends. Options:
- **A.** New optional ctor param / enum `DeviceType { CPU, GPU, NPU }` threaded through
  `InferenceInterface` — cleanest, but touches the shared interface (ripples to neuriplo-infer).
- **B.** Non-CPU EPs piggyback on `use_gpu=true` + a CMake/env flag — minimal interface change,
  but conflates "GPU" with NPU/edge accelerators.
- **C.** Env var (`NEURIPLO_ORT_EP=<provider>`) read inside `ORTInfer` only — zero interface
  change, contained entirely to the ORT backend. Good for Phase 1, revisit for Phase 2.

Recommendation: **C for this increment** (fully contained, no cross-repo ripple), with A as
the eventual target.

Phase 1 provider names:
- `cpu` — default CPU EP.
- `cuda` — keep existing CUDA path.
- `tensorrt` — optional; only if the linked ORT package exposes TensorRT.
- `openvino` — optional; only if the linked ORT package exposes OpenVINO.
- `directml` — Windows-only; document but do not wire Linux CI.
- `migraphx` — optional; preferred AMD GPU path.
- `qnn` — optional Qualcomm AI Engine Direct / QNN EP.
- `nnapi`, `coreml`, `xnnpack`, `acl`, `armnn`, `rknpu`, `vitisai`, `cann`, `azure`, `tvm`
  — document as ORT-supported provider names, but only implement/test as their SDKs and ORT
  builds are explicitly approved.

Do **not** add new ROCm work. ORT marks AMD ROCm as deprecated; keep existing code only for
backward compatibility until a separate removal/deprecation decision is made.

## ExecuTorch sibling path
ExecuTorch accelerator integrations should be modeled as backend/delegate variants of the
existing `EXECUTORCH` backend, not ORT EPs. Relevant delegates/backends include:
- `xnnpack` — default mobile/ARM CPU acceleration.
- `coreml` — Apple Core ML / ANE / GPU path.
- `qnn` — Qualcomm AI Engine Direct / Hexagon DSP/NPU path.
- `vulkan` — GPU compute on Android/Linux.
- `mps` — Apple Metal Performance Shaders.
- `ethos-u` — Arm microcontroller NPU path.
- custom delegates — vendor-specific accelerator integrations.

Key difference from ORT: ExecuTorch delegates are usually selected during PyTorch export and
lowering, then consumed by the lightweight runtime. That means neuriplo support is not just a
runtime option; it also needs documented `.pte` export flows, delegate-specific build flags,
and tests using models exported for that delegate.

Recommended later phase: add `NEURIPLO_EXECUTORCH_DELEGATE=<delegate>` or a future shared
device-selection API, then wire delegate-specific ExecuTorch builds and `.pte` test fixtures.

## Concrete changes (Phase 1)
| # | File | Change |
|---|------|--------|
| 1 | `cmake/ONNXRuntime.cmake` | Add provider-selection CMake cache options, e.g. `ORT_ENABLE_QNN_EP=OFF`, `ORT_ENABLE_OPENVINO_EP=OFF`, etc. Add compile defs only for enabled providers. |
| 2 | `cmake/DependencyValidation.cmake` + `cmake/LinkBackend.cmake` | Validate the selected ORT package exposes the requested provider and link/copy any provider libraries needed at runtime, such as ORT provider plugins or vendor SDK libraries. |
| 3 | `backends/onnx-runtime/src/ORTInfer.cpp` | Replace hard-coded CUDA→ROCm probing with an ordered provider selector: `NEURIPLO_ORT_EP=cpu,cuda,qnn` or single-provider shorthand. Append supported EPs in priority order. Log selected providers and fallback policy. |
| 4 | `backends/onnx-runtime/src/ORTInfer.hpp` | Minimal — only if option C needs a member; likely no header change. |
| 5 | `backends/onnx-runtime/test/` | Add tests for provider string parsing, unavailable-provider fallback/error behavior, and CPU default behavior. Provider-specific runtime tests only where dependencies exist. |
| 6 | `.github/workflows/ci.yml` | Optional provider matrix entries only for providers whose dependency setup is approved. For QNN on ARM: build/configure only unless real hardware is available. |
| 7 | `docs/backends.yaml` + `python3 scripts/gen_backend_docs.py` | Either add variant-aware ORT provider docs or keep generated backend docs unchanged and link to an ORT EP guide. Regenerate `GEN:` sections if YAML changes. |
| 8 | `docs/` backend guide | ORT EP guide: provider list, env var syntax, fallback policy, dependency requirements, QNN quantization/device notes. |

## Known caveat affecting the test plan
**QNN HTP only runs quantized models (INT8/INT16).** The existing ORT test model
(`export_torchvision_classifier.py`) is fp32 — it will *not* run on the Hexagon NPU as-is.
Phase 1 testing options:
- (a) run the QNN EP on CPU backend in CI (validates wiring, not NPU),
- (b) add a quantized test model for real on-device runs on the RUBIK Pi 3.

Real NPU validation is manual, on hardware — CI cannot cover it.

Other EPs have similar constraints: the ORT package must be built with that EP, vendor runtime
libraries must be present, and model/operator coverage may cause partial assignment. CPU fallback
must be controlled explicitly with ORT's CPU fallback config instead of inferred from logs.

## Explicitly out of scope (later phases)
- LLM paths (llama.cpp Hexagon backend)
- ExecuTorch delegate support (`xnnpack`, `qnn`, `coreml`, `vulkan`, `mps`, `ethos-u`)
- QNN SDK dependency acquisition wiring (needs the `new-dependency` sign-off)
- `DeviceType` enum refactor of `InferenceInterface` (option A)
- Cross-repo device-selection plumbing in `neuriplo-infer`

## Open questions
1. **Device selection mechanism** — OK with option **C** (`NEURIPLO_ORT_EP`, contained) for Phase 1?
2. **Provider scope** — implement only `cpu`, `cuda`, `migraphx`, and `qnn` first, or also wire
   provider entries for OpenVINO/TensorRT/XNNPACK/etc. behind compile guards?
3. **Fallback policy** — should non-CPU EP requests default to strict mode
   (`session.disable_cpu_ep_fallback=1`) or allow explicit CPU fallback when the env var includes
   `cpu` in the provider list?
4. **CI** — want the arm64 build-only QNN job now, or skip CI until there's a real device in the loop?
5. **Test model** — should Phase 1 include adding a quantized ONNX test model, or defer all
   NPU testing to manual on-device?

## Context references
- Hardware: RUBIK Pi 3 (QCS6490) as dev board; OnePlus 10T / Realme GT7 **Pro** as phone targets.
- arm64 Linux already proven (full stack runs on Jetson Nano); arm64 CI lane already exists
  (CACTUS → `ubuntu-24.04-arm`).
- Repo cluster: `neuriplo-infer` (app) → `neuriplo-tasks` (CV pre/post) + `neuriplo` (this repo,
  backend abstraction) + `videocapture` (I/O). QNN work is contained to neuriplo.
