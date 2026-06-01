# Refactoring Plan — Design-Pattern-Driven Backend Architecture

Status: proposal / not started
Owner: backend orchestration
Scope: `backends/`, `include/`, `src/`, `cmake/`, `docs/`

This plan refactors neuriplo's backend layer around five design patterns
(**Adapter**, **Bridge**, **Abstract Factory**, **Decorator**, **State**) while
preserving every hard constraint in `REPO_META.yaml`. Each step is written to be
**atomic** (one concern, independently reviewable). Steps are grouped into
**parallel waves**: every step inside a wave is independent and can be dispatched
to a separate subagent concurrently. A step may only start once **all** steps in
its declared dependency waves are merged.

> **Interface decision:** the existing `InferenceInterface`
> (`backends/src/InferenceInterface.hpp`) **is** the common backend contract — all
> 13 backends already derive from it. We keep it as-is and build every pattern on
> top of it. No separate `IBackend` type is introduced.

---

## 1. Constraints that bound every step

From `REPO_META.yaml` and `AGENTS.md`:

- `preserve_backend_selection_behavior: true` — `-DDEFAULT_BACKEND` semantics and
  the set of `NEURIPLO_BACKEND_IDS` must not change.
- `preserve_device_placement_expectations: true` — CPU/GPU/EP selection (e.g.
  `ORTInfer` CUDA→ROCm probe, `NEURIPLO_ORT_EP`) must behave identically.
- `protect_fallback_behavior: true` — provider/device fallback stays explicit and
  logged, never silent.
- `new-dependency` **forbidden** — no new third-party libs; everything below is
  internal C++17 + existing deps (glog, OpenCV).
- `inference-logic-change` / `perf-critical-kernel-change` **forbidden** — the
  numeric path inside `get_infer_results` must not change. Structural extraction
  is allowed; altering math/quantization defaults is **not** (see `QuantizedBackend`
  caveat in Wave 3).
- **Cross-repo contract:** `setup_inference_engine(model_path, use_gpu, batch_size,
  input_sizes) -> std::unique_ptr<InferenceInterface>` is consumed by
  `vision-inference`. Its signature and return type **must keep working** after the
  refactor. This is the single most important compatibility rule — and it is
  automatically satisfied because `InferenceInterface` stays the public type.
- **Compile-time single-backend model:** only the `DEFAULT_BACKEND` is compiled
  (`#ifdef USE_*`). Therefore Abstract Factory / Decorator / Bridge are all
  expressed within a single-backend translation unit; the "family" selection is
  resolved at compile time, not runtime. The pattern interfaces still hold; only
  the registration mechanism differs from a textbook runtime factory.
- Branch off `develop`. Run `clang-format --dry-run --Werror` on touched
  `.cpp`/`.hpp` before pushing. Docs-only commits use `[skip ci]`.

---

## 2. Target architecture

```
                       ┌────────────────────────────────────┐
   high-level user ───▶│            ModelRunner              │  (Bridge: abstraction)
                       │ holds unique_ptr<InferenceInterface>│
                       └─────────────────┬──────────────────┘
                                         │ delegates to
                                         ▼
                       ┌────────────────────────────────────┐
                       │        InferenceInterface           │  (existing common contract)
                       │  get_infer_results(), metadata(),   │
                       │  + new: state(), load()             │
                       └─────────────────┬──────────────────┘
            ┌────────────────────────────┼─────────────────────────┐
            ▼ (Decorator)                ▼ (Adapter)                ▼
   ┌──────────────────┐        ┌────────────────────┐    BackendState (State)
   │ ProfilingBackend │        │ OrtBackend          │    Uninitialized→Loading
   │ LoggingBackend   │        │ TensorRtBackend     │           →Ready→Failed
   │ CachingBackend   │        │ OpenVinoBackend     │
   │ QuantizedBackend │        │ ...all 13 backends  │
   └────────┬─────────┘        └────────────────────┘
            │ wraps an InferenceInterface (chainable)
            ▼
     InferenceInterface

   ┌──────────────────────────────────────────────────────────┐
   │            IBackendRuntimeFactory (Abstract Factory)        │
   │  create_backend()   -> InferenceInterface                   │
   │  create_allocator() -> IAllocator                           │
   │  create_converter() -> ITensorConverter                     │
   └──────────────────────────────────────────────────────────┘
        one concrete factory per backend family
        (OrtRuntimeFactory, TensorRtRuntimeFactory, ...)
```

### Pattern → concrete mapping

| Pattern | Role | Concrete artifacts |
|---|---|---|
| **Adapter** | `InferenceInterface` over each vendor SDK | `OrtBackend`, `TensorRtBackend`, `OpenVinoBackend`, `OpenCvDnnBackend`, `LibTorchBackend`, `LibTensorFlowBackend`, `GgmlBackend`, `TvmBackend`, `MiGraphXBackend`, `CactusBackend`, `LlamaCppBackend`, `ExecuTorchBackend`, `LiteRtBackend` (all 13 — these are the *existing* `*Infer` classes) |
| **Bridge** | Abstraction decoupled from implementor | `ModelRunner` → `InferenceInterface` |
| **Abstract Factory** | Create a coherent runtime family | `IBackendRuntimeFactory` + `{InferenceInterface, IAllocator, ITensorConverter}`; one factory per backend |
| **Decorator** | Cross-cutting behavior, same interface | `BackendDecorator` base + `ProfilingBackend`, `LoggingBackend`, `CachingBackend`, `QuantizedBackend` |
| **State** | Explicit backend lifecycle | `BackendState{Uninitialized,Loading,Ready,Failed}` + transition logic |

### Compatibility strategy (non-negotiable)

- `InferenceInterface` remains the common contract and the public return type, so
  the cross-repo `setup_inference_engine` contract is preserved by construction.
- Add **two optional virtuals with default implementations** to
  `InferenceInterface`: `virtual BackendState state() const` and `virtual void
  load()`. Defaults make load a no-op that reports `Ready`, matching today's
  load-in-constructor behavior — so **all 13 existing backends keep compiling with
  zero edits** until each is migrated.
- `ModelRunner`, all decorators, and factory products are all typed against
  `InferenceInterface` (no new base type to thread through).
- The `std::exit(1)` failure path in `ORTInfer` (and similar) becomes a
  `Failed` state + thrown `ModelLoadException`. Because `ModelLoadException`
  already exists, confirm `vision-inference` tolerates it; otherwise gate behind
  the facade returning `nullptr` as today.

---

## 3. Execution model (waves & subagents)

- **Wave N** runs only after **Wave N-1** is fully merged.
- Inside a wave, each step is independent → assign **one subagent per step** and
  run them concurrently.
- Per-backend steps (×13) are the widest parallel fan-out; they share no files.
- Every step ends with: build the affected backend, run its tests, `clang-format`.

Dependency graph (high level):

```
Wave 0 (state enum + interface hooks + helper interfaces)
        │
        ▼
Wave 1 (Bridge: ModelRunner)
        │
        ├──────────────▶ Wave 2 (per-backend adapter cleanups ×13)
        │                                │
        └──────────────▶ Wave 3 (decorators ×4) ◀── (independent of Wave 2)
                                         │
                                         ▼
                              Wave 4 (factories ×13) ──▶ Wave 5 (State integration)
                                         │
                                         ▼
                              Wave 6 (wiring, CMake, tests, docs)
```

---

## Wave 0 — Foundations

### S0.1 — `BackendState` enum + transition helper  · subagent A · no deps · new file
- New: `backends/src/BackendState.hpp`
- Define `enum class BackendState { Uninitialized, Loading, Ready, Failed };`
  plus `bool is_valid_transition(BackendState from, BackendState to)` and
  `to_string`.
- Allowed transitions: `Uninitialized→Loading`, `Loading→Ready`, `Loading→Failed`,
  `Ready→Failed` (runtime error), `Failed→Loading` (retry).
- Accept: header compiles standalone; tiny unit test asserting the transition table.

### S0.2 — Add lifecycle hooks to `InferenceInterface`  · subagent B · deps: S0.1 · **keystone**
- Edit: `backends/src/InferenceInterface.hpp` / `.cpp`
- Add `virtual BackendState state() const noexcept { return state_; }` and
  `virtual void load() { state_ = BackendState::Ready; }` with **default bodies**,
  plus a protected `BackendState state_{BackendState::Uninitialized}`.
- Keep all existing members/helpers (`start_timer`, `validate_input`, …) intact.
- Accept: full matrix still builds (`./scripts/test_backends.sh`) with **zero
  per-backend edits**. This proves the foundation is backward-compatible before
  anything else changes.

### S0.3 — `IAllocator` interface  · subagent C · no deps · new file
- New: `backends/src/IAllocator.hpp`
- Minimal allocation abstraction for the factory family: `allocate(size_t)`,
  `deallocate(void*)`, `name()`. Include a header-only `HostAllocator` default so
  non-migrated backends ignore it.
- Accept: compiles standalone + trivial alloc/free test.

### S0.4 — `ITensorConverter` interface  · subagent D · no deps · new file
- New: `backends/src/ITensorConverter.hpp`
- Abstraction over the raw-bytes↔typed-tensor conversion currently inlined in
  backends (e.g. `ORTInfer::get_infer_results` byte casting). Methods expressed in
  terms of `TensorElement` + shapes + element-type enum. **Interface only — no
  conversion math changes here.**
- Accept: compiles standalone.

### S0.5 — `IBackendRuntimeFactory` interface  · subagent E · deps: S0.3, S0.4 · new file
- New: `backends/src/IBackendRuntimeFactory.hpp`
- Abstract Factory:
  `create_backend(...) -> std::unique_ptr<InferenceInterface>`,
  `create_allocator() -> std::unique_ptr<IAllocator>`,
  `create_converter() -> std::unique_ptr<ITensorConverter>`.
- Accept: compiles standalone.

### S0.6 — `BackendDecorator` base  · subagent F · deps: S0.2 · new file
- New: `backends/src/BackendDecorator.hpp`
- Derives from `InferenceInterface`, holds
  `std::unique_ptr<InferenceInterface> inner_`, and forwards **every** method
  (including the new `state()`/`load()`) to `inner_`. Decorators subclass and
  override only what they augment.
- Accept: compiles; a passthrough decorator over a mock yields identical results.

> Wave 0 ordering: S0.1, S0.3, S0.4 have no deps (run first, 3 subagents). S0.2,
> S0.5, S0.6 follow once their deps land (3 subagents). If you prefer a flat wave,
> split into Wave 0a {S0.1,S0.3,S0.4} and Wave 0b {S0.2,S0.5,S0.6}.

---

## Wave 1 — Bridge: ModelRunner

### S1.1 — `ModelRunner` (Bridge abstraction)  · subagent A · deps: S0.2
- New: `backends/src/ModelRunner.hpp` + `backends/src/ModelRunner.cpp`
- Holds `std::unique_ptr<InferenceInterface> backend_`; exposes the high-level run
  API and delegates to the interface. Owns lifecycle orchestration (calls `load()`,
  observes `state()`), but contains **no** backend-specific code.
- Accept: unit test drives `ModelRunner` over a mock backend (load → ready → infer
  → results); failure path surfaces `Failed` state + exception.

> Wave 1 gate: after S0.2 + S1.1, the codebase fully works with the Bridge in
> place and nothing else changed behaviorally.

---

## Wave 2 — Per-backend Adapter cleanup (13 parallel steps)

The 13 `*Infer` classes already *are* Adapters over their vendor SDKs (they derive
from `InferenceInterface`). This wave makes the role explicit and isolates
SDK-specific calls, optionally routing trivial conversions through
`ITensorConverter`. **No two steps touch the same files** → 13 subagents in
parallel. Each step:

> For backend `X`: keep the class deriving from `InferenceInterface`, keep the
> constructor signature `(model_path, use_gpu, batch_size, input_sizes)`, isolate
> SDK calls, and **do not** alter device-selection or numeric logic. Build that
> backend + run its tests + clang-format.

| Step | Backend id | Files | subagent |
|---|---|---|---|
| S2.1 | OPENCV_DNN | `backends/opencv-dnn/src/OCVDNNInfer.{hpp,cpp}` | A |
| S2.2 | ONNX_RUNTIME | `backends/onnx-runtime/src/ORTInfer.{hpp,cpp}` | B |
| S2.3 | LIBTORCH | `backends/libtorch/src/LibtorchInfer.{hpp,cpp}` | C |
| S2.4 | LIBTENSORFLOW | `backends/libtensorflow/src/TFDetectionAPI.{hpp,cpp}` | D |
| S2.5 | TENSORRT | `backends/tensorrt/src/TRTInfer.{hpp,cpp}` | E |
| S2.6 | OPENVINO | `backends/openvino/src/OVInfer.{hpp,cpp}` | F |
| S2.7 | GGML | `backends/ggml/src/GGMLInfer.{hpp,cpp}` | G |
| S2.8 | TVM | `backends/tvm/src/TVMInfer.{hpp,cpp}` | H |
| S2.9 | MIGRAPHX | `backends/migraphx/src/MIGraphXInfer.{hpp,cpp}` | I |
| S2.10 | CACTUS | `backends/cactus/src/CactusInfer.{hpp,cpp}` | J |
| S2.11 | LLAMACPP | `backends/llamacpp/src/LlamaCppInfer.{hpp,cpp}` | K |
| S2.12 | EXECUTORCH | `backends/executorch/src/ExecuTorchInfer.{hpp,cpp}` | L |
| S2.13 | LITERT | `backends/litert/src/LiteRTInfer.{hpp,cpp}` | M |

- Per-step accept: `cmake -S . -B build -DDEFAULT_BACKEND=<ID> -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build && ctest --test-dir build` (only for backends whose deps are installed; otherwise configure-only).
- **Device-placement guard:** S2.2 (ORT) and S2.5 (TRT) must keep CUDA→ROCm probe,
  `NEURIPLO_ORT_EP` parsing, and the TRT `use_gpu=true` override byte-for-byte.

---

## Wave 3 — Decorators (4 parallel steps)

All depend only on S0.6 (`BackendDecorator`); independent of each other and of
Wave 2 → 4 subagents, **can overlap Wave 2**. Each is a new file pair; none
modifies existing code.

### S3.1 — `ProfilingBackend`  · subagent A
- New: `backends/src/decorators/ProfilingBackend.hpp(.cpp)`
- Moves timing (`start_timer`/`end_timer`, `last_inference_time_ms_`,
  `total_inferences_`) into a decorator around `get_infer_results`. Base-class
  timing stays for backward compat until Wave 6 cleanup.
- Accept: wraps a mock, reports nonzero time + increments count.

### S3.2 — `LoggingBackend`  · subagent B
- New: `backends/src/decorators/LoggingBackend.hpp(.cpp)`
- glog-based entry/exit logging of load + inference (shapes, duration). No new dep.
- Accept: emits expected log lines; results pass through unchanged.

### S3.3 — `CachingBackend`  · subagent C
- New: `backends/src/decorators/CachingBackend.hpp(.cpp)`
- Caches results keyed by a hash of input bytes + shapes; bounded LRU. Off by
  default to protect determinism expectations.
- Accept: identical input returns cached tuple; cache clear via `clear_cache()`.

### S3.4 — `QuantizedBackend`  · subagent D · **needs reviewer sign-off**
- New: `backends/src/decorators/QuantizedBackend.hpp(.cpp)`
- Optional input/output (de)quantization wrapper. **Borderline vs
  `inference-logic-change`**: must be opt-in, must not alter default numeric
  behavior of any backend, and must be flagged for human review per `REPO_META.yaml`.
- Accept: disabled = exact passthrough; enabled path covered by isolated test only.

---

## Wave 4 — Abstract Factory per backend (parallel, gated by Wave 2)

One concrete factory per compiled backend producing `{InferenceInterface,
IAllocator, ITensorConverter}`. Because only `DEFAULT_BACKEND` compiles, each
factory lives in its backend dir and is selected by the same `#ifdef USE_*`
mechanism.

- S4.1..S4.13 — `XRuntimeFactory` in each `backends/<x>/src/` (mirror Wave 2 table),
  implementing `IBackendRuntimeFactory`. deps: matching Wave 2 step + S0.5.
- Subagent fan-out identical to Wave 2 (one per backend).
- Accept per step: factory builds the same backend object the old path built;
  golden-output test unchanged.

---

## Wave 5 — State integration (gated by Wave 4)

### S5.1 — Separate `load()` from construction  · deps: Wave 2 · can split per-backend (×13)
- Edit each adapter so heavy loading runs in `load()` driven by `ModelRunner`,
  transitioning `Uninitialized→Loading→Ready/Failed`. Constructor stays cheap.
- Preserve the public contract by having `setup_inference_engine` call `load()`
  eagerly (preserving today's "constructed = ready" semantics).

### S5.2 — Replace `std::exit(1)` with `Failed` + exception  · deps: S5.1
- Edit `ORTInfer` (and any other backend using `std::exit`) to set `Failed` and
  throw `ModelLoadException`. Verify `vision-inference` tolerance first; if not,
  keep the facade translating to `nullptr` return.
- Accept: load failure is observable via `state()==Failed`; no process exit.

---

## Wave 6 — Wiring, CMake, tests, docs (mostly parallel)

### S6.1 — Factory dispatch in `setup_inference_engine`  · deps: Wave 4/5
- Edit `include/InferenceBackendSetup.hpp` + `src/InferenceBackendSetup.cpp` to
  build via `IBackendRuntimeFactory` + optional decorator chain, still returning
  `std::unique_ptr<InferenceInterface>`. Preserve TRT's `use_gpu=true` quirk.

### S6.2 — CMake: register new source files  · deps: Wave 0–4
- Add new `backends/src/*.hpp` (header-only mostly) and any new `.cpp` to the build
  in the relevant `cmake/*.cmake` modules and `backends/src` aggregation.

### S6.3 — Update `MockInferenceInterface` / `BackendTestTemplate`  · parallel
- `backends/src/MockInferenceInterface.hpp` currently mocks a `cv::Mat` overload
  that does not match the real `vector<vector<uint8_t>>` signature — align it to
  `InferenceInterface` and add `state()` mocking. (Pre-existing drift; fix here.)

### S6.4 — Decorator + ModelRunner + State unit tests  · parallel
- Add focused tests for the Bridge, each Decorator, and the State machine.

### S6.5 — Docs  · parallel · commit with `[skip ci]`
- Update `docs/ADDING_BACKEND.md` (new backend = derive `InferenceInterface` +
  provide a `XRuntimeFactory`), add an architecture section to `Readme.md` linking
  here.
- Do **not** touch `versions.env`/`docs/backends.yaml` (no new deps); if any
  `GEN:` source changes, run `python3 scripts/gen_backend_docs.py`.

---

## 4. Verification checklist (run at each wave gate)

```bash
# format gate (required before push)
clang-format --dry-run --Werror $(git diff --name-only --diff-filter=ACM '*.cpp' '*.hpp')

# default backend build + test
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure

# full backend matrix (deps permitting)
./scripts/test_backends.sh
```

Behavioral parity bar:
- `NEURIPLO_BACKEND_IDS` unchanged; `-DDEFAULT_BACKEND` unchanged.
- `setup_inference_engine` signature/return unchanged; `vision-inference` builds.
- ORT EP selection + CUDA→ROCm fallback + TRT GPU override identical.
- No new entries in `versions.env`; no new linked libraries.

---

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Breaking cross-repo `setup_inference_engine` contract | Return type stays `unique_ptr<InferenceInterface>`; facade preserves "constructed = ready"; CI-build `vision-inference` against the branch. |
| Accidental device-placement change (ORT/TRT) | Byte-for-byte preserve selection blocks; dedicated guard tests in S2.2/S2.5. |
| `QuantizedBackend` crossing `inference-logic-change` line | Opt-in only, exact passthrough default, mandatory human review. |
| Compile-time single-backend model vs runtime factory mismatch | Document that factory family is `#ifdef`-selected; no runtime registry added. |
| Removing `std::exit(1)` surprises callers | Verify downstream first; fall back to `nullptr` via facade if needed. |
| CMake source-list drift for new headers/cpp | S6.2 explicitly audits every new file into the build. |

---

## 6. Parallel-dispatch summary (for subagent orchestration)

| Wave | Parallel steps | Max subagents | Gate to start |
|---|---|---|---|
| 0a | S0.1, S0.3, S0.4 | 3 | none |
| 0b | S0.2, S0.5, S0.6 | 3 | 0a merged |
| 1 | S1.1 | 1 | 0b merged |
| 2 | S2.1–S2.13 | 13 | 1 merged |
| 3 | S3.1–S3.4 | 4 | 0b merged (can overlap Wave 2) |
| 4 | S4.1–S4.13 | 13 | 2 merged |
| 5 | S5.1 (×13) → S5.2 | 13 then 1 | 4 merged |
| 6 | S6.1–S6.5 | 5 | 4 & 5 merged |

Wave 3 has no dependency on Wave 2 and may run **concurrently with Wave 2** once
Wave 0b is merged — the largest available parallelization win.
