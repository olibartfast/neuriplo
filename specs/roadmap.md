# Neuriplo Roadmap

> Status: living brownfield roadmap, reviewed on 2026-08-29 against the current
> tree, recent history, and open repository work. Priorities after Phase 3 are a
> working sequence and should be confirmed as delivery evidence accumulates.

This roadmap is scoped to the Neuriplo backend-orchestration library. It favors
thin, independently reviewable phases over broad rewrites.

## Status Key

- **Complete** — the intended repository capability has landed; ongoing
  maintenance may remain.
- **Next** — the next phase to specify and implement.
- **Planned** — ordered but not yet specified as a feature packet.
- **Blocked** — cannot proceed without an identified decision or dependency.

## Roadmap Principles

- Preserve the existing public construction contract unless a feature spec
  explicitly approves and validates a migration.
- Treat runtime version changes as compatibility work, not routine bumps.
- Keep backend registration, setup, versions, generated docs, Docker metadata,
  tests, and CI synchronized.
- Make new runtime paths, providers, plugins, decorators, and fallback modes
  opt-in.
- Never silently fall back across devices, providers, backends, or model
  formats.
- Validate one observable slice before expanding scope.

## Phase 0 — Multi-Backend Foundation

**Status: Complete**

The current foundation includes a shared backend interface, runtime registry,
multi-backend builds, dependency-isolated plugins, typed raw outputs, explicit
lifecycle state, quality tooling, generated backend docs, ORT execution-provider
selection, ExecuTorch delegate workflows, and 14 registered backends including
the DALI preprocessing pipeline.

Ongoing defects in this surface are stabilization work and should be handled as
focused fixes rather than reopening this foundation phase.

## Phase 1 — Backend Metadata Consistency

**Status: Next**

Goal: make backend inventory drift fail early without changing runtime behavior.

Scope:

- Compare `docs/backends.yaml`, `cmake/BackendRegistry.cmake`, setup scripts,
  Docker metadata, test directories, and CI backend identifiers.
- Report the precise missing or conflicting property and owning backend.
- Keep non-selected backend dependencies ignored during normal configuration.
- Document the check and run it in the appropriate local and CI gates.

Exit criteria:

- Adding, removing, or renaming a backend cannot leave the maintained
  inventories silently inconsistent.
- The default `OPENCV_DNN` configure/build/test path and generated-doc check stay
  green.

## Phase 2 — Plugin ABI and Loader Hardening

**Status: Planned**

Goal: make runtime-loaded backends predictable under malformed and incompatible
plugin conditions.

Scope:

- Add focused coverage for ABI mismatch, missing symbols, malformed metadata,
  factory and inference errors, and output-release ownership.
- Verify a broken plugin does not destabilize compiled-in backends or other
  plugins.
- Document packaging layout, dependency discovery, compatibility, and one
  deployment example.

Exit criteria:

- Plugin failure and ownership behavior are executable contract tests.
- A consumer can build and load one documented backend plugin without relying
  on unwritten environment behavior.

## Phase 3 — Device and Fallback Contract

**Status: Planned**

Goal: resolve the public design before changing the shared interface.

Scope:

- Inventory device and accelerator semantics for all 14 registered backends:
  ORT providers, OpenVINO devices, LiteRT and ExecuTorch delegates,
  LibTorch/TensorRT/MIGraphX selection, llama.cpp/GGML offload, OpenCV DNN
  targets, TVM targets, TensorFlow placement, Cactus capabilities, and DALI.
- Specify backend-neutral `EngineOptions` fields for device,
  provider/delegate, and fallback policy.
- Define compatibility behavior for the legacy `bool use_gpu` overload.
- Define strict failure, explicit fallback, logging, and test expectations.
- Resolve the open follow-ups in
  `docs/plans/ort-execution-providers.md` without adding an unapproved SDK.

Exit criteria:

- A reviewed feature packet fixes the contract, compatibility rules, and proof
  required for implementation.
- No code or new dependency is introduced as part of the design-only phase.

## Phase 4 — Opt-In Device Semantics

**Status: Planned; depends on Phase 3**

Goal: implement the approved device and fallback contract without breaking
existing callers.

Scope:

- Extend `EngineOptions` in small backend-family increments.
- Preserve CPU defaults and the legacy overload.
- Add provider/delegate parsing, unavailable-device errors, explicit fallback,
  and selection logging tests with each increment.
- Keep SDK acquisition and hardware-specific enablement in separately approved
  feature packets.

Exit criteria:

- Existing consumers retain their current behavior.
- Requested placement and actual selection are observable.
- No backend silently moves work to CPU.

## Phase 5 — Contract Tests and Model Fixtures

**Status: Planned**

Goal: compare shared behavior across backend families while keeping vendor and
hardware constraints explicit.

Scope:

- Separate contract, backend smoke, model-format, device/provider, and
  performance tests.
- Standardize small deterministic classification and object-detection fixtures
  where supported.
- Add quantized fixtures for runtimes that require or materially benefit from
  them.
- Record manual validation for x86_64, ARM64, NVIDIA GPU, AMD ROCm, and
  hardware-only NPU targets.

Exit criteria:

- Every supported backend has at least one deterministic declared smoke path.
- Metadata shapes and types, batching, invalid inputs, unsupported formats, and
  failure behavior have backend-appropriate coverage.
- Hardware-only checks include exact commands and expected results.

## Phase 6 — Consumer and Release Readiness

**Status: Planned**

Goal: make integration and releases repeatable for `neuriplo-infer` and other
consumers.

Scope:

- Document construction, errors, metadata, buffers, batching, lifecycle, and
  thread-safety as the stable consumer contract.
- Add compiled-in and plugin-backed selection examples.
- Define stable, experimental, deprecated, and hardware-only support tiers.
- Add a release checklist covering the backend matrix, generated docs,
  dependency versions, exclusions, migration notes, and rollback information.

Exit criteria:

- Consumers can select a backend and supported device path without relying on
  undocumented behavior.
- A release can move from `develop` to `master` with a recorded validation
  trail and understandable compatibility notes.

## Operating Rule

Before beginning an incomplete phase, create a dated feature directory:

```text
specs/YYYY-MM-DD-feature-name/
├── requirements.md
├── plan.md
└── validation.md
```

Write validation before implementation. Update the feature packet, this
roadmap, and any durable constitution decision in the same branch as the code
when discovery changes the intended contract.
