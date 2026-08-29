# Neuriplo Mission

> Status: working brownfield constitution, reconstructed on 2026-08-29 from
> the current code, documentation, repository metadata, and recent history.
> Product assumptions that still need maintainer confirmation are listed below.

## Why Neuriplo Exists

Neuriplo gives C++ inference applications one stable backend-orchestration
layer across vision, graph, GGUF-native generative, and GPU preprocessing
runtimes. Consumers should be able to select the runtime that fits their model,
hardware, compatibility, and performance needs without rewriting their
application around each vendor SDK.

The library is primarily used by `neuriplo-infer`, but its public interfaces
should remain usable by other C++ consumers.

## Who It Serves

- Application developers who need to run models through a consistent C++ API.
- Maintainers who must add, upgrade, test, and release native inference
  backends without destabilizing existing ones.
- Backend contributors integrating vendor runtimes, device providers,
  delegates, plugins, or preprocessing pipelines.
- Operators who need backend selection, device placement, fallback behavior,
  and failures to be explicit and diagnosable.

## Product Promise

Neuriplo provides:

- a common construction and inference contract across supported backends;
- compile-time, multi-backend, and runtime-plugin selection paths;
- backend metadata and typed output buffers that consumers can interpret
  without vendor-specific object types;
- opt-in acceleration and extension paths that preserve established defaults;
- clear failure behavior instead of silent device, provider, model-format, or
  backend substitution;
- repeatable setup, validation, and documentation for supported runtime
  combinations.

## Success Criteria

The project is succeeding when:

- existing consumers keep working across compatible releases unless an API
  migration is deliberately specified and documented;
- every supported backend can be configured, built, and exercised through the
  shared contract on at least one declared platform;
- backend selection and device placement are observable, and fallback happens
  only when the caller explicitly allows it;
- dependency upgrades and new backends fail early when versions, libraries,
  CMake metadata, scripts, Docker configuration, tests, or docs disagree;
- failures return or report actionable context without terminating the host
  process unexpectedly;
- performance-sensitive changes include regression evidence appropriate to the
  affected runtime and hardware;
- a new maintainer or coding agent can discover the project boundaries and run
  the required checks from version-controlled files.

## Boundaries and Non-Goals

- Neuriplo orchestrates inference runtimes; it does not replace vendor SDKs or
  hide their real model-format and hardware constraints.
- Task-specific pre/postprocessing, serving APIs, capture, and product UI belong
  in consumer or sibling projects. DALI is intentionally supported as a GPU
  preprocessing pipeline because it occupies the same composable backend slot.
- The project does not promise that every backend supports every model,
  architecture, datatype, accelerator, or plugin mode.
- Dependency expansion, public interface migration, inference-logic changes,
  and performance-critical kernel changes require explicit human review; they
  are not routine automated maintenance.

## Engineering Values and Tone

Correctness, backend compatibility, dependency safety, device-placement
semantics, fallback behavior, and performance regressions take priority over
surface-level uniformity. Documentation should be direct, technical, and
operational: state supported paths, known exclusions, exact commands, and
failure behavior without implying coverage that has not been validated.

## Open Product Questions

- Quantitative latency, throughput, memory, and binary-size targets are not yet
  defined across backend families.
- Support tiers such as stable, experimental, deprecated, and hardware-only are
  described informally but do not yet have repository-wide entry/exit rules.
- The compatibility policy for future public API migrations needs a more formal
  deprecation window.

Until these are resolved, feature specifications must state their own measurable
performance and compatibility requirements rather than inventing global targets.
