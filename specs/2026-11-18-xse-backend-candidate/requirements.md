# Requirements — XSE backend candidate

Spec: `specs/2026-11-18-xse-backend-candidate/requirements.md` · Branch: `feature/xse-backend-candidate` · Roadmap: proposed backend intake (does not displace Phase 1 as `Next`)

**Date:** 2026-11-18
**Status:** Specified, discovery only — no implementation in this packet

## Goal

Decide whether XLA Serving Engine (XSE, `openxla/xla-serving`, Apache-2.0) fits neuriplo as an **opt-in** backend for JAX-exported StableHLO bundles, and if so, specify the smallest spike that proves it without changing defaults, public contracts, or device semantics.

Sources: [launch post](https://opensource.googleblog.com/2026/11/supercharging-xla-inference-xla-serving-engine.html), [docs](https://xla-serving.readthedocs.io/en/latest/). XSE pitch: "Train in Python, Serve in C++" via XLA SavedModel (StableHLO + Orbax checkpoint + host pre/post-processing), C++ runtime over PJRT, gRPC + in-process `ModelRunner` + embeddable `Interpreter`, Docker images for CPU/GPU/TPU, dynamic batching/warmup/AOT. Targets: recommenders/rankers, vision/multimodal encoders, encoder LMs, short fixed-horizon generation. Explicit non-target: open-ended autoregressive LLMs (directs to vLLM/SGLang).

## In Scope

- [R-1] Record the XSE export/serve contract relevant to neuriplo: bundle layout, supported entry points (`Predict`/`GetModelMetadata`), and the JAX (`xla_saved_model`) and torchax export paths.
- [R-2] Assess fit against the constitution: backend-orchestration role, explicit selection/fallback, model-format honesty, and the serving-API boundary (`specs/mission.md`).
- [R-3] Assess build/packaging fit: how XSE (Bazel-built C++ upstream) could be consumed from CMake without a second manual inventory, with versions pinned in `versions.env` and metadata in `docs/backends.yaml` (`specs/tech-stack.md`).
- [R-4] Assess device-contract fit: CPU default, opt-in GPU/TPU, no silent CPU fallback, against roadmap Phases 3–4 and the TF-packet precedent that in-process TPU/PJRT is out of scope for this repo (`specs/2026-11-09-tensorflow-gpu-remote-tpu/requirements.md`).
- [R-5] Specify the smallest opt-in spike shape (compiled-in vs `dlopen` plugin, new backend ID, model-format handling for StableHLO bundles) with no change to `DEFAULT_BACKEND`, legacy overload, C++ API, consumer C ABI, or plugin ABI.
- [R-6] End with a recorded go/no-go and, on go, the follow-up packet(s) required before any backend landing.

## Out of Scope

- Landing an `XSE` backend, touching `backends/`, `cmake/BackendRegistry.cmake`, Docker images, CI jobs, or `versions.env` pins. This packet is discovery + spike specification only.
- Changing `DEFAULT_BACKEND`, `EngineOptions`, the legacy `setup_inference_engine` overload, either ABI, or any device/fallback default.
- In-process TPU execution claims. Remote-TPU precedent stands; any in-process TPU path needs its own approval.
- LLM continuous-batching serving, TorchTPU export, AOT, and C++ preprocessing promises from the XSE roadmap — noted as upstream futures, not neuriplo commitments.
- Performance claims. XSE's 2–5x figures are upstream's; no neuriplo parity target is set here (per `specs/mission.md` open questions).

## Decisions

- [D-1] Discovery before code. No source change in this packet; validation is document/probe review only.
- [D-2] Opt-in or nothing. If XSE ever lands, it lands behind explicit selection like every other non-default backend — Rationale: `specs/tech-stack.md` device/fallback assumptions and mission explicit-selection promise.
- [D-3] Plugin-first bias for the spike, to keep XSE's dependency closure isolated — Rationale: plugin ABI exists for exactly this (`specs/tech-stack.md`); final call belongs to the follow-up packet after the probe.
- [D-4] StableHLO bundles are a new model format, not a reinterpretation of SavedModel/ONNX. Any spike must name the accepted path/extension explicitly and reject everything else at load.

## Constraints

- `cmake/BackendRegistry.cmake` is the backend-ID authority; `versions.env` pins versions; `docs/backends.yaml` is the single human-readable inventory with `GEN:` regeneration (`python3 scripts/gen_backend_docs.py`). No second inventory.
- New dependencies need explicit approval + compatibility review (`specs/tech-stack.md` explicit non-choices). This packet gathers the review inputs; it is not the approval.
- C++17, CMake 3.10+ (3.16+ for NERT default path). Public headers and both ABIs unchanged.
- Phase 1 (metadata consistency) stays `Next`; Phase 3 stays design-only. This packet adds no implementation that would jump that queue.

## Dependencies

- Upstream: `openxla/xla-serving` repo, docs, Docker image list — external, read-only in this packet.
- Repo: `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`, `cmake/BackendRegistry.cmake`, `docs/backends.yaml`, `versions.env` — read-only in this packet.

## Context

- neuriplo is the backend layer of `neuriplo-infer` with 14 registered backends; serving APIs and task pre/post-processing live in siblings (DALI is the deliberate preprocessing-pipeline exception) — `specs/mission.md:68-77`, `specs/tech-stack.md:19-23`.
- The 2026-11-09 TF packet fixed `LIBTENSORFLOW` `use_gpu` handling and deliberately kept in-process TPU/PJRT, StableHLO, and `libtpu` out of this repo. XSE re-opens exactly that question, so the precedent is load-bearing.
- XSE upstream scale proof cited: 6M QPS internally (Shorts/Ads/Search/parts of Gemini Omni); private-preview partners Arm, Intel, Moloco. Finding, not a neuriplo claim.

## Assumptions & Open Questions

- [A-1] XSE remains Apache-2.0 and its docs quickstart reflects the consumable interface — Basis: launch post + docs landing page; confirm at probe time.
- [Q-1] (blocks any landing) Can a pinned XSE release be consumed from CMake (prebuilt lib, FetchContent, or plugin `.so`) without Bazel in the neuriplo build? Owner: backend contributor. Probe in Group 0.
- [Q-2] (blocks scope) Is in-process TPU execution in scope for neuriplo at all, or does the remote-only precedent hold and XSE-TPU stay a `neuriplo-infer` concern? Owner: maintainer. Must be answered before any spike beyond CPU.
- [Q-3] What is the exact XLA SavedModel directory contract (required files, versioning, AOT artifacts) the spike must validate at load? Owner: spike author via upstream docs/repo.
- [Q-4] Compiled-in vs plugin: does the XSE link closure (PJRT/XLA/CUDA/`libtpu`) force plugin isolation? Owner: spike author via Group 0 probe.

## Definition of Done (requirements level)

- [ ] Every [R-n] is answered in this packet or explicitly deferred with a tracked location
- [ ] No In Scope item silently dropped; no Out of Scope implementation smuggled in
