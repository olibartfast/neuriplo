# Plan — XSE backend candidate

**Packet:** `2026-11-18-xse-backend-candidate`
**Branch:** `feature/xse-backend-candidate`

Discovery + spike specification only. Each group ends in a written, reviewable result. No source change lands in this packet.

## Group 0 — Contract and inventory probe (read-only)

- [T-1] Extract the XSE bundle contract from upstream docs/repo: required files, StableHLO + Orbax layout, serving configs, symbolic-batch handling, `xsm-cli` inspection. Deliverable: contract note in `requirements.md` context or a linked probe note.
- [T-2] Map the neuriplo touchpoints a future backend would need: `BackendRegistry.cmake` ID properties, `docs/backends.yaml` fields, `versions.env` pin slot, setup-script/Docker/CI/test slots. Deliverable: gap list proving Phase-1-style metadata work is sized, not started.
- [T-3] Answer [Q-1]: identify one CMake-consumable distribution path (or record that none exists). Deliverable: recorded path or explicit no-go input.
  - Checks: upstream release artifacts + docs installation pages; no build required, record URLs + SHAs read.

## Group 1 — Device and boundary ruling

- [T-4] Reconcile XSE's CPU/CUDA/TPU-via-PJRT model with roadmap Phases 3–4 and the TF remote-TPU precedent. Deliverable: [Q-2] answered (maintainer call) and recorded in requirements.
- [T-5] Draw the serving boundary: what stays in neuriplo (`InferenceInterface` adapter over the C++ runtime) vs what stays in siblings (gRPC server, task pipelines). Deliverable: boundary paragraph + [D-3]/[D-4] confirmed or revised.
  - Checks: re-read `specs/mission.md` boundaries + `2026-11-09-tensorflow-gpu-remote-tpu` out-of-scope section.

## Group 2 — Spike specification (no implementation)

- [T-6] Specify the spike: backend ID proposal, accepted model path, construction via `EngineOptions`/`use_gpu` without new public fields, strict load-failure (no silent fallback), logging of requested vs actual device. Deliverable: spike spec section in plan/validation of the follow-up packet, not code.
- [T-7] Specify spike validation: CPU-only bundle load + infer through the shared contract, negative tests (bad bundle, missing device fails, never falls back), docs/ABI invariance checks. Deliverable: `V-`/`M-` list drafted for the follow-up packet.
- [T-8] Record go/no-go with reasons and the exact follow-up packet(s) (spike, then hardening). A no-go names the blocker ([Q-1]–[Q-4]) and the revisit trigger. Deliverable: decision paragraph in `validation.md` deviations/decision area.

## Acceptance

The packet is done when [R-1]–[R-6] are answered on paper, [Q-2] has a maintainer call, the spike shape is specified without code, and a reviewer can approve or kill the idea in one sitting.
