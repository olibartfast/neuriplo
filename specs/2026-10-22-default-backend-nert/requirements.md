# Requirements — Default backend migration to NERT

Date: 2026-10-22. Branch: `wip/default-backend-nert`, base `develop` 6282d0e.
Roadmap: Phase N3, `specs/roadmap.md`.

## Decisions

- [D-1] `develop` defaults to `NERT`. A bare `cmake -S . -B build` fetches NERT
  at `NERT_VERSION` (CMake 3.16+), builds it with no vendor SDK, and needs no
  OpenCV.
- [D-2] `OPENCV_DNN` is an explicit opt-out, selected with
  `-DDEFAULT_BACKEND=OPENCV_DNN` plus `libopencv-dev`. It is no longer the
  default and is not part of the default build.
- [D-3] The compiled default is listed first by the runtime backend list. The
  registry keeps its declaration order; `available_backend_ids` rotates the
  default id to the front after collecting built-ins and plugins.

## Requirements

- [R-1] `available_backend_ids` returns the compiled default at index 0 in
  single- and multi-backend builds. The C ABI contract tests assert the base
  behavior (`neuriplo_backend_list_get(list, 0)` and `ids.front()`).
- [R-2] A fixture-free default configure builds `NertInferTest` but registers no
  CTest entry, and the full default ctest passes. With `-DNERT_FIXTURE=<path>`
  the entry is registered and fails, never skips, if the fixture is unusable.
- [R-3] Enabling NERT with CMake < 3.16 fails with a clear `FATAL_ERROR`. The
  `BUILD_INFERENCE_ENGINE_TESTS` shadowing in `cmake/Nert.cmake` emits no
  CMP0077 developer warning.
- [R-4] Quality scripts and docs default to NERT and do not require OpenCV;
  explicit OpenCV paths still work. No new dependency is introduced.
- [R-5] CI has a small backend-free default NERT configure/build/test job, and
  the existing NERT parity job remains.
- [R-6] Readme, CHANGELOG, REPO_META, and roadmap record the default change, the
  cached-build migration, the fixture-conditional test, and NERT's
  correctness-first (not performance) status.

## Out of scope

- NERT performance work, NERT pin changes, engine/adapter changes, and
  reordering `BackendRuntimeRegistry`.
