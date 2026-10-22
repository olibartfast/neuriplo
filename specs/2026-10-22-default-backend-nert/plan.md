# Plan — Default backend migration to NERT

## Group 0 — Runtime contract

- Keep `DEFAULT_BACKEND=NERT` in `CMakeLists.txt`.
- Leave `BackendRuntimeRegistry.cpp` in declaration order; rotate the compiled
  default to the front in `available_backend_ids`.
- Restore the base index-0 and `ids.front()` assertions in the C ABI tests.

## Group 1 — Build and test policy

- In `backends/nert/test/CMakeLists.txt`, remove the host-specific fixture
  fallback. Build `NertInferTest` unconditionally but register its CTest entry
  only when `NERT_FIXTURE` is set; print a STATUS message otherwise.
- Set CMP0077 NEW when available, and reject NERT on CMake < 3.16 with a clear
  message in `CMakeLists.txt`.

## Group 2 — CI and quality

- Add a `default-nert` CI job with no explicit backend and no fixture, building
  and running the full ctest with no OpenCV, CUDA, ONNX Runtime, or torch.
- Default `scripts/quality/sanitizers.sh` to NERT and update the clang-tidy
  example configure.

## Group 3 — Documentation and meta

- Update `Readme.md`, `CHANGELOG.md`, `docs/CODE_QUALITY.md`, `REPO_META.yaml`,
  and `specs/roadmap.md`; record this packet.
