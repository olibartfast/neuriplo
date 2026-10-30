# Prebuilt SDK base images (LiteRT pilot)

## Requirement

Per-PR CI jobs that need the LiteRT SDK (`clang-tidy`, `build-warnings`,
`build-and-test-cpu`) each rebuild `docker/Dockerfile.litert` from scratch:
clone of TensorFlow 2.19 plus a `tensorflow-lite` source build (10-20 min),
then the neuriplo build. Observed cost: ~28 min on success, ~25 min to fail
(3 whole-build retries against transient `gitlab.com` fetch errors), and the
exported BuildKit layers compete for the 10 GB GHA cache budget that also
protects the Windows vcpkg archives.

Consume a prebuilt, version-pinned `litert-deps` image from GHCR in CI, so a
PR run only compiles neuriplo itself (minutes). The same Dockerfile must keep
working for local builds without GHCR access. LiteRT is the pilot; the scheme
must extend to other source-built SDKs without redesign.

## Non-goals

- No change to inference code, backend behavior, dependency pins, or the
  `tensorflow-lite` build itself.
- No new third-party GitHub Actions (registry login via `docker login`).
- No mutable tags: the base image is addressed by the exact `LITERT_VERSION`
  pin, never `latest`.

## Consolidation scope (2026-10-29)

Extend the delivered LiteRT pilot to seven SDK bases: LiteRT, TVM, llama.cpp,
OpenVINO, GGML, LibTensorFlow and ExecuTorch. All retain the versions declared
in `versions.env`. SDK acquisition falls back to each Dockerfile's complete
source stage when a registry pull fails. Validate resolution against real pin
names, including `LIBTENSORFLOW` -> `TENSORFLOW_VERSION`.

Use the existing ExecuTorch feature branch/PR to integrate the remaining
OpenVINO and GGML changes. Preserve their commit ancestry; delete their branches
only after the combined head is integrated into develop with green CI. Keep
actual model inference claims separate from SDK build/install/reuse evidence.
