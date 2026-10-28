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
