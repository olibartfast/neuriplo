# Prebuilt SDK base images — validation

Acceptance defined before implementation:

- [ ] `docker/Dockerfile.litert` parses and the `BASE_IMAGE` default still
      selects `ubuntu:24.04` (inspect, plus `act … --dryrun` shows the arg
      wiring in all three consumer jobs).
- [ ] Stamp-skip path is real: `setup_litert.sh` against a directory that
      already carries the pinned stamp exits 0 without network access.
- [ ] Both workflows parse as YAML; `act push --job <job> --dryrun` resolves
      the new steps for a LITERT matrix cell.
- [ ] `ci.yml` fallback: when the GHCR pull fails, the build arg resolves to
      `ubuntu:24.04` and the image builds exactly as before (behavior
      unchanged for backends without `base_image`).
- [ ] `base-images.yml` uses registry-only caching and `packages: write`,
      pushes a version tag (never `latest`), and retries the build.
- [ ] Repo gates pass: `./scripts/quality/format.sh --check`, branch policy
      (`feature/*` → `develop`), and the remote CI matrix on the PR —
      including at least one LITERT job exercising the new steps.
- [ ] No inference, pin, or behavior change: `git diff` touches only the
      Dockerfile arg, workflows, matrix JSON, CHANGELOG, and this packet.
