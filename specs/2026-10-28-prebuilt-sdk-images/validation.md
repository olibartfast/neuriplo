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

## Combined SDK validation

- [x] Seven distinct SDK publishing jobs survive conflict resolution.
- [x] Matching source fallback fields survive for every opted-in matrix entry.
- [x] Real dependency pin names and SDK change-routing regressions pass locally.
- [x] Combined OpenVINO, GGML and ExecuTorch consumer builds pass locally.
- [x] Workflow lint and act checks pass for the combined tree.
- [ ] Hosted CI passes on the exact consolidated head before integration.
- [ ] The source heads of PRs 63 and 64 are reachable from develop before their
      branches are deleted.

These checks validate packaging and build wiring. TVM/GGML placeholder model
outputs are not evidence of real-weight inference. Full model/backend output
validation remains a separate release gate; AMD and Jetson runs stay deferred.
