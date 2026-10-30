# Prebuilt SDK base images — plan

## Design

1. `docker/Dockerfile.litert` gains `ARG BASE_IMAGE=ubuntu:24.04` and
   `FROM ${BASE_IMAGE} AS litert-deps`. `scripts/setup_litert.sh` already
   exits 0 without rebuilding when `$LITERT_DIR` carries a matching version
   stamp, so pointing the arg at a prebuilt image turns the expensive `RUN`
   into a no-op while local builds keep compiling from source.
2. New `base-images.yml` workflow builds `--target litert-deps` (default
   Ubuntu base) and pushes
   `ghcr.io/olibartfast/neuriplo/litert-deps:<LITERT_VERSION>`. Triggers:
   push touching the Dockerfile, the setup script, the stamp helper,
   `versions.env`, or the workflow itself; weekly schedule (fires on the
   default branch once merged there); manual dispatch. Its own layer cache
   lives in the registry (`type=registry,…:buildcache`), never in GHA, so
   base builds add zero pressure to the 10 GB budget the Windows archives
   depend on.
3. `ci.yml` (`clang-tidy`, `build-warnings`, `build-and-test-cpu`) resolves
   the base per run: GHCR login, `docker pull` the pinned tag, fall back to
   `ubuntu:24.04` (full source build) when the pull fails, and pass
   `--build-arg BASE_IMAGE=<ref>`. The matrix opt-ins via an optional
   `base_image` field in `scripts/ci/backend_matrices.json`, so other
   backends are unaffected until they adopt the scheme.
4. `CHANGELOG.md` records the change under `[Unreleased]` / `Added`.

## Ordering

The fallback keeps every run green before the first base image exists: this
PR's own CI may still take the slow path, the `base-images.yml` push build
publishes the image, and later runs go fast. No flag day.

## Extension

A new backend adopts the scheme with: a `*-deps` stage honoring the same
`BASE_IMAGE` arg plus stamp skip (or equivalent no-op), one matrix field,
and one job in `base-images.yml`.

## Consolidation order

1. Merge current develop into the existing ExecuTorch SDK branch.
2. Merge OpenVINO and GGML feature heads, preserving every publishing job,
   matrix field, fallback and resolver regression.
3. Run resolver/routing tests, format/docs gates, workflow validation and act.
4. Build the three combined consumers against matching SDK images; the
   ExecuTorch SDK must configure, link and run an installed consumer after its
   source tree has been removed.
5. Update PR 66 to describe the complete remaining SDK scope, monitor its exact
   head CI, merge into develop, verify all three source heads are ancestors,
   and then remove the integrated feature branches.
