# Running CI Locally with `act`

Use [nektos/act](https://github.com/nektos/act) to replay GitHub Actions workflows on your machine before pushing.

## Install

```bash
mkdir -p ~/.local/bin
curl -sL "https://github.com/nektos/act/releases/download/v0.2.88/act_Linux_x86_64.tar.gz" \
  | tar -xz -C ~/.local/bin act
# add ~/.local/bin to PATH if not already there
```

The repo ships an `.actrc` that pins the runner image and wires the Docker socket — no extra configuration needed.

## Usage

All commands run from the repo root.

### Inspect a job without executing it

```bash
act push --job build-executorch --dryrun
```

`--dryrun` prints every resolved step, action clone, and Docker command without running anything — useful for checking that YAML expressions expand to what you expect.

### Run a single job

```bash
act push --job build-executorch --verbose
```

`--verbose` streams each step's shell commands and their stdout as they execute, so you see exactly what the runner does.

### Run a matrix job (pick one cell)

Matrix jobs need an explicit filter because `act` can't run a partial matrix automatically:

```bash
# e.g. Clang-Tidy for GGML
act push --job clang-tidy \
  --matrix backend:GGML \
  --verbose
```

### Inspect the job graph

```bash
act push --graph
```

### List all jobs defined in the workflow

```bash
act push --list
```

## How it works

`act` spins up a `catthehacker/ubuntu:act-latest` container (≈500 MB) that mimics the GitHub-hosted runner. Composite actions (`actions/checkout@v4`, `docker/setup-buildx-action@v3`, `nick-fields/retry@v3`) are cloned from GitHub on first run and cached under `~/.cache/act`.

The container shares the **host Docker socket**, so any `docker build` / `docker run` inside the workflow writes to your local daemon — including the build cache. This is exactly the behaviour you want for validating Dockerfiles.

### Why `Free disk space` is skipped locally

`act` sets the environment variable `ACT=true` inside the runner. The `Free disk space` steps in `ci.yml` carry `if: ${{ !env.ACT }}` so they are skipped locally. On GitHub the variable is unset and the steps run normally.

## Quick reference: per-job direct Docker equivalents

For the build-only GPU/ExecuTorch jobs you can skip `act` entirely and just run the Docker build directly — this is faster and avoids pulling the runner image:

```bash
# ExecuTorch (build only, ~10-15 min first time)
docker buildx build \
  --file docker/Dockerfile.executorch \
  --tag neuriplo-executorch:ci \
  --load \
  .

# TensorRT (build only)
docker buildx build \
  --file docker/Dockerfile.tensorrt \
  --tag neuriplo-tensorrt:ci \
  --load \
  .
```

For jobs that also run tests (CPU backends), `act` is the right tool because it wires up the `docker run` step automatically.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: command not found` inside the runner | `.actrc` already mounts the host socket; ensure `docker` CLI is in `$PATH` on the host |
| `permission denied` on `/var/run/docker.sock` | Add your user to the `docker` group: `sudo usermod -aG docker $USER` |
| `--cache-from type=gha` warnings | Expected — GitHub Actions Cache is not available locally; act ignores the flag gracefully |
| Action clone fails with rate-limit | Set `GITHUB_TOKEN` in the environment: `act push --job … --secret GITHUB_TOKEN=$(gh auth token)` |
