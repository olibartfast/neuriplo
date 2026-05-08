# Agent Instructions

- Repo-local agent metadata lives in `REPO_META.yaml`.
- Use `REPO_META.yaml` as the local source of truth for build/test entrypoints, owned paths, and allowed automated change classes.
- `develop` is the integration branch for normal work.
- `master` is release-only.
- Prioritize correctness, backend compatibility, dependency safety, device placement assumptions, fallback behavior, and performance regressions.
- Best practice: commit intentional, scoped changes before branch handoff.
- Best practice: push the working branch before starting branch-closure or integration steps.
- Best practice: after merging a feature branch into `develop`, push local `develop` to `origin/develop`, remove the merged feature branch locally and remotely, and update related docs and `Readme.md` when behavior or workflow changes.
- When committing documentation-only changes, include `[skip ci]` in the commit message.
- Keep `Readme.md` as a general-purpose project entrypoint. Put backend-specific setup, model-format, Docker, build, and troubleshooting details in the appropriate docs section, such as `docs/DEPENDENCY_MANAGEMENT.md` or a backend-specific guide, and link from the README only when the link is broadly useful.
- Before pushing any code changes, run `clang-format --dry-run --Werror` on all `.cpp`/`.hpp` files. The pre-push hook in `.githooks/pre-push` does this automatically — activate it once per clone with: `git config core.hooksPath .githooks`
- Before pushing Dockerfile or workflow changes, validate locally with `act`: `act push --job <job-id> --dryrun` to inspect resolved steps, then `act push --job <job-id> --verbose` for a full run. See `docs/LOCAL_CI.md` for setup and per-job examples.
