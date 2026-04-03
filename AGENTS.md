# Agent Instructions

- Repo-local agent metadata lives in `REPO_META.yaml`.
- Use `REPO_META.yaml` as the local source of truth for build/test entrypoints, owned paths, and allowed automated change classes.
- `develop` is the integration branch for normal work.
- `master` is release-only.
- Prioritize correctness, backend compatibility, dependency safety, device placement assumptions, fallback behavior, and performance regressions.
- Best practice: commit intentional, scoped changes before branch handoff.
- Best practice: push the working branch before starting branch-closure or integration steps.
- Best practice: after merging a feature branch into `develop`, push local `develop` to `origin/develop`, remove the merged feature branch locally and remotely, and update related docs and `Readme.md` when behavior or workflow changes.
