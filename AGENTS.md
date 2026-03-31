# Agent Instructions

- Repo-local agent metadata lives in `REPO_META.yaml`.
- Use `REPO_META.yaml` as the local source of truth for build/test entrypoints, owned paths, and allowed automated change classes.
- `develop` is the integration branch for normal work.
- `master` is release-only.
- Prioritize correctness, backend compatibility, dependency safety, device placement assumptions, fallback behavior, and performance regressions.
