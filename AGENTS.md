# Agent Instructions

- Repo-local agent metadata lives in `REPO_META.yaml`.
- Use `REPO_META.yaml` as the local source of truth for build/test entrypoints, owned paths, and allowed automated change classes.
- The project constitution lives in `specs/mission.md`, `specs/tech-stack.md`,
  and `specs/roadmap.md`. Read it before planning new or actively changing
  feature work.
- For a non-trivial new roadmap phase, create
  `specs/YYYY-MM-DD-feature-name/{requirements.md,plan.md,validation.md}` before
  implementation. Do not backfill feature packets for completed work. Trivial
  fixes and documentation edits do not require a three-file packet.
- Keep feature requirements, plan, validation, code, and roadmap status in sync
  in the same branch whenever implementation changes the intended contract.
- `develop` is the integration branch for normal work.
- `master` is release-only. CI runs on pushes and PRs for `master`, `develop`, `release/**`, and `feature/**`.
- All new feature work — every new branch AND every new git worktree — MUST be created from `develop`, never from `master`. `master` lags behind `develop` and only receives release PRs; branching a feature off `master` produces a tree missing the latest backends and integration work. When a worktree tool defaults to the repo's main branch (`master`), reset it onto `develop` before starting work.
- Prioritize correctness, backend compatibility, dependency safety, device placement assumptions, fallback behavior, and performance regressions.
- Best practice: commit intentional, scoped changes before branch handoff.
- Best practice: push the working branch before starting branch-closure or integration steps.
- Best practice: after merging a feature branch into `develop`, push local `develop` to `origin/develop`, remove the merged feature branch locally and remotely, and update related docs and `Readme.md` when behavior or workflow changes.
- After completing a `release/*` or `hotfix/*` flow, delete the finished branch
  locally and on `origin`; see `.cursor/rules/gitflow-release-cleanup.mdc`.
- When committing documentation-only changes, include `[skip ci]` in the commit message.
- **Hyperlink verification:** When editing `Readme.md` or any documentation with hyperlinks, verify all relative links resolve to existing files and absolute GitHub URLs are reachable. Prefer absolute GitHub blob/tree URLs over fragile cross-repo relative paths.
- Keep `Readme.md` as a general-purpose project entrypoint. Put backend-specific setup, model-format, Docker, build, and troubleshooting details in the appropriate docs section, such as `docs/DEPENDENCY_MANAGEMENT.md` or a backend-specific guide, and link from the README only when the link is broadly useful.
- When debugging CI failures, build errors, or test failures, consult `docs/TROUBLESHOOTING.md` for known patterns and hard-won lessons before starting from scratch.
- Code quality: see `docs/CODE_QUALITY.md`. Fast local gate: `./scripts/quality/run.sh`. One-time hook setup: `./scripts/quality/setup_hooks.sh` (installs pre-commit into `.githooks/`). ASan+UBSan: `./scripts/quality/sanitizers.sh` or `-DSANITIZERS=ON` with Debug build.
- Before pushing any code changes, run `./scripts/quality/format.sh --check` (or `pre-commit run --all-files` after hook setup). Pre-push hooks also run full-tree cppcheck and backend-docs sync.
- When changing `versions.env` or `docs/backends.yaml`, always run `python3 scripts/gen_backend_docs.py` before committing to keep auto-generated `GEN:` sections in `docs/DEPENDENCY_MANAGEMENT.md` in sync. The pre-push hook also enforces this.
- Before pushing Dockerfile or workflow changes, validate locally with `act`: `act push --job <job-id> --dryrun` to inspect resolved steps, then `act push --job <job-id> --verbose` for a full run. See `docs/LOCAL_CI.md` for setup and per-job examples.
