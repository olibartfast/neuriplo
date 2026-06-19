# Versioning and Changelog

## Overview

This project uses two files to track releases:

| File | Purpose |
|------|---------|
| `VERSION` | Single source of truth for the project version (read by CMake) |
| `CHANGELOG.md` | Human-readable history of notable changes per release |
| `versions.env` | Backend dependency versions (ONNX Runtime, TensorRT, etc.) — separate from project version |

## VERSION file

Contains a single line like `0.2.0-dev`.

- The `-dev` suffix indicates unreleased development work on `develop`.
- CMake reads this file at configure time and strips the suffix to set `project(neuriplo VERSION X.Y.Z)`.
- Follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

## CHANGELOG.md

Follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

Sections per release:
- **Added** — new features
- **Changed** — changes to existing functionality
- **Fixed** — bug fixes
- **Removed** — removed features
- **Deprecated** — features marked for future removal

Unreleased work goes under the `[Unreleased]` heading at the top.

## Day-to-day workflow

When merging a PR into `develop`, add a line under `[Unreleased]` in the appropriate section. Example:

```markdown
## [Unreleased]

### Added
- MIGraphX backend support
```

## Release workflow

1. **Create a release branch** from `develop`:
   ```
   git checkout -b release/0.2.0 develop
   ```

2. **Update VERSION** — remove the `-dev` suffix:
   ```
   0.2.0
   ```

3. **Update CHANGELOG.md** — rename `[Unreleased]` to the new version with today's date, and add a fresh empty `[Unreleased]` section:
   ```markdown
   ## [Unreleased]

   ## [0.2.0] - 2026-04-15

   ### Added
   - ...
   ```
   Update the comparison links at the bottom:
   ```markdown
   [Unreleased]: https://github.com/olibartfast/neuriplo/compare/v0.2.0...HEAD
   [0.2.0]: https://github.com/olibartfast/neuriplo/compare/v0.1.0...v0.2.0
   ```

4. **Validate and push the release branch**:
   ```
   ./scripts/quality/format.sh --check
   ./scripts/quality/run.sh
   git push -u origin release/0.2.0
   ```

5. **Open and merge a PR into `master`**, then tag the merged `master` commit:
   ```
   gh pr create --base master --head release/0.2.0 --title "Release 0.2.0"
   gh pr merge <pr-number> --merge --delete-branch
   git switch master
   git pull
   git tag v0.2.0
   git push origin v0.2.0
   ```

6. **Back-merge to `develop` and delete the release branch**:
   ```
   git switch develop
   git pull
   git merge master
   git push origin develop
   git branch -d release/0.2.0
   git fetch --prune origin
   ```

7. **Start the next development cycle** on `develop` when needed by setting
   `VERSION` to the next `X.Y.Z-dev`, committing, and pushing.
