---
name: docs-sync
description: Regenerate auto-generated GEN sections in docs/DEPENDENCY_MANAGEMENT.md from docs/backends.yaml and versions.env. Use whenever versions.env or docs/backends.yaml has been modified — before committing.
---

# Docs Sync

Keeps auto-generated `<!-- GEN:... -->` sections in `docs/DEPENDENCY_MANAGEMENT.md`
in sync with the sources of truth:

- `docs/backends.yaml` — backend metadata (names, scripts, paths, formats)
- `versions.env` — pinned dependency versions

## When to use

Run this skill after **any** change to `versions.env` or `docs/backends.yaml`,
and before committing. The pre-push hook blocks pushes with stale docs.

## Regenerate docs

```bash
python3 scripts/gen_backend_docs.py
```

## Check only (no modifications)

```bash
python3 scripts/gen_backend_docs.py --check
```

## What it updates

Six auto-generated sections in `docs/DEPENDENCY_MANAGEMENT.md`:

| GEN tag | Content |
|---|---|
| `backend-overview` | Backend ID, name, version, arch, GPU table |
| `setup-scripts-table` | Script → backend mapping |
| `cmake-dir-variables` | Per-backend CMake `*_DIR` variables |
| `cmake-version-variables` | Version variables from `versions.env` |
| `env-variables` | Exported environment variables block |
| `test-models-table` | Model format and source per backend |
