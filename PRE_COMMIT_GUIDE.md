# Pre-commit and Git Hooks

This project uses [pre-commit](https://pre-commit.com/) for fast checks on commit and stricter checks on push. See [docs/CODE_QUALITY.md](docs/CODE_QUALITY.md) for the full toolchain (clang-format, cppcheck, clang-tidy, ASan/UBSan).

## Recommended setup

```bash
./scripts/quality/setup_hooks.sh
source .venv/bin/activate   # pre-commit CLI lives in the project venv
```

This will:

1. Create `.venv` and install `pre-commit` from `requirements-dev.txt`
2. Install hook templates into `.githooks/`
3. Set `git config core.hooksPath .githooks`
4. Register commit and pre-push hooks

## What runs when

### On `git commit`

- Trim trailing whitespace, fix EOF newlines
- Validate YAML and large files
- **clang-format** — auto-formats staged `*.cpp` / `*.hpp` under `src/`, `include/`, `backends/`
- **cppcheck** — static analysis on staged C++ files

If a hook modifies files, re-stage and commit again.

### On `git push`

- **clang-format** full-tree check (`--dry-run --Werror`, CI parity)
- **cppcheck** full tree (`src/`, `backends/`)
- **Backend docs** — `scripts/gen_backend_docs.py --check`

## Manual usage

```bash
pre-commit run --all-files              # all commit-stage hooks
pre-commit run --hook-stage pre-push    # push-stage hooks without pushing
./scripts/quality/run.sh                # format + cppcheck without git
./scripts/quality/run.sh --fix          # format in place
```

## Updating hook versions

```bash
pre-commit autoupdate
```

## Skipping hooks

```bash
git commit --no-verify
git push --no-verify
SKIP=cppcheck-full-tree git push    # skip one hook (pre-commit SKIP env)
```

## Troubleshooting

### `command not found: pre-commit`

```bash
pip install -r requirements-dev.txt
# ensure ~/.local/bin is on PATH
```

### clang-format version mismatch

CI uses `clang-format-18`. Install `clang-format-18` locally; `scripts/quality/format.sh` prefers it automatically.

### cppcheck not installed

```bash
sudo apt install cppcheck
```

### Hooks not running

Confirm: `git config core.hooksPath` → `.githooks` and `ls -la .githooks/pre-commit` is executable.

Re-run `./scripts/quality/setup_hooks.sh` if hooks were overwritten or missing.
