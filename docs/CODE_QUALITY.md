# Code Quality

Local tooling mirrors the CI jobs in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml): formatting, static analysis (cppcheck, clang-tidy), warnings-as-errors, and AddressSanitizer + UndefinedBehaviorSanitizer (ASan/UBSan).

## Quick setup

```bash
# Install pre-commit hooks into .githooks/ (recommended; creates .venv)
./scripts/quality/setup_hooks.sh
source .venv/bin/activate

# Or manual venv + hooks
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
pre-commit install --hook-type pre-push
```

With `core.hooksPath=.githooks` (set by `setup_hooks.sh`):

| Event | Checks |
|-------|--------|
| `git commit` | trailing whitespace, YAML, **clang-format** (auto-fix staged), **cppcheck** on staged C++ |
| `git push` | full-tree **clang-format --dry-run**, **cppcheck** full tree, **backend docs** GEN sync |

Skip hooks when needed: `git commit --no-verify` or `SKIP=clang-format-check-all git push`.

## Run checks manually

```bash
# Fast (matches most pre-push checks)
./scripts/quality/run.sh

# Auto-format all C++ sources
./scripts/quality/run.sh --fix

# Everything you can run locally without Docker
./scripts/quality/run.sh --all

# Individual tools
./scripts/quality/format.sh --check    # or --fix
./scripts/quality/cppcheck.sh
./scripts/quality/clang_tidy.sh build  # needs compile_commands.json
./scripts/quality/sanitizers.sh        # ASan+UBSan build + ctest (default OPENCV_DNN)
```

### clang-tidy

Requires a configured build with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`:

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build
./scripts/quality/clang_tidy.sh build
```

CI runs clang-tidy per backend inside Docker images (see `clang-tidy` job in `ci.yml`).

### Sanitizers (ASan + UBSan)

CMake flag `-DSANITIZERS=ON` adds `-fsanitize=address,undefined` (see `cmake/SetCompilerFlags.cmake`).

```bash
./scripts/quality/sanitizers.sh
./scripts/quality/sanitizers.sh --backend OPENCV_DNN --build-dir build-asan
```

Use Debug builds. Some backends are excluded in CI sanitizers (LibTensorFlow, OpenVINO TBB/ASan conflict) — see `sanitizers` job comments in `ci.yml`.

Environment knobs:

```bash
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
```

### Werror

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DWERROR=ON
cmake --build build
```

## CI parity

| Local script / hook | CI job |
|---------------------|--------|
| `format.sh` | `format-check` |
| `cppcheck.sh` | `cppcheck` |
| `clang_tidy.sh` | `clang-tidy` (per backend, Docker) |
| `sanitizers.sh` | `sanitizers` (per backend, Docker) |
| `-DWERROR=ON` | `build-warnings` |

Heavy per-backend jobs are also runnable via [act](LOCAL_CI.md):

```bash
act push --job clang-tidy --matrix backend:OPENCV_DNN --verbose
act push --job sanitizers --matrix backend:OPENCV_DNN --verbose
```

## Configuration files

| File | Purpose |
|------|---------|
| `.clang-format` | LLVM style formatting |
| `.clang-tidy` | clang-tidy check profile (`HeaderFilterRegex`) |
| `.pre-commit-config.yaml` | Git hook definitions |
| `cmake/SetCompilerFlags.cmake` | `SANITIZERS`, `WERROR` options |

## Legacy `.githooks/`

If you use `core.hooksPath=.githooks` without `setup_hooks.sh`, the shell hooks in `.githooks/pre-push` still run clang-format and docs checks. Prefer `./scripts/quality/setup_hooks.sh` so **pre-commit** manages hooks consistently.

Optional Docker cppcheck via act (legacy): `SKIP_ACT=1` skips it when pushing.
