# Git hooks

**Recommended:** run `./scripts/quality/setup_hooks.sh` once per clone. That installs [pre-commit](https://pre-commit.com/) templates here and sets `core.hooksPath=.githooks`.

Until setup runs, the shell scripts `pre-commit` and `pre-push` provide a minimal fallback (format + docs + optional act cppcheck).

See [docs/CODE_QUALITY.md](../docs/CODE_QUALITY.md).
