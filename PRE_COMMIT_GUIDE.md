# Pre-commit Configuration Guide

This project uses [pre-commit](https://pre-commit.com/) to manage and maintain multi-language pre-commit hooks. These hooks help identify simple issues before submission to code review.

## 1. Prerequisites

Ensure you have Python and `pip` installed on your system.

```bash
python3 --version
pip3 --version
```

## 2. Installation

We recommend installing `pre-commit` in a virtual environment to avoid conflicts with system packages.

### Setting up a Virtual Environment

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Installing pre-commit

With the virtual environment activated:

```bash
pip install pre-commit
```

## 3. Configuration

The configuration is managed in the `.pre-commit-config.yaml` file at the root of the repository.

Example configuration:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
```

## 4. Setting up the Git Hook

To set up the git hook scripts, run:

```bash
pre-commit install
```

You should see: `pre-commit installed at .git/hooks/pre-commit`.

Now, `pre-commit` will run automatically on `git commit`!

## 5. Usage

### Automatic Usage
When you run `git commit`, pre-commit will check the staged files.
- If all hooks pass, the commit proceeds.
- If a hook fails (e.g., it fixes trailing whitespace), the commit is aborted. You will need to stage the changes made by the hook (`git add <file>`) and commit again.

### Manual Usage
To run the hooks against all files in the repository (useful when setting up for the first time):

```bash
pre-commit run --all-files
```

To run against specific files:

```bash
pre-commit run --files path/to/file1 path/to/file2
```

## 6. Updating Hooks

To update the versions of the hooks in your config file to the latest stable versions:

```bash
pre-commit autoupdate
```

## 7. Troubleshooting

### Skipping Hooks
In rare cases where you need to bypass the hooks (e.g., committing a file that intentionally breaks a rule), you can use the `--no-verify` flag:

```bash
git commit -m "wip" --no-verify
```

### "Command not found"
If you see `command not found: pre-commit` when committing, ensure you have activated your virtual environment or that `pre-commit` is in your system PATH.
