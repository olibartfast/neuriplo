#!/usr/bin/env python3
"""
Regenerate GEN: sections in markdown files from docs/backends.yaml.

Usage:
  python3 scripts/gen_backend_docs.py           # update files in place
  python3 scripts/gen_backend_docs.py --check   # exit 1 if any file would change

GEN: blocks in markdown look like:
  <!-- GEN:tag -->
  ...auto-generated content...
  <!-- /GEN:tag -->

Everything between the delimiters is replaced on each run.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------

def load_versions():
    versions = {}
    for line in (ROOT / "versions.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            versions[k.strip()] = v.strip()
    return versions


def load_backends():
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")
    data = yaml.safe_load((ROOT / "docs" / "backends.yaml").read_text())
    return data["backends"]


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def gen_overview_table(backends, versions):
    rows = ["| Backend ID | Name | Version | Arch | GPU |",
            "|---|---|---|---|---|"]
    for b in backends:
        ver = versions.get(b["version_var"], "—")
        arch_list = b.get("arch", ["x86_64", "arm64"])
        arch = "ARM64 only" if arch_list == ["arm64"] else (
               "x86_64 only" if arch_list == ["x86_64"] else "x86_64, ARM64")
        gpu = "yes" if b.get("gpu") else "no"
        rows.append(f"| `{b['id']}` | {b['name']} | `{ver}` | {arch} | {gpu} |")
    return "\n".join(rows)


def gen_setup_scripts_table(backends, _versions):
    rows = ["| Script | Backend / purpose |",
            "|---|---|"]
    rows.append("| `setup_dependencies.sh` | Unified dispatcher — delegates to the script below for the chosen `--backend` |")
    for b in backends:
        script = b.get("setup_script")
        if script is None:
            note = f"{b['name']} — {b.get('setup_note', 'system package')}"
            rows.append(f"| _(none)_ | {note} |")
        else:
            note = b.get("setup_note", "")
            label = f"{b['name']}" + (f" — {note}" if note else "")
            rows.append(f"| `{script}` | {label} |")
    rows.append("| `build_cactus.sh` | Build the Cactus Docker image (ARM64 only) |")
    return "\n".join(rows)


def gen_cmake_dir_variables(backends, _versions):
    rows = ["| Variable | Default path |",
            "|---|---|"]
    for b in backends:
        if b.get("dir_var") and b.get("dir_default"):
            rows.append(f"| `{b['dir_var']}` | `{b['dir_default']}` |")
    return "\n".join(rows)


def gen_cmake_version_variables(backends, versions):
    rows = ["| Variable | Current value in `versions.env` |",
            "|---|---|"]
    for b in backends:
        var = b["version_var"]
        val = versions.get(var, "—")
        rows.append(f"| `{var}` | `{val}` |")
    return "\n".join(rows)


def gen_env_variables(backends, versions):
    lines = ["```bash",
             'export DEPENDENCY_ROOT="$HOME/dependencies"']
    for b in backends:
        var = b.get("dir_var")
        default = b.get("dir_default")
        if var and default:
            ver_var = b["version_var"]
            ver = versions.get(ver_var, "<ver>")
            resolved = default.replace("<ver>", ver)
            lines.append(f'export {var}="{resolved}"')
    lines += [
        'export LD_LIBRARY_PATH="\\',
    ]
    ld_parts = []
    for b in backends:
        var = b.get("dir_var")
        default = b.get("dir_default")
        if not var or not default:
            continue
        if var == "MIGRAPHX_ROOT":
            ld_parts.append(f"${var}/lib")
        elif var == "TVM_DIR":
            ld_parts.append(f"${var}/build")
        elif var == "OPENVINO_DIR":
            ld_parts.append(f"${var}/runtime/lib/intel64")
        else:
            ld_parts.append(f"${var}/lib")
    for i, part in enumerate(ld_parts):
        sep = ":\\" if i < len(ld_parts) - 1 else ':\\'
        lines.append(f"{part}{sep}")
    lines.append('$LD_LIBRARY_PATH"')
    lines.append("```")
    return "\n".join(lines)


def gen_test_models_table(backends, _versions):
    rows = ["| Backend | Model format | How it is obtained |",
            "|---|---|---|"]
    for b in backends:
        rows.append(f"| {b['name']} | {b['model_format']} | {b['model_source']} |")
    return "\n".join(rows)


GENERATORS = {
    "backend-overview":          gen_overview_table,
    "setup-scripts-table":       gen_setup_scripts_table,
    "cmake-dir-variables":       gen_cmake_dir_variables,
    "cmake-version-variables":   gen_cmake_version_variables,
    "env-variables":             gen_env_variables,
    "test-models-table":         gen_test_models_table,
}

# ---------------------------------------------------------------------------
# Inject / check
# ---------------------------------------------------------------------------

DELIM_RE = re.compile(
    r"(<!-- GEN:(?P<tag>[^>]+) -->)"
    r".*?"
    r"(<!-- /GEN:(?P=tag) -->)",
    re.DOTALL,
)


def process_file(path: Path, backends, versions, check: bool) -> bool:
    """Return True if the file was (or would be) changed."""
    original = path.read_text()
    result = original

    def replacer(m):
        tag = m.group("tag")
        gen = GENERATORS.get(tag)
        if gen is None:
            print(f"  warning: unknown GEN tag '{tag}' in {path.name}", file=sys.stderr)
            return m.group(0)
        content = gen(backends, versions)
        return f"<!-- GEN:{tag} -->\n{content}\n<!-- /GEN:{tag} -->"

    result = DELIM_RE.sub(replacer, result)

    if result == original:
        return False

    if check:
        print(f"OUTDATED: {path.relative_to(ROOT)}")
        return True

    path.write_text(result)
    print(f"updated:  {path.relative_to(ROOT)}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if any file would change (CI mode)")
    args = parser.parse_args()

    versions = load_versions()
    backends = load_backends()

    targets = [
        ROOT / "docs" / "DEPENDENCY_MANAGEMENT.md",
    ]

    changed = any(process_file(p, backends, versions, args.check)
                  for p in targets if p.exists())

    if args.check and changed:
        print("\nRun `python3 scripts/gen_backend_docs.py` to fix.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
