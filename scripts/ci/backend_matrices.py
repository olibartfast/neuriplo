#!/usr/bin/env python3
"""Filter a CI backend matrix to the backends selected by the change set.

The matrix data lives in `backend_matrices.json`; keeping it out of the
workflow lets the change detector emit a dynamic `matrix.include` without
duplicating the dockerfile/test metadata in YAML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts" / "ci" / "backend_matrices.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", required=True, help="JSON array of backend ids")
    parser.add_argument("--job", required=True, choices=["clang-tidy", "build-warnings", "sanitizers", "cpu"])
    args = parser.parse_args()

    try:
        selected = set(json.loads(args.backends))
    except json.JSONDecodeError as exc:
        print(f"backend_matrices: invalid --backends JSON: {exc}", file=sys.stderr)
        return 2

    matrices = json.loads(CONFIG.read_text(encoding="utf-8"))
    entries = [entry for entry in matrices[args.job] if entry["backend"] in selected]
    print(json.dumps(entries, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
