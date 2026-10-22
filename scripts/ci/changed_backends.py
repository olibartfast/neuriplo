#!/usr/bin/env python3
"""Return the backend ids whose CI jobs should run for a change set.

Backend-specific changes run that backend only. Shared runtime/build/test
paths run every registered backend. Documentation-only paths run none. Unknown
paths are treated as shared so an unclassified directory cannot silently skip
a backend.

Usage:
    scripts/ci/changed_backends.py --base <sha> --head <sha>
    scripts/ci/changed_backends.py --all
    scripts/ci/changed_backends.py --paths backends/opencv-dnn/src/foo.cpp
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cmake" / "BackendRegistry.cmake"

# Dockerfile name -> backend ids that consume it. The ONNX Runtime image also
# carries the NERT parity job.
DOCKER_BACKENDS = {
    "Dockerfile.opencvdnn": {"OPENCV_DNN"},
    "Dockerfile.onnx-runtime": {"ONNX_RUNTIME", "NERT"},
    "Dockerfile.libtorch": {"LIBTORCH"},
    "Dockerfile.libtensorflow": {"LIBTENSORFLOW"},
    "Dockerfile.openvino": {"OPENVINO"},
    "Dockerfile.ggml": {"GGML"},
    "Dockerfile.tvm": {"TVM"},
    "Dockerfile.migraphx": {"MIGRAPHX"},
    "Dockerfile.tensorrt": {"TENSORRT"},
    "Dockerfile.executorch": {"EXECUTORCH"},
    "Dockerfile.litert": {"LITERT"},
    "Dockerfile.llamacpp": {"LLAMACPP"},
    "Dockerfile.cactus": {"CACTUS"},
    "Dockerfile.nert": {"NERT"},
}

# Directories whose changes affect every backend (binary, headers, shared
# adapter code, build system, tests, CI, or sanitizer configuration).
SHARED_PREFIXES = (
    "src/",
    "include/",
    "test/",
    "backends/src/",
    "cmake/",
    ".github/workflows/",
    "scripts/quality/",
)

# Changes that only affect whole-tree checks or documentation.
IGNORED_PREFIXES = (
    "docs/",
    "specs/",
    "scripts/ci/",
)

IGNORED_FILES = {
    "REPO_META.yaml",
}


def registry() -> tuple[list[str], dict[str, str]]:
    text = REGISTRY.read_text(encoding="utf-8")
    entries = re.findall(
        r"set\(NEURIPLO_BACKEND_([A-Z0-9_]+)_TEST_DIR\s+([^\)]+)\)", text
    )
    if not entries:
        raise SystemExit(f"no backend test directories found in {REGISTRY}")
    ids = sorted({backend for backend, _ in entries})
    directories = {}
    for backend, test_dir in entries:
        # backends/<dir>/test -> backends/<dir>
        path = Path(test_dir.strip())
        if len(path.parts) >= 2 and path.parts[0] == "backends":
            directories[path.parts[1]] = backend
    return ids, directories


def changed_paths(base: str, head: str) -> list[str] | None:
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        # Fail open: an unavailable base (force-push, shallow ref, or a new
        # branch) runs the full matrix rather than silently skipping backends.
        print(
            f"changed_backends: git diff failed, running all backends: "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )
        return None
    return [line for line in result.stdout.splitlines() if line]


def classify(path: str, all_ids: list[str], backend_dirs: dict[str, str]) -> set[str]:
    if path in IGNORED_FILES:
        return set()
    if path.startswith(IGNORED_PREFIXES):
        return set()

    if path.startswith("backends/"):
        parts = Path(path).parts
        if len(parts) < 2:
            return set(all_ids)
        if parts[1] == "src":
            return set(all_ids)
        backend = backend_dirs.get(parts[1])
        return {backend} if backend is not None else set(all_ids)

    if path.startswith("docker/"):
        name = Path(path).name
        return set(DOCKER_BACKENDS.get(name, all_ids))

    if path == "cmake/Nert.cmake":
        return {"NERT"}
    if path in {"CMakeLists.txt", "versions.env"}:
        return set(all_ids)
    if path.startswith(SHARED_PREFIXES):
        return set(all_ids)
    if path.startswith("scripts/abi/"):
        return {"OPENCV_DNN"}
    if path in {
        "scripts/ci/backend_matrices.json",
        "scripts/ci/backend_matrices.py",
    }:
        return set(all_ids)
    if path.startswith("scripts/"):
        return set()
    if path.endswith(".md"):
        return set()

    # Unknown paths are shared: fail open rather than skip a backend.
    return set(all_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--paths", nargs="*")
    args = parser.parse_args()

    all_ids, backend_dirs = registry()
    if args.all:
        selected = set(all_ids)
    else:
        paths = args.paths
        if paths is None:
            if not args.base or not args.head:
                parser.error("--base and --head are required unless --all/--paths is used")
            paths = changed_paths(args.base, args.head)
        if paths is None:
            selected = set(all_ids)
        else:
            selected = set()
            for path in paths:
                selected.update(classify(path, all_ids, backend_dirs))

    print(json.dumps(sorted(selected), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
