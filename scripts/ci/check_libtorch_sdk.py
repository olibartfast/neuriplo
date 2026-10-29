#!/usr/bin/env python3
"""Validate a reusable LibTorch install against the requested archive variant."""
import argparse
from pathlib import Path
import subprocess


def check_sdk(root, version, platform):
    expected = f"{version}+{platform}"
    actual = (root / "build-version").read_text().strip()
    if actual != expected:
        raise ValueError(f"LibTorch SDK is {actual}, requested {expected}; use the matching base or BASE_IMAGE=base")
    for relative in ["lib/libtorch.so", "lib/libtorch_cpu.so", "share/cmake/Torch/TorchConfig.cmake",
                     "include/torch/csrc/api/include/torch/torch.h"]:
        if not (root / relative).is_file():
            raise ValueError(f"LibTorch SDK is incomplete: missing {relative}")
    return actual


def check_python(python, version):
    actual = subprocess.check_output([str(python), "-c", "import torch; print(torch.__version__)"], text=True).strip()
    expected = version + "+cpu"
    if actual != expected:
        raise ValueError(f"Python Torch is {actual}, expected CPU exporter {expected}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", default="cpu")
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    print("LibTorch SDK:", check_sdk(args.root, args.version, args.platform))
    if args.python:
        print("Python Torch:", check_python(args.python, args.version))


if __name__ == "__main__":
    main()
