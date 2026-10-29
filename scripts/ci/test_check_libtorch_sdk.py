#!/usr/bin/env python3
"""Guard cached SDK reuse across version, variant and incomplete installs."""
import os
from pathlib import Path
import tempfile
import unittest
from check_libtorch_sdk import check_sdk, check_python


class CachedLibTorchTest(unittest.TestCase):
    def fixture(self, root, version="2.14.0+cpu"):
        for path in ["lib/libtorch.so", "lib/libtorch_cpu.so", "share/cmake/Torch/TorchConfig.cmake",
                     "include/torch/csrc/api/include/torch/torch.h"]:
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()
        (root / "build-version").write_text(version + "\n")

    def test_matching_native_sdk_is_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            self.assertEqual(check_sdk(root, "2.14.0", "cpu"), "2.14.0+cpu")

    def test_other_version_or_cuda_variant_cannot_satisfy_cpu(self):
        for actual in ["2.12.0+cpu", "2.14.0+cu130", "2.14.0", ""]:
            with self.subTest(actual=actual), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.fixture(root, actual)
                with self.assertRaisesRegex(ValueError, "requested 2.14.0\\+cpu"):
                    check_sdk(root, "2.14.0", "cpu")

    def test_cpu_base_cannot_satisfy_cuda_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            with self.assertRaisesRegex(ValueError, "requested 2.14.0\\+cu130"):
                check_sdk(root, "2.14.0", "cu130")

    def test_partial_sdk_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            (root / "lib/libtorch_cpu.so").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                check_sdk(root, "2.14.0", "cpu")

    def test_python_environment_cannot_hide_another_variant(self):
        for actual in ["2.14.0+cpu", "2.12.0+cpu", "2.14.0+cu130"]:
            with self.subTest(actual=actual), tempfile.TemporaryDirectory() as tmp:
                executable = Path(tmp) / "python"
                executable.write_text("#!/bin/sh\nprintf '%s\\n' '" + actual + "'\n")
                executable.chmod(0o755)
                if actual == "2.14.0+cpu":
                    self.assertEqual(check_python(executable, "2.14.0"), actual)
                else:
                    with self.assertRaisesRegex(ValueError, "expected CPU exporter"):
                        check_python(executable, "2.14.0")


if __name__ == '__main__':
    unittest.main()
