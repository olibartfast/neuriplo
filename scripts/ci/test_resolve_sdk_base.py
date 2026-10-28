#!/usr/bin/env python3
"""Exercise SDK resolution with a fake registry, including offline act runs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ResolveSDKBaseTest(unittest.TestCase):
    def test_matrix_resolution(self):
        matrices = json.loads((ROOT / "scripts/ci/backend_matrices.json").read_text())
        for entries in matrices.values():
            for entry in entries:
                if "base_image" not in entry:
                    continue
                backend = entry["backend"]
                # A custom fallback carries the SDK's own build prerequisites
                # (TVM needs LLVM/GTest; llama.cpp needs OpenBLAS/GTest); the
                # default is a bare Ubuntu for fully self-contained Dockerfiles.
                expected_fallback = entry.get("base_fallback", "ubuntu:24.04")
                for mode in ("pull-success", "pull-failure", "act"):
                    with self.subTest(backend=backend, mode=mode), tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp)
                        (path / "versions.env").write_text(f"{backend}_VERSION=1.2.3\n")
                        docker = path / "docker"
                        docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$PULL_LOG"\nexit "$PULL_STATUS"\n')
                        docker.chmod(0o755)
                        env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}",
                                   ACT="true" if mode == "act" else "",
                                   GITHUB_OUTPUT=str(path / "output"),
                                   PULL_LOG=str(path / "pull"),
                                   PULL_STATUS="1" if mode == "pull-failure" else "0")
                        subprocess.run(["bash", str(ROOT / "scripts/ci/resolve_sdk_base.sh"),
                                        backend, entry["base_image"],
                                        entry.get("base_fallback", "ubuntu:24.04")],
                                       cwd=path, env=env, check=True, capture_output=True)
                        expected = f'{entry["base_image"]}:1.2.3' if mode == "pull-success" else expected_fallback
                        self.assertEqual((path / "output").read_text(), f"ref={expected}\n")
                        if mode == "act":
                            self.assertFalse((path / "pull").exists())
                        else:
                            self.assertEqual((path / "pull").read_text(),
                                             f'pull {entry["base_image"]}:1.2.3\n')


if __name__ == "__main__":
    unittest.main()
