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
    def test_sdk_changes_select_backend_builds(self):
        detector = str(ROOT / "scripts/ci/changed_backends.py")
        all_backends = json.loads(subprocess.check_output(
            ["python3", detector, "--all"], text=True))
        for changed in ("scripts/ci/resolve_sdk_base.sh",
                        "scripts/ci/backend_matrices.json",
                        "scripts/ci/backend_matrices.py"):
            with self.subTest(changed=changed):
                selected = json.loads(subprocess.check_output(
                    ["python3", detector, "--paths", changed], text=True))
                self.assertEqual(selected, all_backends)
        selected = json.loads(subprocess.check_output(
            ["python3", detector, "--paths", "scripts/ci/executorch-sdk-smoke/main.cpp"], text=True))
        self.assertEqual(selected, ["EXECUTORCH"])
        for changed in ("scripts/ci/check_libtorch_sdk.py", "scripts/ci/test_check_libtorch_sdk.py", "scripts/ci/libtorch-sdk-smoke/main.cpp"):
            with self.subTest(changed=changed):
                selected = json.loads(subprocess.check_output(
                    ["python3", detector, "--paths", changed], text=True))
                self.assertEqual(selected, ["LIBTORCH"])

    def test_libtorch_is_cached_in_every_matrix(self):
        matrices = json.loads((ROOT / "scripts/ci/backend_matrices.json").read_text())
        self.assertEqual(set(matrices), {"clang-tidy", "build-warnings", "sanitizers", "cpu"})
        for job, entries in matrices.items():
            rows = [entry for entry in entries if entry["backend"] == "LIBTORCH"]
            with self.subTest(job=job):
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["base_image"], "ghcr.io/olibartfast/neuriplo/libtorch-deps")
                self.assertEqual(rows[0]["base_fallback"], "base")

    def test_matrix_resolution(self):
        matrices = json.loads((ROOT / "scripts/ci/backend_matrices.json").read_text())
        for entries in matrices.values():
            for entry in entries:
                if "base_image" not in entry:
                    continue
                backend = entry["backend"]
                # A custom fallback carries the SDK's own build prerequisites;
                # the default is a bare Ubuntu for fully self-contained
                # Dockerfiles.
                expected_fallback = entry.get("base_fallback", "ubuntu:24.04")
                for mode in ("pull-success", "pull-failure", "act"):
                    with self.subTest(backend=backend, mode=mode), tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp)
                        # Exercise real pin names: backend IDs need not match SDK keys.
                        pins = (ROOT / "versions.env").read_text()
                        (path / "versions.env").write_text(pins)
                        key = {"LIBTENSORFLOW": "TENSORFLOW", "LIBTORCH": "PYTORCH"}.get(backend, backend)
                        version = next(line.split("=", 1)[1] for line in pins.splitlines()
                                       if line.startswith(f"{key}_VERSION="))
                        if backend == "LIBTORCH":
                            version += "-cpu"
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
                        expected = f'{entry["base_image"]}:{version}' if mode == "pull-success" else expected_fallback
                        self.assertEqual((path / "output").read_text(), f"ref={expected}\n")
                        if mode == "act":
                            self.assertFalse((path / "pull").exists())
                        else:
                            self.assertEqual((path / "pull").read_text(),
                                             f'pull {entry["base_image"]}:{version}\n')


if __name__ == "__main__":
    unittest.main()
