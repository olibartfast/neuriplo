#!/usr/bin/env python3
"""Typed export regression tests; run with the pinned PyTorch environment."""
import argparse
import os
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

import torch
from lib.torchscript_metadata import (
    AOTI_KEY, TORCHSCRIPT_FILE, aoti_metadata, decode_dtypes, encode_dtypes,
    has_aoti_metadata, has_torchscript_metadata, save_torchscript,
)


class TwoInputs(torch.nn.Module):
    def forward(self, image, sizes):
        return image + sizes.to(image).sum()


class TypedExportTest(unittest.TestCase):
    def test_roundtrip_uses_actual_example_types(self):
        for dtype, tag in [(torch.int32, 'INT32'), (torch.int64, 'INT64'),
                           (torch.uint8, 'UINT8'), (torch.int8, 'INT8'), (torch.bool, 'BOOL')]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                image = torch.ones(1, 3, 2, 2)
                sizes = torch.ones(1, 2, dtype=dtype)
                module = torch.jit.trace(TwoInputs(), (image, sizes))
                path = Path(tmp) / 'typed.pt'
                save_torchscript(module, path, (image, sizes))
                files = {TORCHSCRIPT_FILE: ''}
                loaded = torch.jit.load(str(path), _extra_files=files)
                self.assertEqual(decode_dtypes(files[TORCHSCRIPT_FILE], 2), ['FP32', tag])
                torch.testing.assert_close(loaded(image, sizes), TwoInputs()(image, sizes))
                self.assertTrue(has_torchscript_metadata(path))

    def test_untyped_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.pt'
            inputs = (torch.ones(1), torch.ones(1, dtype=torch.int64))
            torch.jit.trace(TwoInputs(), inputs).save(str(path))
            self.assertFalse(has_torchscript_metadata(path))

    def test_unsupported_types_and_invalid_metadata(self):
        for inputs in [(), (1,), (torch.ones(1, dtype=torch.float16),)]:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                encode_dtypes(inputs)
        for text in ['', 'wrong 1 FP32', 'NEURIPLO_INPUT_DTYPES_V1 2 FP32',
                     'NEURIPLO_INPUT_DTYPES_V1 1 FP16', 'NEURIPLO_INPUT_DTYPES_V1 1 FP32 extra', None]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_dtypes(text)

    def test_aoti_types_come_from_compiler_graph(self):
        for dtype, tag in [(torch.int32, 'INT32'), (torch.int64, 'INT64')]:
            with self.subTest(dtype=dtype):
                program = torch.export.export(TwoInputs(),
                                              (torch.ones(1, 3, 2, 2), torch.ones(1, 2, dtype=dtype)))
                self.assertEqual(decode_dtypes(aoti_metadata(program)[AOTI_KEY]), ['FP32', tag])

    def test_forward_count_and_known_graph_type_mismatch(self):
        inputs = (torch.ones(1), torch.ones(1, dtype=torch.int64))
        module = torch.jit.trace(TwoInputs(), inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.pt'
            with self.assertRaises(ValueError):
                save_torchscript(module, path, inputs[:1])
            with self.assertRaises(ValueError):
                save_torchscript(module, path, (inputs[0], torch.ones(1, dtype=torch.int32)))
            self.assertFalse(path.exists())


class ExportRuntimeInteropTest(unittest.TestCase):
    probe = None

    def test_mixed_inputs_reach_cpp_loaders(self):
        env = os.environ | {"AOTI_RUNTIME_CHECK_INPUTS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"}
        for dtype, tag in [(torch.int32, 'INT32'), (torch.int64, 'INT64')]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                inputs = (torch.ones(1, 3, 2, 2), torch.ones(1, 2, dtype=dtype))
                module = TwoInputs().eval()
                script_path = Path(tmp) / 'typed.pt'
                save_torchscript(torch.jit.trace(module, inputs), script_path, inputs)
                subprocess.run([self.probe, str(script_path), tag], env=env, check=True)
                program = torch.export.export(module, inputs)
                aoti_path = Path(tmp) / 'typed.pt2'
                torch._inductor.aoti_compile_and_package(
                    program, package_path=str(aoti_path),
                    inductor_configs={"aot_inductor.metadata": aoti_metadata(program)})
                self.assertTrue(has_aoti_metadata(aoti_path))
                subprocess.run([self.probe, str(aoti_path), tag], env=env, check=True)
                subprocess.run([self.probe, str(aoti_path), tag, 'bad-shape'], env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', help='C++ component probe for TorchScript/AOTI interoperability')
    args, remaining = parser.parse_known_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TypedExportTest)
    if args.probe:
        ExportRuntimeInteropTest.probe = str(Path(args.probe).resolve(strict=True))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ExportRuntimeInteropTest))
    else:
        print('Exporter-only suite; C++/AOTI interoperability requires --probe', file=sys.stderr)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
