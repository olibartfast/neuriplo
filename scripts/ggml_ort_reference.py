#!/usr/bin/env python3
"""ORT reference logits for the GGML parity goldens (evidence only).

Runs an ONNX model under stock onnxruntime on the fixed inputs the C++
parity tests feed the GGML loader (--img-size selects geometry):

  gray  - NCHW float32 blob with every element == 128/255, byte-exact with
          neuriplo::testing::constant_blob_tensors(kGray128Normalized).
  zeros - NCHW float32 all-zeros, byte-exact with zero_blob_tensors().
  gradient - deterministic spatially-nonuniform NCHW pattern, byte-exact
          with gradient_blob_tensors() in GGMLInferTest.cpp:
          v(n,c,y,x) = ((x*13 + y*17 + c*19) % 256) / 255 as float32.
          Uniform inputs cannot validate the NCHW layout claim; this one can.

Saves raw logits (float32) to /tmp/ort_ref_{gray,zeros,gradient}.npy
and prints the top-5 classes of each. Not shipped; never imported by the build.

Run: /tmp/ortenv/bin/python scripts/ggml_ort_reference.py
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def gray128_normalized_float32() -> np.float32:
    # Single-precision division, exactly what C++ 128.0F/255.0F computes
    # (IEEE-754 round-to-nearest on float operands).
    return np.float32(128.0) / np.float32(255.0)


def top5(logits: np.ndarray):
    idx = np.argsort(logits)[::-1][:5]
    return [(int(i), float(logits[i])) for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate ORT reference logits for GGML parity test")
    ap.add_argument("--model", default="/tmp/resnet18-v1-7.onnx")
    ap.add_argument("--gray-out", default="/tmp/ort_ref_gray.npy")
    ap.add_argument("--zeros-out", default="/tmp/ort_ref_zeros.npy")
    ap.add_argument("--gradient-out", default="/tmp/ort_ref_gradient.npy")
    ap.add_argument("--img-size", type=int, default=224,
                    help="square spatial size (224 resnet, 640 yolo)")
    ap.add_argument("--all-outputs", action="store_true",
                    help="save every session output as <stem>_out<i>.npy (default: first output only)")
    args = ap.parse_args()

    import onnxruntime as ort

    h = w = args.img_size

    gray_value = gray128_normalized_float32()
    # Byte-exactness check vs the host double-rounding path: both must agree.
    straight = np.float32(128.0 / 255.0)
    print(f"gray value: {gray_value!r} bits={struct.pack('<f', float(gray_value)).hex()} "
          f"double-rounded-agrees={gray_value == straight}")
    assert gray_value == straight, "float32 division differs from double rounding; investigate before trusting parity"

    gray = np.full((1, 3, h, w), gray_value, dtype=np.float32)
    zeros = np.zeros((1, 3, h, w), dtype=np.float32)
    assert gray.nbytes == 1 * 3 * h * w * 4 == zeros.nbytes

    # Gradient pattern: integer math, then float32 division — mirrors the C++
    # helper exactly (int v, then (float)v/255.0F). NCHW (1,3,H,W).
    x = np.arange(w, dtype=np.int64)[None, None, :]
    y = np.arange(h, dtype=np.int64)[None, :, None]
    c = np.arange(3, dtype=np.int64)[:, None, None]
    vint = (x * 13 + y * 17 + c * 19) % 256
    gradient = (vint.astype(np.float32) / np.float32(255.0))[None]
    assert gradient.shape == (1, 3, h, w) and gradient.dtype == np.float32

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]
    print(f"model={args.model} input={input_name} outputs={output_names} "
          f"ort={ort.__version__} numpy={np.__version__}")

    def output_path(base, index):
        if not args.all_outputs:
            return base
        path = Path(base)
        return path.with_name(f"{path.stem}_out{index}{path.suffix}")

    for label, blob, out_path in (("gray", gray, args.gray_out), ("zeros", zeros, args.zeros_out),
                                   ("gradient", gradient, args.gradient_out)):
        raw_outputs = sess.run(output_names, {input_name: blob})
        for index, raw_output in enumerate(raw_outputs):
            logits = np.asarray(raw_output, dtype=np.float32)
            flat = logits.reshape(-1)
            target = output_path(out_path, index)
            np.save(target, flat)
            print(f"{label}[{index}]: saved {target} shape={logits.shape} dtype={flat.dtype}")
            if flat.size == 1000:
                for rank, (cls, score) in enumerate(top5(flat), start=1):
                    print(f"  top{rank}: class={cls} logit={score:.6f}")
            else:
                amax = int(np.argmax(flat))
                print(f"  max_cell={amax} value={float(flat[amax]):.6f}")
            if not args.all_outputs:
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
