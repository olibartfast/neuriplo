#!/usr/bin/env python3
"""Convert an ONNX classifier to the minimal NGGL weight file read by GGMLInfer.

NGGL layout (little-endian, Q-1 decision in
specs/2026-11-13-ggml-real-loader/requirements.md):

    magic:       4 bytes "NGGL"
    version:     uint32 == 1
    num_tensors: uint32
    per tensor:  uint32 name_len, name bytes (utf-8),
                 uint32 ndims, uint32 dims[ndims] (ONNX order),
                 uint64 num_elements, float32 data[num_elements]

The loader (backends/ggml/src/GGMLInfer.cpp) requires the exact ResNet18
classifier tensors `resnetv15_dense0_weight` ([1000,512] or [512,1000] float;
ONNX Gemm transB=1 weight) and `resnetv15_dense0_bias` ([1000] float), plus
the fixed conv/BN role tensors of the ResNet18 topology. All other float
initializers are carried verbatim; integer initializers and 0-d/>4-d float
tensors are skipped with a printed note. Extra tensors are ignored by the
loader. This file is custom NGGL, not GGUF.

Example:
    python3 scripts/convert_onnx_to_ggml.py \
        --input /opt/models/resnet18-v1-7.onnx \
        --output /opt/models/resnet18.ggml \
        --input-shape 3,224,224
"""

import argparse
import struct
import sys

NGGL_MAGIC = b"NGGL"
NGGL_VERSION = 1
FC_WEIGHT_SHAPE = (1000, 512)
FC_BIAS_SHAPE = (1000,)


def _read_float_initializers(onnx_path):
    """Return [(name, dims_tuple, float32_numpy_array)] for float initializers."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    tensors = []
    for init in model.graph.initializer:
        if init.data_type in (onnx.TensorProto.FLOAT,):
            arr = numpy_helper.to_array(init).astype(np.float32)
        elif init.data_type in (onnx.TensorProto.DOUBLE,):
            arr = numpy_helper.to_array(init).astype(np.float32)
            print(f"note: casting float64 initializer '{init.name}' to float32")
        elif init.data_type in (onnx.TensorProto.FLOAT16,):
            arr = numpy_helper.to_array(init).astype(np.float32)
            print(f"note: casting float16 initializer '{init.name}' to float32")
        else:
            print(f"note: skipping non-float initializer '{init.name}' "
                  f"(data_type={init.data_type})")
            continue
        if arr.ndim == 0 or arr.ndim > 4:
            # The NGGL reader only accepts 1D..4D tensors; a 0-d scalar or a
            # >4-d tensor can never be the classifier head, so skip it
            # instead of writing a file the loader must reject.
            print(f"note: skipping '{init.name}': rank {arr.ndim} not in 1..4")
            continue
        tensors.append((init.name, tuple(int(d) for d in init.dims), arr))
    return tensors


def _check_fc_pair(tensors):
    """Fail unless a [1000,512] weight and a [1000] bias are present."""
    shapes = {tuple(dims) for _, dims, _ in tensors}
    if tuple(FC_WEIGHT_SHAPE) not in shapes:
        print(f"error: minimum FC contract not satisfied: no 2D "
              f"{list(FC_WEIGHT_SHAPE)} weight found", file=sys.stderr)
        return False
    if tuple(FC_BIAS_SHAPE) not in shapes:
        print("error: minimum FC contract not satisfied: no 1D [1000] bias found",
              file=sys.stderr)
        return False
    return True


def convert(onnx_path, output_path):
    import numpy as np

    try:
        tensors = _read_float_initializers(onnx_path)
    except FileNotFoundError:
        print(f"error: input file not found: {onnx_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: failed to read '{onnx_path}': {e}", file=sys.stderr)
        return 1
    if not tensors:
        print("error: no float initializers found", file=sys.stderr)
        return 1
    if not _check_fc_pair(tensors):
        return 1

    with open(output_path, "wb") as f:
        f.write(NGGL_MAGIC)
        f.write(struct.pack("<I", NGGL_VERSION))
        f.write(struct.pack("<I", len(tensors)))
        for name, dims, arr in tensors:
            flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<Q", flat.size))
            f.write(flat.astype("<f4", copy=False).tobytes())

    # Read-back verification: every byte must be accounted for.
    try:
        with open(output_path, "rb") as f:
            blob = f.read()
    except OSError as e:
        print(f"error: failed to read back '{output_path}': {e}", file=sys.stderr)
        return 1
    try:
        offset = 12
        for _ in tensors:
            (name_len,) = struct.unpack_from("<I", blob, offset)
            offset += 4 + name_len
            (ndims,) = struct.unpack_from("<I", blob, offset)
            offset += 4 + 4 * ndims
            (num_elements,) = struct.unpack_from("<Q", blob, offset)
            offset += 8 + 4 * num_elements
    except struct.error as e:
        print(f"error: read-back verification failed for '{output_path}': {e}",
              file=sys.stderr)
        return 1
    if offset != len(blob):
        print("error: read-back verification failed (trailing bytes)", file=sys.stderr)
        return 1

    print(f"wrote {output_path}: {len(tensors)} tensors, {len(blob)} bytes")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input .onnx file")
    parser.add_argument("--output", required=True, help="output .ggml (NGGL) file")
    parser.add_argument("--input-shape", default="3,224,224",
                        help="model input shape (recorded for callers; layout is fixed)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size (recorded for callers; loader uses its own)")
    args = parser.parse_args(argv)
    # Fixed layout packet: only the default geometry is supported. Reject
    # anything else loudly instead of silently ignoring it.
    if args.input_shape != "3,224,224":
        print(f"error: unsupported --input-shape '{args.input_shape}': "
              "this packet supports only 3,224,224", file=sys.stderr)
        return 2
    if args.batch_size != 1:
        print(f"error: unsupported --batch-size {args.batch_size}: "
              "this packet supports only 1", file=sys.stderr)
        return 2
    return convert(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
