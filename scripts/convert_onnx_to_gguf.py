#!/usr/bin/env python3
"""Convert an ONNX model to a GGUF weight file read by GGMLInfer.

GGUF layout (little-endian, version 3, F32 tensors only):

    magic:      4 bytes "GGUF"
    version:    uint32 == 3
    n_tensors:  uint64
    n_kv:       uint64
    metadata:   per KV: key string, value type uint32, value
    tensors:    per tensor: name string, n_dims uint32, ne[n_dims] uint64
                (ggml order, i.e. reversed ONNX dims), type uint32 (0 = F32),
                offset uint64 (relative to the data section start, 32-aligned)
    padding to 32 bytes, then the tensor data section.

Strings are uint64 length + bytes (no NUL). Alignment is 32.

KV entries written:

    general.architecture = "neuriplo-vision" (string)
    neuriplo.rank.<tensor-name> = ONNX rank (int32), one per tensor

The rank entries exist because the GGUF C API exposes tensor dims as a
GGML_MAX_DIMS array padded with trailing ones, with no rank accessor; a
shape like [1,16,1,1] cannot be recovered from dims alone. The loader
rejects files without these keys instead of guessing.

Only F32 (and F16/double widened to F32) initializers of rank 1..4 are
carried; integer
initializers and 0-d/>4-d tensors are skipped with a note. The loader
matches tensors by exact name, so extra tensors are ignored there.

Example:
    python3 scripts/convert_onnx_to_gguf.py \
        --input /opt/models/resnet18-v1-7.onnx \
        --output /opt/models/resnet18.gguf
"""

import argparse
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# ggml_type values (must match ggml.h).
GGML_TYPE_F32 = 0
# gguf_type values (must match gguf.h).
GGUF_TYPE_INT32 = 5
GGUF_TYPE_STRING = 8


def _pack_string(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


# Accepted classifier-head weights: ResNet18 ([1000,512]) and ResNet50
# ([1000,2048]). Both are ONNX Gemm transB=1 row-major.
FC_WEIGHT_SHAPES = ((1000, 512), (1000, 2048))
FC_BIAS_SHAPE = (1000,)


def _read_float_initializers(onnx_path):
    # Float tensors widened to F32. Constant-node float values are carried
    # too: RF-DETR's projector normalizations are exported as graph
    # Constants (`onnx::Mul_*` / `onnx::Add_*`), not initializers, and the
    # hand-written topology needs those exact names.
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    tensors = []
    seen = set()

    def append(name, arr):
        if arr.ndim == 0 or arr.ndim > 4:
            return
        if name in seen:
            return
        seen.add(name)
        tensors.append((name, tuple(int(d) for d in arr.shape), arr.astype("float32")))

    for init in model.graph.initializer:
        if init.data_type in (onnx.TensorProto.FLOAT,):
            arr = numpy_helper.to_array(init).astype("float32")
        elif init.data_type in (onnx.TensorProto.DOUBLE,):
            arr = numpy_helper.to_array(init).astype("float32")
            print(f"note: casting float64 initializer '{init.name}' to float32")
        elif init.data_type in (onnx.TensorProto.FLOAT16,):
            arr = numpy_helper.to_array(init).astype("float32")
            print(f"note: casting float16 initializer '{init.name}' to float32")
        else:
            print(f"note: skipping non-float initializer '{init.name}' "
                  f"(data_type={init.data_type})")
            continue
        if arr.ndim == 0 or arr.ndim > 4:
            print(f"note: skipping '{init.name}': rank {arr.ndim} not in 1..4")
            continue
        append(init.name, arr)

    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            tensor = attr.t
            if tensor.data_type not in (onnx.TensorProto.FLOAT,
                                        onnx.TensorProto.DOUBLE,
                                        onnx.TensorProto.FLOAT16):
                continue
            try:
                arr = numpy_helper.to_array(tensor).astype("float32")
            except (TypeError, ValueError):
                continue
            append(node.output[0], arr)
    return tensors


def _check_fc_pair(tensors):
    # Minimum loadable-head contract: YOLO files need the fixed DFL conv
    # weight; classifier files need one 2D [1000,512]/[1000,2048] weight
    # plus the [1000] bias; RF-DETR files need the 91-class head
    # (`class_embed.bias` [91] plus a 2D `[256,91]` class weight).
    names = {name for name, _, _ in tensors}
    if any(n.startswith("model.") for n in names):
        # YOLOv8n family: DFL at model.22; YOLO11 family: DFL at model.23
        # (plus C2PSA attention at model.10). Accept either; the GGML loader
        # dispatches on the exact pair.
        if "model.22.dfl.conv.weight" not in names and "model.23.dfl.conv.weight" not in names:
            print("error: YOLO file missing fixed DFL conv weight "
                  "'model.22.dfl.conv.weight' or 'model.23.dfl.conv.weight'", file=sys.stderr)
            return False
        return True
    by_name = {name: dims for name, dims, _ in tensors}
    if by_name.get("class_embed.bias") == (91,):
        if not any(dims == (256, 91) for dims in by_name.values()):
            print("error: RF-DETR file missing 2D [256,91] class weight",
                  file=sys.stderr)
            return False
        return True
    shapes = {tuple(dims) for _, dims, _ in tensors}
    if not any(tuple(s) in shapes for s in FC_WEIGHT_SHAPES):
        print(f"error: minimum FC contract not satisfied: no 2D "
              f"{list(FC_WEIGHT_SHAPES)} weight found", file=sys.stderr)
        return False
    if tuple(FC_BIAS_SHAPE) not in shapes:
        print("error: minimum FC contract not satisfied: no 1D [1000] bias found",
              file=sys.stderr)
        return False
    return True


def _aligned(offset, alignment=GGUF_ALIGNMENT):
    return (offset + alignment - 1) // alignment * alignment


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

    flat_data = []
    for _, _, arr in tensors:
        flat_data.append(np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))

    # KV section: general.architecture plus one rank entry per tensor.
    kv_blob = b""
    kv_entries = [("general.architecture", GGUF_TYPE_STRING, "neuriplo-vision")]
    for name, dims, _ in tensors:
        kv_entries.append((f"neuriplo.rank.{name}", GGUF_TYPE_INT32, len(dims)))
    kv_blob += struct.pack("<Q", 0)  # placeholder, replaced below
    kv_count = 0
    kv_body = b""
    for key, value_type, value in kv_entries:
        kv_body += _pack_string(key)
        kv_body += struct.pack("<I", value_type)
        if value_type == GGUF_TYPE_STRING:
            kv_body += _pack_string(value)
        elif value_type == GGUF_TYPE_INT32:
            kv_body += struct.pack("<i", value)
        else:
            raise AssertionError(f"unhandled KV type {value_type}")
        kv_count += 1

    # Tensor infos with 32-aligned data offsets.
    info_body = b""
    offset = 0
    for (name, dims, _), flat in zip(tensors, flat_data):
        ne = list(reversed(dims))  # ggml order
        info_body += _pack_string(name)
        info_body += struct.pack("<I", len(ne))
        for d in ne:
            info_body += struct.pack("<Q", d)
        info_body += struct.pack("<I", GGML_TYPE_F32)
        info_body += struct.pack("<Q", offset)
        offset = _aligned(offset + flat.nbytes)

    header = GGUF_MAGIC
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", kv_count)
    meta = kv_body + info_body
    pad = _aligned(len(header) + len(meta)) - (len(header) + len(meta))
    meta += b"\x00" * pad

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(meta)
        for (_, _, _), flat in zip(tensors, flat_data):
            chunk = flat.astype("<f4", copy=False).tobytes()
            f.write(chunk)
            pad = _aligned(len(chunk)) - len(chunk)
            if pad:
                f.write(b"\x00" * pad)

    # Read-back verification with an independent parse.
    try:
        with open(output_path, "rb") as f:
            blob = f.read()
    except OSError as e:
        print(f"error: failed to read back '{output_path}': {e}", file=sys.stderr)
        return 1
    try:
        off = 0
        assert blob[off:off + 4] == GGUF_MAGIC
        off += 4
        (version,) = struct.unpack_from("<I", blob, off)
        off += 4
        assert version == GGUF_VERSION
        (n_tensors,) = struct.unpack_from("<Q", blob, off)
        off += 8
        (n_kv,) = struct.unpack_from("<Q", blob, off)
        off += 8
        assert n_tensors == len(tensors) and n_kv == kv_count
        for _ in range(n_kv):
            (klen,) = struct.unpack_from("<Q", blob, off)
            off += 8 + klen
            (vtype,) = struct.unpack_from("<I", blob, off)
            off += 4
            if vtype == GGUF_TYPE_STRING:
                (vlen,) = struct.unpack_from("<Q", blob, off)
                off += 8 + vlen
            elif vtype == GGUF_TYPE_INT32:
                off += 4
            else:
                raise AssertionError(f"unexpected KV type {vtype}")
        data_start = None
        infos = []
        for _ in range(n_tensors):
            (nlen,) = struct.unpack_from("<Q", blob, off)
            off += 8
            iname = blob[off:off + nlen].decode("utf-8")
            off += nlen
            (nd,) = struct.unpack_from("<I", blob, off)
            off += 4
            ne = struct.unpack_from("<" + "Q" * nd, blob, off)
            off += 8 * nd
            (ttype,) = struct.unpack_from("<I", blob, off)
            off += 4
            (toffset,) = struct.unpack_from("<Q", blob, off)
            off += 8
            infos.append((iname, nd, ne, ttype, toffset))
        data_start = _aligned(off)
        for (name, dims, _), flat in zip(tensors, flat_data):
            pass
        for (name, dims, flat), (iname, nd, ne, ttype, toffset) in zip(tensors, infos):
            assert iname == name
            assert nd == len(dims)
            assert tuple(ne) == tuple(reversed(dims))
            assert ttype == GGML_TYPE_F32
            assert toffset % GGUF_ALIGNMENT == 0
            assert blob[data_start + toffset:data_start + toffset + flat.nbytes] == \
                flat.astype("<f4", copy=False).tobytes()
    except (AssertionError, struct.error) as e:
        print(f"error: read-back verification failed for '{output_path}': {e}",
              file=sys.stderr)
        return 1

    print(f"wrote {output_path}: {len(tensors)} tensors, {len(blob)} bytes")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input .onnx file")
    parser.add_argument("--output", required=True, help="output .gguf file")
    args = parser.parse_args(argv)
    return convert(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
