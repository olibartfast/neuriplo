#!/usr/bin/env python3
"""Provision the YOLO NEURIPLO_ENGINE<->ONNX_RUNTIME parity fixture (packet Y4).

Exports Ultralytics ``yolo26n`` with NMS embedded to
``<output_dir>/yolo26n.onnx`` (opset 18, ``imgsz=640``) and verifies the
emitted graph against the operators the first-party neuriplo engine
implements. This mirrors ``provision_parity_fixture.py`` (ResNet-18) but
takes its output directory as an argument and never writes to the old
hard-coded ``/workspace/`` path.

Downloading the ``yolo26n.pt`` weights (~5 MB) on first run is expected:
``ultralytics`` fetches them automatically when they are not cached
locally.

Packet Y11 additionally provisions a REAL image and a deterministic input
tensor:

1. ``bus.jpg`` — the Ultralytics sample photo, downloaded from the pinned
   URL below and verified against a pinned sha256. A mismatch or missing
   download is a loud provisioning failure, same discipline as the
   ``yolo26n.pt`` weight download.
2. ``yolo26n_input.bin`` — a raw little-endian float32 binary of exactly
   ``1x3x640x640`` elements, built deterministically from the photo:
   open with PIL, convert to RGB, bilinear-resize to 640x640, scale to
   ``[0,1]`` (``/255``), reshape to CHW. Both backends receive these exact
   bits, so preprocessing determinism only matters for reproducibility of
   the fixture across runs, not for the comparison itself.

Usage:
    provision_yolo_fixture.py <output_dir>
"""

import hashlib
import os
import shutil
import sys
import urllib.request

# The exporter and the emitted graph are pinned so the fixture is reproducible.
PINNED_ULTRALYTICS_VERSION = "8.4.113"
PINNED_TORCH_VERSION_PREFIX = "2.12."
PINNED_ONNX_VERSION_PREFIX = "1.22."
OPSET_VERSION = 18
EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_OUTPUT_SHAPE = [1, 300, 6]

# Packet Y11: the real photo is the Ultralytics sample bus.jpg, pinned by url
# and content hash; refuse to proceed on any mismatch.
PINNED_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
PINNED_IMAGE_SHA256 = "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"
IMAGE_MODEL_NAME = "bus.jpg"
INPUT_TENSOR_NAME = "yolo26n_input.bin"
TENSOR_ELEMENTS = 1 * 3 * 640 * 640

# Operators the neuriplo engine implements (the 31-op allowlist in
# engine/src/ModelLoader.cpp: allowed_attributes()). The fixture must use
# only these so the same file runs through both backends.
# Note: "Constant" is admitted here because the loader folds Constant nodes
# into initializers before the allowlist is enforced (ModelLoader.cpp), so
# Constant nodes never reach the allowlist check at inference time.
SUPPORTED_OPS = {
    "Add",
    "Cast",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "Div",
    "Equal",
    "Expand",
    "Flatten",
    "Gather",
    "GatherElements",
    "Gemm",
    "MatMul",
    "MaxPool",
    "Mod",
    "Mul",
    "ReduceMax",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Resize",
    "Shape",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Split",
    "Sub",
    "TopK",
    "Transpose",
    "Unsqueeze",
    "Where",
}


def fail(message):
    print(f"provision_yolo_fixture: {message}", file=sys.stderr)
    sys.exit(1)


def check_version(label, found, pinned):
    if found != pinned:
        fail(
            f"{label} version mismatch: fixture export is pinned to "
            f"{pinned}, found {found!r}"
        )


def check_version_prefix(label, found, prefix):
    base = found.split("+", 1)[0]
    if not base.startswith(prefix):
        fail(
            f"{label} version mismatch: fixture export is pinned to "
            f"{prefix}*, found {found!r} (base {base!r})"
        )


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_verify_image(image_path):
    """Download the pinned sample photo and verify its content hash."""
    request = urllib.request.Request(
        PINNED_IMAGE_URL,
        headers={"User-Agent": "neuriplo-provision-yolo-fixture/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except OSError as exc:
        fail(f"cannot download pinned image {PINNED_IMAGE_URL}: {exc}")
    if not data:
        fail(f"downloaded image body is empty ({PINNED_IMAGE_URL})")
    found = hashlib.sha256(data).hexdigest()
    if found != PINNED_IMAGE_SHA256:
        fail(
            f"pinned image sha256 mismatch for {PINNED_IMAGE_URL}: "
            f"expected {PINNED_IMAGE_SHA256}, got {found}"
        )
    with open(image_path, "wb") as handle:
        handle.write(data)


def build_input_tensor(image_path, tensor_path):
    """Decode/resize/normalize the photo into a deterministic float32 bin."""
    try:
        import numpy
        from PIL import Image
    except ImportError as exc:
        fail(f"required Python package missing: {exc}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((640, 640), Image.BILINEAR)
        pixels = numpy.asarray(image, dtype=numpy.float32) / 255.0
        chw = numpy.ascontiguousarray(pixels.transpose(2, 0, 1))
        assert chw.shape == (3, 640, 640)
        flat = chw.reshape(-1)
        assert flat.size == TENSOR_ELEMENTS
        with open(tensor_path, "wb") as handle:
            handle.write(flat.tobytes(order="C"))


def static_shape(value_info):
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            fail(
                f"tensor {value_info.name!r} has a dynamic dimension; "
                "the YOLO parity fixture must be fully static"
            )
    return shape


def main():
    if len(sys.argv) != 2:
        fail(f"expected exactly one argument (output dir), got {len(sys.argv) - 1}")

    output_dir = os.path.abspath(sys.argv[1])
    if output_dir == "/workspace" or output_dir.startswith("/workspace/"):
        fail("refusing to write under /workspace")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "yolo26n.onnx")
    image_path = os.path.join(output_dir, IMAGE_MODEL_NAME)
    tensor_path = os.path.join(output_dir, INPUT_TENSOR_NAME)

    try:
        import onnx
        import torch
        import ultralytics
    except ImportError as exc:
        fail(f"required Python package missing: {exc}")

    check_version(
        "ultralytics",
        getattr(ultralytics, "__version__", ""),
        PINNED_ULTRALYTICS_VERSION,
    )
    check_version_prefix("torch", getattr(torch, "__version__", ""), PINNED_TORCH_VERSION_PREFIX)
    check_version_prefix("onnx", getattr(onnx, "__version__", ""), PINNED_ONNX_VERSION_PREFIX)

    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    exported = model.export(format="onnx", imgsz=640, opset=OPSET_VERSION, nms=True)
    exported = os.path.abspath(exported)
    if os.path.normpath(exported) != os.path.normpath(model_path):
        shutil.move(exported, model_path)

    graph = onnx.load(model_path)

    opsets = {entry.domain or "": entry.version for entry in graph.opset_import}
    for domain, version in opsets.items():
        if version != OPSET_VERSION:
            fail(f"opset_import {domain!r} is version {version}, expected {OPSET_VERSION}")

    initializers = {init.name for init in graph.graph.initializer}
    inputs = [i for i in graph.graph.input if i.name not in initializers]
    if len(inputs) != 1:
        fail(f"expected a single graph input, found {len(inputs)}")
    input_shape = static_shape(inputs[0])
    if input_shape != EXPECTED_INPUT_SHAPE:
        fail(f"input shape is {input_shape}, expected {EXPECTED_INPUT_SHAPE}")

    outputs = list(graph.graph.output)
    if len(outputs) != 1:
        fail(f"expected a single graph output, found {len(outputs)}")
    output_shape = static_shape(outputs[0])
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        fail(f"output shape is {output_shape}, expected {EXPECTED_OUTPUT_SHAPE}")

    op_types = [node.op_type for node in graph.graph.node]
    op_set = set(op_types)

    unsupported = sorted(op_set - SUPPORTED_OPS)
    if unsupported:
        fail(f"graph contains op(s) the neuriplo engine does not implement: {unsupported}")

    # Packet Y11: real photo + deterministic tensor, both into the build tree.
    if os.path.exists(image_path):
        os.remove(image_path)
    download_and_verify_image(image_path)
    if sha256_of(image_path) != PINNED_IMAGE_SHA256:
        fail("post-write verification of the pinned image failed")

    if os.path.exists(tensor_path):
        os.remove(tensor_path)
    build_input_tensor(image_path, tensor_path)
    if os.path.getsize(tensor_path) != TENSOR_ELEMENTS * 4:
        fail(f"tensor bin size is wrong: {os.path.getsize(tensor_path)} bytes")

    histogram = {op: op_types.count(op) for op in sorted(op_set)}
    print(f"provisioned {os.path.abspath(model_path)}")
    print(f"provisioned {os.path.abspath(image_path)} (sha256 {PINNED_IMAGE_SHA256[:12]}...)")
    print(f"nodes={len(op_types)} ops={histogram}")


if __name__ == "__main__":
    main()
