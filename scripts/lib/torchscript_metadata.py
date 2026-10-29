"""Capture model input dtypes at export; consumers read them from the artifact."""

import json
from pathlib import Path
import zipfile

TORCHSCRIPT_FILE = "neuriplo_input_dtypes.txt"
AOTI_KEY = "NEURIPLO_INPUT_DTYPES"
AOTI_SHAPES_KEY = "NEURIPLO_INPUT_SHAPES"
SHAPES_HEADER = "NEURIPLO_INPUT_SHAPES_V1"
HEADER = "NEURIPLO_INPUT_DTYPES_V1"
TAGS = {"FP32", "INT32", "INT64", "UINT8", "INT8", "BOOL"}


def encode_dtypes(tensors):
    import torch

    names = {torch.float32: "FP32", torch.int32: "INT32", torch.int64: "INT64",
             torch.uint8: "UINT8", torch.int8: "INT8", torch.bool: "BOOL"}
    values = tuple(tensors)
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        raise ValueError("Input metadata requires a non-empty flat tensor input sequence")
    try:
        tags = [names[value.dtype] for value in values]
    except KeyError as error:
        raise ValueError(f"Unsupported model input dtype: {error.args[0]}") from error
    return " ".join([HEADER, str(len(tags)), *tags])


def decode_dtypes(text, expected=None):
    if isinstance(text, bytes):
        text = text.decode("ascii")
    if not isinstance(text, str):
        raise ValueError("Input datatype metadata must be text")
    fields = text.split()
    if len(fields) < 2 or fields[0] != HEADER:
        raise ValueError("Invalid input datatype metadata header")
    count = int(fields[1])
    if count < 1 or len(fields) != count + 2 or (expected is not None and count != expected):
        raise ValueError("Input datatype metadata count mismatch")
    if not set(fields[2:]) <= TAGS:
        raise ValueError("Unsupported input datatype metadata tag")
    return fields[2:]


def save_torchscript(module, path, example_inputs):
    inputs = tuple(example_inputs)
    metadata = encode_dtypes(inputs)
    nodes = list(module.forward.graph.inputs())[1:]
    if len(nodes) != len(inputs):
        raise ValueError("Example inputs do not match the TorchScript forward signature")
    for node, value in zip(nodes, inputs, strict=True):
        if node.type().kind() != "TensorType":
            raise ValueError("Only flat tensor input signatures are supported")
        known = node.type().dtype()
        if known is not None and known != value.dtype:
            raise ValueError("Example input dtype disagrees with the traced graph")
    module.save(str(path), _extra_files={TORCHSCRIPT_FILE: metadata})


def has_torchscript_metadata(path):
    import torch

    try:
        files = {TORCHSCRIPT_FILE: ""}
        module = torch.jit.load(str(path), _extra_files=files)
        decode_dtypes(files[TORCHSCRIPT_FILE], len(list(module.forward.graph.inputs())) - 1)
        return True
    except (OSError, RuntimeError, ValueError, UnicodeError):
        return False


def aoti_metadata(exported_program):
    """Use the compiler graph's actual user tensor specs, not a caller dtype list."""
    from torch.export.graph_signature import InputKind, TensorArgument

    nodes = {node.name: node for node in exported_program.graph.nodes}
    inputs = []
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind != InputKind.USER_INPUT:
            continue
        if not isinstance(spec.arg, TensorArgument):
            raise ValueError("AOTInductor input metadata supports tensor inputs only")
        inputs.append(nodes[spec.arg.name].meta["val"])
    shapes = [SHAPES_HEADER, str(len(inputs))]
    for value in inputs:
        if any(type(dimension) is not int or dimension <= 0 for dimension in value.shape):
            raise ValueError("AOTInductor input contracts currently require static positive shapes")
        if not value.is_contiguous():
            raise ValueError("AOTInductor input contracts require contiguous input tensors")
        shapes.extend([str(value.ndim), *(str(dimension) for dimension in value.shape)])
    return {AOTI_KEY: encode_dtypes(inputs), AOTI_SHAPES_KEY: " ".join(shapes)}


def has_aoti_metadata(path):
    try:
        with zipfile.ZipFile(Path(path)) as archive:
            for name in archive.namelist():
                if name.endswith("_metadata.json"):
                    metadata = json.loads(archive.read(name))
                    if AOTI_KEY in metadata and AOTI_SHAPES_KEY in metadata:
                        decode_dtypes(metadata[AOTI_KEY])
                        return metadata[AOTI_SHAPES_KEY].startswith(SHAPES_HEADER + " ")
    except (OSError, ValueError, zipfile.BadZipFile):
        pass
    return False
