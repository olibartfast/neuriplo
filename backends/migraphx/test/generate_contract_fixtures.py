#!/usr/bin/env python3
"""Generate deterministic typed ONNX graphs for MIGraphX component tests.

These are synthetic contracts, not learned-model validation evidence.
"""
import argparse
from pathlib import Path

import onnx
from onnx import TensorProto as T, helper as h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kind in [("int64", T.INT64), ("int32", T.INT32)]:
        nodes = [h.make_node("Cast", ["sizes"], ["cast"], to=T.FLOAT),
                 h.make_node("Add", ["image", "cast"], ["output"])]
        inputs = [h.make_tensor_value_info("image", T.FLOAT, [1, 2]),
                  h.make_tensor_value_info("sizes", kind, [1, 2])]
        outputs = [h.make_tensor_value_info("output", T.FLOAT, [1, 2])]
        save(args.output_dir, "mixed-" + suffix, nodes, inputs, outputs)
    nodes = [h.make_node("Add", ["sizes", "sizes"], ["output"])]
    save(args.output_dir, "integer-output", nodes,
         [h.make_tensor_value_info("sizes", T.INT64, [1, 2])],
         [h.make_tensor_value_info("output", T.INT64, [1, 2])])


def save(root, name, nodes, inputs, outputs):
    model = h.make_model(h.make_graph(nodes, name, inputs, outputs),
                         opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, root / (name + ".onnx"))


if __name__ == "__main__":
    main()
