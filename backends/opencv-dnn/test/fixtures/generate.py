#!/usr/bin/env python3
"""Regenerate relu.onnx with onnx==1.22.0; no downloaded weights."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [helper.make_node("Relu", ["input"], ["output"])],
    "opencv-compatibility",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 3])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 3])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("relu.onnx"))
