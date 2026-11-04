#!/usr/bin/env python3
"""Regenerate the tiny typed, multi-input OpenCV graph fixture (requires onnx)."""
from pathlib import Path
import onnx
from onnx import TensorProto, helper

float_info = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])
int_info = lambda name: helper.make_tensor_value_info(name, TensorProto.INT64, [1, 2])
graph = helper.make_graph([
    helper.make_node('Cast', ['tokens'], ['float_tokens'], to=TensorProto.FLOAT),
    helper.make_node('Add', ['image', 'float_tokens'], ['sum']),
    helper.make_node('Add', ['tokens', 'offset'], ['token_output']),
], 'mixed_inputs', [float_info('image'), int_info('tokens')],
    [float_info('sum'), int_info('token_output')],
    [helper.make_tensor('offset', TensorProto.INT64, [1], [2**35])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name('mixed_inputs.onnx'))
