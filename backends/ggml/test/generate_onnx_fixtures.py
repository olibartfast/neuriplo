#!/usr/bin/env python3
"""Regenerate the tiny ONNX fixtures used by GGMLOnnxTest.

Run from the repository root:
    python3 backends/ggml/test/generate_onnx_fixtures.py

The fixtures are checked in so the C++ tests do not need the Python `onnx`
package at build or test time.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _save(name, model):
    onnx.checker.check_model(model)
    path = FIXTURE_DIR / name
    onnx.save(model, path)
    print(f"wrote {path}")


def multi_input_add():
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    c = numpy_helper.from_array(np.full((2, 3), 0.5, dtype=np.float32), name="c")
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["a", "b"], ["ab"], name="add_ab"),
            helper.make_node("Add", ["ab", "c"], ["y"], name="add_c"),
        ],
        "multi_input_add",
        [a, b],
        [y],
        initializer=[c],
    )
    _save("multi_input_add.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def int64_gather():
    data = helper.make_tensor_value_info("indices", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    table = numpy_helper.from_array(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32), name="table")
    graph = helper.make_graph(
        [helper.make_node("Gather", ["table", "indices"], ["y"], name="gather", axis=0)],
        "int64_gather",
        [data],
        [y],
        initializer=[table],
    )
    _save("int64_gather.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def attention_block():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    initializers = [
        numpy_helper.from_array(np.array([[0.1, 0.0], [0.0, 0.2]], dtype=np.float32), name="wq"),
        numpy_helper.from_array(np.array([0.0, 0.0], dtype=np.float32), name="bq"),
        numpy_helper.from_array(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), name="wk"),
        numpy_helper.from_array(np.array([0.0, 0.0], dtype=np.float32), name="bk"),
        numpy_helper.from_array(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), name="wv"),
        numpy_helper.from_array(np.array([0.0, 0.0], dtype=np.float32), name="bv"),
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two"),
    ]
    nodes = [
        helper.make_node("Gemm", ["x", "wq", "bq"], ["q"], name="q_proj", transB=1),
        helper.make_node("Gemm", ["x", "wk", "bk"], ["k"], name="k_proj", transB=1),
        helper.make_node("Gemm", ["x", "wv", "bv"], ["v"], name="v_proj", transB=1),
        helper.make_node("Transpose", ["k"], ["kt"], name="k_transpose", perm=[1, 0]),
        helper.make_node("MatMul", ["q", "kt"], ["scores"], name="scores"),
        helper.make_node("Sqrt", ["two"], ["sqrt_two"], name="sqrt_two"),
        helper.make_node("Div", ["scores", "sqrt_two"], ["scaled"], name="scale"),
        helper.make_node("Softmax", ["scaled"], ["probs"], name="softmax", axis=-1),
        helper.make_node("MatMul", ["probs", "v"], ["y"], name="output"),
    ]
    graph = helper.make_graph(nodes, "attention_block", [x], [y], initializer=initializers)
    _save("attention_block.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def shape_ops():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])
    initializers = [
        numpy_helper.from_array(np.array([3, 4], dtype=np.int64), name="target"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_axis"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="squeeze_axis"),
    ]
    nodes = [
        helper.make_node("Relu", ["x"], ["relu"], name="relu"),
        helper.make_node("Sigmoid", ["x"], ["sigmoid"], name="sigmoid"),
        helper.make_node("Concat", ["relu", "sigmoid"], ["concat"], name="concat", axis=1),
        helper.make_node("Reshape", ["concat", "target"], ["reshaped"], name="reshape"),
        helper.make_node("Unsqueeze", ["reshaped", "unsqueeze_axis"], ["unsqueezed"], name="unsqueeze"),
        helper.make_node("Squeeze", ["unsqueezed", "squeeze_axis"], ["y"], name="squeeze"),
    ]
    graph = helper.make_graph(nodes, "shape_ops", [x], [y], initializer=initializers)
    _save("shape_ops.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def conv_pool():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2, 2])
    rng = np.random.default_rng(7)
    weight = numpy_helper.from_array(rng.normal(0, 0.2, (3, 2, 3, 3)).astype(np.float32), name="weight")
    bias = numpy_helper.from_array(np.array([0.1, -0.2, 0.3], dtype=np.float32), name="bias")
    nodes = [
        helper.make_node("Conv", ["x", "weight", "bias"], ["conv"], name="conv", pads=[1, 1, 1, 1]),
        helper.make_node("MaxPool", ["conv"], ["y"], name="pool", kernel_shape=[2, 2], strides=[2, 2]),
    ]
    graph = helper.make_graph(nodes, "conv_pool", [x], [y], initializer=[weight, bias])
    _save("conv_pool.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def depthwise_conv():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])
    rng = np.random.default_rng(11)
    weight = numpy_helper.from_array(rng.normal(0, 0.2, (2, 1, 3, 3)).astype(np.float32), name="weight")
    bias = numpy_helper.from_array(np.array([0.05, -0.05], dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "weight", "bias"], ["y"], name="dwconv", group=2, pads=[1, 1, 1, 1])],
        "depthwise_conv",
        [x],
        [y],
        initializer=[weight, bias],
    )
    _save("depthwise_conv.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def constant_of_shape():
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    shape = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="shape")
    value = numpy_helper.from_array(np.array(1.5, dtype=np.float32), name="value")
    graph = helper.make_graph(
        [helper.make_node("ConstantOfShape", ["shape"], ["y"], name="fill", value=value)],
        "constant_of_shape",
        [],
        [y],
        initializer=[shape],
    )
    _save("constant_of_shape.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def gemm_scaled():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    initializers = [
        numpy_helper.from_array(np.array([[0.5, 1.0], [1.5, -0.5]], dtype=np.float32), name="weight"),
        numpy_helper.from_array(np.array([0.2, -0.3], dtype=np.float32), name="bias"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm", ["x", "weight", "bias"], ["y"], name="gemm", alpha=0.5, beta=0.25, transB=1
            )
        ],
        "gemm_scaled",
        [x],
        [y],
        initializer=initializers,
    )
    _save("gemm_scaled.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def unsupported_op():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("GridSample", ["x", "grid"], ["y"], name="grid_sample")],
        "unsupported_grid_sample",
        [x, grid],
        [y],
    )
    _save("unsupported_grid_sample.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def dynamic_shape():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3, 4])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"], name="relu")],
        "dynamic_shape",
        [x],
        [y],
    )
    _save("dynamic_shape.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    multi_input_add()
    int64_gather()
    attention_block()
    shape_ops()
    conv_pool()
    depthwise_conv()
    constant_of_shape()
    gemm_scaled()
    unsupported_op()
    dynamic_shape()


if __name__ == "__main__":
    main()
