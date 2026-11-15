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


def batched_matmul():
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 2, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["a", "b"], ["y"], name="matmul")],
        "batched_matmul",
        [a, b],
        [y],
    )
    _save("batched_matmul.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def transpose3():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 2, 3])
    graph = helper.make_graph(
        [helper.make_node("Transpose", ["x"], ["y"], name="transpose", perm=[2, 0, 1])],
        "transpose3",
        [x],
        [y],
    )
    _save("transpose3.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def erf_fixture():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [5])
    graph = helper.make_graph(
        [helper.make_node("Erf", ["x"], ["y"], name="erf")],
        "erf",
        [x],
        [y],
    )
    _save("erf.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def resize_nearest():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales")
    graph = helper.make_graph(
        [
            helper.make_node(
                "Resize",
                ["x", "", "scales"],
                ["y"],
                name="resize",
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
        ],
        "resize_nearest",
        [x],
        [y],
        initializer=[scales],
    )
    _save("resize_nearest.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def split_fixture():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 2])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1, 2, 2, 2])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 2, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("Split", ["x"], ["y0", "y1"], name="split", axis=1)],
        "split",
        [x],
        [y0, y1],
    )
    _save("split.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def topk_fixture():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 2])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2])
    k = numpy_helper.from_array(np.array([2], dtype=np.int64), name="k")
    graph = helper.make_graph(
        [helper.make_node("TopK", ["x", "k"], ["values", "indices"], name="topk", axis=-1, largest=1, sorted=1)],
        "topk",
        [x],
        [values, indices],
        initializer=[k],
    )
    _save("topk.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def expand_tile():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 2])
    shape = numpy_helper.from_array(np.array([1, 1, 2], dtype=np.int64), name="shape")
    repeats = numpy_helper.from_array(np.array([1, 2, 1], dtype=np.int64), name="repeats")
    graph = helper.make_graph(
        [
            helper.make_node("Expand", ["x", "shape"], ["expanded"], name="expand"),
            helper.make_node("Tile", ["expanded", "repeats"], ["y"], name="tile"),
        ],
        "expand_tile",
        [x],
        [y],
        initializer=[shape, repeats],
    )
    _save("expand_tile.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def gather_elements_fixture():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 4, 2])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("GatherElements", ["data", "indices"], ["y"], name="gather_elements", axis=1)],
        "gather_elements",
        [data, indices],
        [y],
    )
    _save("gather_elements.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def slice_ops():
    reverse_x = helper.make_tensor_value_info("reverse_x", TensorProto.FLOAT, [4, 2])
    reverse_y = helper.make_tensor_value_info("reverse_y", TensorProto.FLOAT, [4, 2])
    step_x = helper.make_tensor_value_info("step_x", TensorProto.FLOAT, [2, 5])
    step_y = helper.make_tensor_value_info("step_y", TensorProto.FLOAT, [2, 4])
    initializers = [
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="reverse_start"),
        numpy_helper.from_array(np.array([np.iinfo(np.int64).min], dtype=np.int64), name="reverse_end"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="reverse_axes"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="reverse_steps"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="step_start"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), name="step_end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="step_axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="step_steps"),
    ]
    nodes = [
        helper.make_node(
            "Slice",
            ["reverse_x", "reverse_start", "reverse_end", "reverse_axes", "reverse_steps"],
            ["reverse_y"],
            name="reverse",
        ),
        helper.make_node(
            "Slice",
            ["step_x", "step_start", "step_end", "step_axes", "step_steps"],
            ["step_y"],
            name="step",
        ),
    ]
    graph = helper.make_graph(nodes, "slice_ops", [reverse_x, step_x], [reverse_y, step_y], initializer=initializers)
    _save("slice_ops.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def pad_fixture():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])
    pads = numpy_helper.from_array(np.array([0, 0, 1, 2], dtype=np.int64), name="pads")
    graph = helper.make_graph(
        [helper.make_node("Pad", ["x", "pads"], ["y"], name="pad", mode="constant")],
        "pad",
        [x],
        [y],
        initializer=[pads],
    )
    _save("pad.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def reduce_ops():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    mean = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2, 3, 1])
    maximum = helper.make_tensor_value_info("maximum", TensorProto.FLOAT, [2, 3])
    total = helper.make_tensor_value_info("total", TensorProto.FLOAT, [2, 3, 1])
    axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes")
    nodes = [
        helper.make_node("ReduceMean", ["x", "axes"], ["mean"], name="mean", keepdims=1),
        helper.make_node("ReduceMax", ["x", "axes"], ["maximum"], name="maximum", keepdims=0),
        helper.make_node("ReduceSum", ["x", "axes"], ["total"], name="total", keepdims=1),
    ]
    graph = helper.make_graph(nodes, "reduce_ops", [x], [mean, maximum, total], initializer=[axes])
    _save("reduce_ops.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def shape_fold():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="slice_start"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), name="slice_end"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="slice_axes"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), name="gather_index"),
    ]
    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape"),
        helper.make_node("Slice", ["shape", "slice_start", "slice_end", "slice_axes"], ["first_two"], name="slice"),
        helper.make_node("Gather", ["shape", "gather_index"], ["last"], name="gather", axis=0),
        helper.make_node("Concat", ["first_two", "last"], ["target"], name="concat", axis=0),
        helper.make_node("Reshape", ["x", "target"], ["y"], name="reshape"),
    ]
    graph = helper.make_graph(nodes, "shape_fold", [x], [y], initializer=initializers)
    _save("shape_fold.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def broadcast_edges():
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["y"], name="add")],
        "broadcast_edges",
        [a, b],
        [y],
    )
    _save("broadcast_edges.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def negative_ops():
    x2 = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [2, 3])
    i2 = helper.make_tensor_value_info("i2", TensorProto.INT64, [2, 3])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph(
        [helper.make_node("GatherElements", ["x2", "i2"], ["y2"], name="gather_elements_rank2", axis=1)],
        "gather_elements_rank2",
        [x2, i2],
        [y2],
    )
    _save("gather_elements_rank2.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))

    x3 = helper.make_tensor_value_info("x3", TensorProto.FLOAT, [1, 4, 2])
    i3 = helper.make_tensor_value_info("i3", TensorProto.INT64, [1, 2, 2])
    y3 = helper.make_tensor_value_info("y3", TensorProto.FLOAT, [1, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("GatherElements", ["x3", "i3"], ["y3"], name="gather_elements_oob", axis=1)],
        "gather_elements_oob",
        [x3, i3],
        [y3],
    )
    _save("gather_elements_oob.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))

    x4 = helper.make_tensor_value_info("x4", TensorProto.FLOAT, [2, 5])
    y4 = helper.make_tensor_value_info("y4", TensorProto.FLOAT, [2, 3])
    starts = numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([5], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([2], dtype=np.int64), name="steps")
    graph = helper.make_graph(
        [helper.make_node("Slice", ["x4", "starts", "ends", "axes", "steps"], ["y4"], name="slice_step2")],
        "slice_step2",
        [x4],
        [y4],
        initializer=[starts, ends, axes, steps],
    )
    _save("slice_step2.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))

    x5 = helper.make_tensor_value_info("x5", TensorProto.FLOAT, [2, 2])
    y5 = helper.make_tensor_value_info("y5", TensorProto.FLOAT, [3, 2])
    pads = numpy_helper.from_array(np.array([1, 0, 0, 0], dtype=np.int64), name="pads")
    graph = helper.make_graph(
        [helper.make_node("Pad", ["x5", "pads"], ["y5"], name="pad_begin", mode="constant")],
        "pad_begin",
        [x5],
        [y5],
        initializer=[pads],
    )
    _save("pad_begin.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def grid_sample():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    grid = numpy_helper.from_array(
        np.array([[-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]], dtype=np.float32).reshape(1, 2, 2, 2),
        name="grid",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "GridSample",
                ["x", "grid"],
                ["y"],
                name="grid_sample",
                align_corners=0,
                mode="linear",
                padding_mode="zeros",
            )
        ],
        "grid_sample",
        [x],
        [y],
        initializer=[grid],
    )
    _save("grid_sample.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def matmul_vector():
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["a", "b"], ["y"], name="matmul_vector")],
        "matmul_vector",
        [a, b],
        [y],
    )
    _save("matmul_vector.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


def unsupported_op():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph(
        [helper.make_node("Softplus", ["x"], ["y"], name="softplus")],
        "unsupported_softplus",
        [x],
        [y],
    )
    _save("unsupported_softplus.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))


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
    batched_matmul()
    transpose3()
    erf_fixture()
    resize_nearest()
    split_fixture()
    topk_fixture()
    expand_tile()
    gather_elements_fixture()
    slice_ops()
    pad_fixture()
    reduce_ops()
    shape_fold()
    broadcast_edges()
    negative_ops()
    grid_sample()
    matmul_vector()
    unsupported_op()
    dynamic_shape()


if __name__ == "__main__":
    main()
