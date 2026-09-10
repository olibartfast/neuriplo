#!/usr/bin/env python3
"""Generate small, deterministic CPU DALI pipelines used by DALIInferTest."""

from pathlib import Path

from nvidia.dali import fn, pipeline_def, types


def make_pipeline(outputs, batch_size=1, output_dtype=None, output_ndim=None):
    kwargs = {"batch_size": batch_size, "num_threads": 1, "device_id": 0}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    if output_ndim is not None:
        kwargs["output_ndim"] = output_ndim

    @pipeline_def(**kwargs)
    def pipeline():
        return outputs()

    return pipeline()


def write(root, name, outputs, batch_size=1, output_dtype=None, output_ndim=None):
    make_pipeline(outputs, batch_size, output_dtype, output_ndim).serialize(filename=str(root / f"{name}.dali"))


def source(name="INPUT", dtype=None, ndim=None):
    kwargs = {"name": name}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if ndim is not None:
        kwargs["ndim"] = ndim
    return fn.external_source(**kwargs)


def two_outputs():
    value = source(dtype=types.UINT8, ndim=1)
    return value, fn.cast(value, dtype=types.INT32)


def three_outputs():
    value = source(dtype=types.UINT8, ndim=1)
    return value, fn.cast(value, dtype=types.INT32), fn.cast(value, dtype=types.INT64)


def four_outputs():
    value = source(dtype=types.UINT8, ndim=1)
    return value, fn.cast(value, dtype=types.INT32), fn.cast(value, dtype=types.INT64), fn.cast(value, dtype=types.FLOAT)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    root = parser.parse_args().output_dir
    root.mkdir(parents=True, exist_ok=True)

    dtypes = {
        "uint8": types.UINT8,
        "int32": types.INT32,
        "int64": types.INT64,
        "float": types.FLOAT,
    }
    for name, dtype in dtypes.items():
        write(root, f"single_{name}", lambda dtype=dtype: source(dtype=dtype, ndim=1), output_dtype=[dtype], output_ndim=[1])

    write(root, "single_int32_2d", lambda: source(dtype=types.INT32, ndim=2), output_dtype=[types.INT32], output_ndim=[2])

    # DALI permits output type and rank declarations to be omitted independently.
    write(root, "undeclared_type", lambda: source(dtype=types.UINT8, ndim=1), output_ndim=[1])
    write(root, "undeclared_rank", lambda: source(dtype=types.UINT8, ndim=1), output_dtype=[types.UINT8])
    write(
        root,
        "mixed_declared",
        lambda: (source("DECLARED", dtype=types.INT32, ndim=1), source("UNDECLARED", dtype=types.UINT8, ndim=1)),
        output_dtype=[types.INT32, None], output_ndim=[1, None],
    )

    write(root, "two_outputs", two_outputs, output_dtype=[types.UINT8, types.INT32], output_ndim=[1, 1])
    write(root, "three_outputs", three_outputs, output_dtype=[types.UINT8, types.INT32, types.INT64], output_ndim=[1, 1, 1])
    write(root, "four_outputs", four_outputs, output_dtype=[types.UINT8, types.INT32, types.INT64, types.FLOAT], output_ndim=[1, 1, 1, 1])

    write(root, "unsupported_declared_float16", lambda: source(dtype=types.UINT8, ndim=1), output_dtype=[types.FLOAT16], output_ndim=[1])
    write(root, "unsupported_runtime_float16", lambda: source(dtype=types.FLOAT16, ndim=1), output_ndim=[1])
    write(root, "batch2", lambda: source(dtype=types.INT32, ndim=1), batch_size=2, output_dtype=[types.INT32], output_ndim=[1])

    write(
        root,
        "multi_input_identity",
        lambda: (source("LEFT", dtype=types.INT64, ndim=1), source("RIGHT", dtype=types.INT64, ndim=1)),
        output_dtype=[types.INT64, types.INT64], output_ndim=[1, 1],
    )
    write(root, "changing_shape", lambda: source(dtype=types.INT32, ndim=1), output_dtype=[types.INT32], output_ndim=[1])


if __name__ == "__main__":
    main()
