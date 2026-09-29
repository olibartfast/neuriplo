#!/usr/bin/env python3
"""Foreign-language proof ([T-16], [V-13]): drive the installed neuriplo C ABI
from Python through ctypes alone -- no binding library, no source tree.

    python3 smoke_ctypes.py <install-prefix> <plugin_dir>

Loads libneuriplo from <install-prefix>, lists backends, creates FIXTURE_GOOD
from <plugin_dir> with model "ok", checks its metadata, infers [1,2,3,4] and
expects [2,4,6,8], checks an error path, and releases everything. Prints
"OK FIXTURE_GOOD 2 4 6 8" and exits 0 on success.
"""

import ctypes
import pathlib
import sys

NEURIPLO_C_API_VERSION = 1
STATUS_OK = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_BACKEND_NOT_FOUND = 2
DTYPE_FLOAT32 = 0


class Dims(ctypes.Structure):
    _fields_ = [("dims", ctypes.POINTER(ctypes.c_int64)), ("ndim", ctypes.c_size_t)]


class EngineConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("use_gpu", ctypes.c_int32),
        ("backend_id", ctypes.c_char_p),
        ("model_path", ctypes.c_char_p),
        ("batch_size", ctypes.c_size_t),
        ("input_sizes", ctypes.POINTER(Dims)),
        ("n_input_sizes", ctypes.c_size_t),
        ("plugin_dir", ctypes.c_char_p),
    ]


class TensorInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("dtype", ctypes.c_int32),
        ("name", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("ndim", ctypes.c_size_t),
        ("batch_size", ctypes.c_size_t),
    ]


class InputView(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size_bytes", ctypes.c_size_t)]


class TensorView(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("dtype", ctypes.c_int32),
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
        ("element_count", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("ndim", ctypes.c_size_t),
    ]


def find_library(prefix: pathlib.Path) -> pathlib.Path:
    names = ("libneuriplo.so", "libneuriplo.dylib", "neuriplo.dll")
    for name in names:
        hits = sorted(prefix.rglob(name))
        if hits:
            return hits[0]
    raise SystemExit(f"smoke_ctypes: no neuriplo library under {prefix}")


def bind(lib: ctypes.CDLL) -> None:
    status = ctypes.c_int32
    handle = ctypes.c_void_p
    size_p = ctypes.POINTER(ctypes.c_size_t)
    signatures = {
        "neuriplo_api_version": (ctypes.c_uint32, []),
        "neuriplo_status_string": (ctypes.c_char_p, [status]),
        "neuriplo_last_error": (ctypes.c_char_p, []),
        "neuriplo_available_backends": (status, [ctypes.c_char_p, ctypes.POINTER(handle)]),
        "neuriplo_backend_list_count": (status, [handle, size_p]),
        "neuriplo_backend_list_get": (status, [handle, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p)]),
        "neuriplo_backend_list_release": (None, [handle]),
        "neuriplo_engine_create": (status, [ctypes.POINTER(EngineConfig), ctypes.POINTER(handle)]),
        "neuriplo_engine_destroy": (None, [handle]),
        "neuriplo_engine_backend_id": (status, [handle, ctypes.POINTER(ctypes.c_char_p)]),
        "neuriplo_engine_input_count": (status, [handle, size_p]),
        "neuriplo_engine_input": (status, [handle, ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(TensorInfo))]),
        "neuriplo_infer": (status, [handle, ctypes.POINTER(InputView), ctypes.c_size_t, ctypes.POINTER(handle)]),
        "neuriplo_result_output_count": (status, [handle, size_p]),
        "neuriplo_result_output": (status, [handle, ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(TensorView))]),
        "neuriplo_result_release": (None, [handle]),
    }
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <install-prefix> <plugin_dir>", file=sys.stderr)
        return 2
    prefix = pathlib.Path(sys.argv[1]).resolve()
    plugin_dir = str(pathlib.Path(sys.argv[2]).resolve()).encode()

    lib = ctypes.CDLL(str(find_library(prefix)))
    bind(lib)

    def check(status: int, what: str) -> None:
        if status != STATUS_OK:
            raise RuntimeError(
                f"{what}: {lib.neuriplo_status_string(status).decode()} ({status}): {lib.neuriplo_last_error().decode()}"
            )

    assert lib.neuriplo_api_version() == NEURIPLO_C_API_VERSION, "API version"

    # Backend listing.
    blist = ctypes.c_void_p()
    check(lib.neuriplo_available_backends(plugin_dir, ctypes.byref(blist)), "neuriplo_available_backends")
    try:
        count = ctypes.c_size_t()
        check(lib.neuriplo_backend_list_count(blist, ctypes.byref(count)), "neuriplo_backend_list_count")
        ids = []
        for i in range(count.value):
            bid = ctypes.c_char_p()
            check(lib.neuriplo_backend_list_get(blist, i, ctypes.byref(bid)), "neuriplo_backend_list_get")
            ids.append(bid.value.decode())
    finally:
        lib.neuriplo_backend_list_release(blist)
    assert "FIXTURE_GOOD" in ids, f"FIXTURE_GOOD not in {ids}"

    config = EngineConfig()
    config.struct_size = ctypes.sizeof(EngineConfig)
    config.plugin_dir = plugin_dir
    config.model_path = b"ok"

    # Error path: unknown backend -> BACKEND_NOT_FOUND with a message.
    config.backend_id = b"NO_SUCH_BACKEND"
    engine = ctypes.c_void_p()
    status = lib.neuriplo_engine_create(ctypes.byref(config), ctypes.byref(engine))
    assert status == STATUS_BACKEND_NOT_FOUND, f"expected BACKEND_NOT_FOUND, got {status}"
    assert engine.value is None
    assert b"NO_SUCH_BACKEND" in lib.neuriplo_last_error()

    config.backend_id = b"FIXTURE_GOOD"
    check(lib.neuriplo_engine_create(ctypes.byref(config), ctypes.byref(engine)), "neuriplo_engine_create")
    try:
        backend = ctypes.c_char_p()
        check(lib.neuriplo_engine_backend_id(engine, ctypes.byref(backend)), "neuriplo_engine_backend_id")

        n_inputs = ctypes.c_size_t()
        check(lib.neuriplo_engine_input_count(engine, ctypes.byref(n_inputs)), "neuriplo_engine_input_count")
        assert n_inputs.value == 1
        info = ctypes.POINTER(TensorInfo)()
        check(lib.neuriplo_engine_input(engine, 0, ctypes.byref(info)), "neuriplo_engine_input")
        assert info.contents.dtype == DTYPE_FLOAT32
        assert [info.contents.shape[i] for i in range(info.contents.ndim)] == [1, 4]

        values = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        view = InputView(ctypes.cast(values, ctypes.c_void_p), ctypes.sizeof(values))
        result = ctypes.c_void_p()
        check(lib.neuriplo_infer(engine, ctypes.byref(view), 1, ctypes.byref(result)), "neuriplo_infer")
        try:
            n_out = ctypes.c_size_t()
            check(lib.neuriplo_result_output_count(result, ctypes.byref(n_out)), "neuriplo_result_output_count")
            assert n_out.value == 1
            out = ctypes.POINTER(TensorView)()
            check(lib.neuriplo_result_output(result, 0, ctypes.byref(out)), "neuriplo_result_output")
            tensor = out.contents
            assert tensor.dtype == DTYPE_FLOAT32 and tensor.element_count == 4
            floats = ctypes.cast(tensor.data, ctypes.POINTER(ctypes.c_float))
            got = [floats[i] for i in range(tensor.element_count)]
        finally:
            lib.neuriplo_result_release(result)
        assert got == [2.0, 4.0, 6.0, 8.0], f"unexpected output {got}"
        print(f"OK {backend.value.decode()} " + " ".join(f"{v:g}" for v in got))
    finally:
        lib.neuriplo_engine_destroy(engine)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, RuntimeError) as error:
        print(f"smoke_ctypes: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
