# Consumer C API

`include/neuriplo/neuriplo_c.h` is neuriplo's stable, versioned C ABI. Use it to
run inference from any application, whatever language it is written in or
compiler it was built with: C, C++ built with another toolchain, Python, C#/Unity,
Rust, Go, and so on. `include/neuriplo/neuriplo.hpp` is a header-only C++ wrapper
over it, and it is the recommended way in for third-party C++ applications.

The header itself is the authoritative contract: every function's doc comment
states its ownership, lifetime, thread-safety, and errors. This page explains
how the pieces fit together and gives worked examples.

## Which API to use

Two C ABIs point in opposite directions:

```text
  third-party application                           backend plugin
  (C, C++, Python, C#, ...)                      (libneuriplo_backend_*.so)
            |                                                ^
            |  neuriplo_c.h   (consumer ABI, this page)      |  plugin_abi.h   (plugin ABI)
            v                                                |
     +-------------------------- libneuriplo ---------------------------+
     |  engine lifecycle -> backend selection -> compiled-in backend    |
     |                                        \-> plugin loader --------+
     +------------------------------------------------------------------+
```

| ABI | Header | Direction | Implemented by | Called by |
| --- | --- | --- | --- | --- |
| Consumer ABI | `neuriplo/neuriplo_c.h` | applications call neuriplo | `libneuriplo` | your application |
| Plugin ABI | `neuriplo/plugin_abi.h` | backends plug into neuriplo | backend plugins | the neuriplo host |

A consumer never sees the plugin ABI. neuriplo picks a compiled-in backend or
loads a plugin, applies the host's plugin validation (see
[Backend Plugins](PLUGIN_BACKENDS.md)), and runs inference.

The existing C++ API (`InferenceBackendSetup.hpp`, `InferenceInterface`) is
unchanged. It passes STL types and exceptions across the library boundary, so
it is only safe for applications built with the same toolchain as
`libneuriplo`, as [neuriplo-infer](https://github.com/olibartfast/neuriplo-infer)
is.

## Installing and linking

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN
cmake --build build
cmake --install build --prefix /opt/neuriplo
```

Install rules are generated when neuriplo is the top-level project. Set
`-DNEURIPLO_INSTALL=OFF` to disable them, or `ON` to enable them under
`add_subdirectory`. The prefix contains:

| Path | Contents |
| --- | --- |
| `lib/libneuriplo.so` (`bin/neuriplo.dll` on Windows) | the library |
| `include/neuriplo/neuriplo_c.h`, `neuriplo.hpp`, `plugin_abi.h` | C ABI, C++ wrapper, plugin ABI |
| `include/InferenceBackendSetup.hpp`, ... | the existing C++ API and its include closure |
| `lib/cmake/neuriplo/` | CMake package, target `neuriplo::neuriplo` |
| `lib/pkgconfig/neuriplo.pc` | pkg-config file (relocatable) |

CMake:

```cmake
find_package(neuriplo CONFIG REQUIRED)   # -DCMAKE_PREFIX_PATH=/opt/neuriplo
target_link_libraries(app PRIVATE neuriplo::neuriplo)
```

pkg-config:

```bash
cc -std=c99 app.c $(PKG_CONFIG_PATH=/opt/neuriplo/lib/pkgconfig pkg-config --cflags --libs neuriplo)
```

**Dependencies.** The C ABI and the C++ wrapper need nothing but neuriplo.
glog and every backend runtime are private to the shared library. They must
be present at run time, where the backends need them, but not at build time.
If you use the installed **C++ API** headers (`InferenceBackendSetup.hpp`),
you must also add `find_package(glog)` and link `glog::glog` yourself. The
package does not do that for you.

## Lifecycle

```text
neuriplo_engine_create(config) --> engine
   neuriplo_engine_input / _output (metadata views, owned by the engine)
   neuriplo_infer(engine, inputs) --> result
       neuriplo_result_output (views, owned by the result)
       neuriplo_result_release(result)
neuriplo_engine_destroy(engine)
```

1. **Configure.** Zero a `neuriplo_engine_config_t`, set `struct_size =
   sizeof(config)`, then set only the fields you need. A zero field means
   "default":
   - `backend_id`: NULL or "" selects the build's default backend.
   - `model_path`: required.
   - `batch_size`: 0 means 1.
   - `input_sizes`: optional.
   - `plugin_dir`: a directory to scan for `libneuriplo_backend_*` plugins.
     `NEURIPLO_PLUGIN_DIR` is honoured as well.
2. **Create.** `neuriplo_engine_create` scans plugins, resolves the backend id,
   and loads the model. It returns `BACKEND_NOT_FOUND` when the id is not
   available, with a message that lists the ids that are. It returns
   `MODEL_LOAD` when the backend fails to load the model, with a message that
   names the backend and the path. `neuriplo_engine_backend_id` tells you which
   backend was actually chosen.
3. **Inspect.** `neuriplo_engine_input_count` / `_input` and `_output_count` /
   `_output` return `neuriplo_tensor_info_t` views: name, dtype, shape (a
   dimension may be -1 when dynamic), and batch size.
4. **Infer.** Pass one `neuriplo_input_view_t` (pointer + byte size) per model
   input. The bytes follow the layout the metadata describes, which is the
   same raw-bytes contract as the C++ `get_infer_results`. On success you get
   a `neuriplo_result_t`.
5. **Read outputs.** `neuriplo_result_output` returns `neuriplo_tensor_view_t`:
   dtype (FLOAT32, INT32, INT64, or UINT8), data, size in bytes, element count,
   and a concrete shape. A zero-element output, such as "no detections", is
   valid: its data may be NULL and its size is 0.
6. **Release.** Call `neuriplo_result_release` once per result and
   `neuriplo_engine_destroy` once per engine. Both accept NULL.

## Ownership and lifetimes

| Object | Owned by | Valid until |
| --- | --- | --- |
| `neuriplo_engine_t*` | caller | `neuriplo_engine_destroy` |
| `neuriplo_tensor_info_t*`, its `name` and `shape` | the engine | `neuriplo_engine_destroy` (stable across any number of infers) |
| `neuriplo_result_t*` | caller | `neuriplo_result_release`, **even after its engine is destroyed** |
| `neuriplo_tensor_view_t*`, its `data` and `shape` | the result | `neuriplo_result_release` |
| `neuriplo_backend_list_t*` and its id strings | caller | `neuriplo_backend_list_release` |
| input bytes | caller | the end of the `neuriplo_infer` call (the library copies them) |
| `neuriplo_last_error()` string | the library, per thread | the next status-returning call on that thread |
| `neuriplo_status_string()` string | the library | forever (static) |

Inputs are copied exactly once, into the buffer the backend interface takes.
Outputs are moved into the result without a copy.

## Errors

Every function either returns `neuriplo_status_t` or is documented as
infallible. The infallible ones are `neuriplo_api_version`,
`neuriplo_status_string`, `neuriplo_last_error`, and the destroy/release
functions. No C++ exception ever crosses the boundary.

| Status | Meaning |
| --- | --- |
| `OK` | success; `neuriplo_last_error()` is now "" |
| `INVALID_ARGUMENT` | NULL or malformed argument, bad `struct_size`, index out of range. The message starts with the function name. |
| `BACKEND_NOT_FOUND` | the requested (or default) backend is not available |
| `MODEL_LOAD` | the backend could not create the engine or load the model |
| `INFERENCE` | the backend rejected the inputs or failed. The engine stays usable. |
| `OUT_OF_MEMORY` | allocation failure |
| `INTERNAL` | anything else |
| `UNIMPLEMENTED` | reserved |

On failure, every out-pointer is set to NULL/0 and `neuriplo_last_error()`
returns a message for the calling thread only. The message is thread-local,
so read it on the thread that made the call, before you make the next call.

**Extensible structs.** `neuriplo_engine_config_t` starts with `struct_size`.
A value smaller than the library's struct is `INVALID_ARGUMENT`. A larger
value is accepted only if every byte the library does not know about is zero.
A newer caller that uses a new field therefore fails loudly on an older
library, instead of having the field silently ignored.

## Threading

- Every function may be called from any thread. Functions that take no engine
  or result handle are thread-safe. That covers creation, backend listing, and
  plugin scanning.
- Different engines are fully independent and can run concurrently.
- On the **same** engine, `neuriplo_infer` calls are serialised by a per-engine
  lock: they are safe, but they do not overlap. For parallel inference on one
  model, create one engine per worker thread. Metadata queries are lock-free.
- Destroying an engine, or releasing a result, while another thread is still
  using it is undefined behaviour. Order the destroy after every other use.

## Logging

By default, neuriplo and its plugins log through glog to stderr, exactly as
before. You never need to initialise glog. To receive messages as well:

```c
static void on_log(neuriplo_log_level_t level, const char* message, void* user_data) {
    /* thread-safe; must not call neuriplo; must not unwind */
}
neuriplo_set_log_callback(on_log, NEURIPLO_LOG_LEVEL_WARNING, my_context);
/* ... */
neuriplo_set_log_callback(NULL, NEURIPLO_LOG_LEVEL_INFO, NULL);   /* remove */
```

- The callback is **additional**: stderr output continues.
- It is invoked synchronously, on whichever thread logged, possibly several
  threads at once. `message` has no trailing newline and is valid only during
  the call.
- When `neuriplo_set_log_callback` returns, no invocation of the previous
  callback is running and none will start. You can free its `user_data`, or
  let a managed delegate be collected, right after that.
- Plugins are scanned once per process. Install the callback before the first
  engine is created or backends are listed if you want to see the first scan.
- The C++ wrapper does not wrap this function in v1. Call it directly.

## Versioning

- `NEURIPLO_C_API_VERSION` (currently **1**) is compiled into your application.
  `neuriplo_api_version()` returns the value of the library you actually
  loaded. Check that they match at start-up.
- This version is independent of the plugin ABI version and of the neuriplo
  release version.
- **Compatible changes, no version bump:** adding a function, appending a field
  after `struct_size`, adding an enumerator. Treat an unknown enumerator as
  "unsupported", never as a library bug.
- **Breaking changes, version bump:** removing or changing anything that
  already exists.
- Every enum is 32 bits wide. Structs without `struct_size`
  (`neuriplo_dims_t`, `neuriplo_input_view_t`) are frozen.
- CI enforces this. `scripts/abi/check_symbols.sh` compares the exported
  `neuriplo_*` symbols in both directions against
  `scripts/abi/neuriplo_c.symbols`, and a C99 translation unit pins every
  struct layout, enum width, and function type. The CMake package uses
  `SameMajorVersion` compatibility.

## Examples

All of the examples below run the first-party `FIXTURE_GOOD` test plugin,
which doubles its input. Its model path is `"ok"`. The build tree puts it in
`<build>/plugin_fixtures` when tests are enabled. With a real backend, you
would pass that backend's id (or NULL for the default) and a real model path
instead. Complete, tested versions of the C, C++, and Python examples live in
[test/consumer/](../test/consumer/).

### C

```c
#include <neuriplo/neuriplo_c.h>
#include <stdio.h>
#include <string.h>

int run(const char* plugin_dir) {
    neuriplo_engine_config_t config;
    neuriplo_engine_t* engine = NULL;
    neuriplo_result_t* result = NULL;
    const neuriplo_tensor_view_t* out = NULL;
    const float in[4] = {1, 2, 3, 4};
    neuriplo_input_view_t input = {in, sizeof(in)};
    neuriplo_status_t status;

    memset(&config, 0, sizeof(config));
    config.struct_size = (uint32_t)sizeof(config);
    config.backend_id = "FIXTURE_GOOD";
    config.model_path = "ok";
    config.plugin_dir = plugin_dir;

    status = neuriplo_engine_create(&config, &engine);
    if (status != NEURIPLO_STATUS_OK) {
        fprintf(stderr, "%s: %s\n", neuriplo_status_string(status), neuriplo_last_error());
        return 1;
    }
    status = neuriplo_infer(engine, &input, 1, &result);
    if (status == NEURIPLO_STATUS_OK && neuriplo_result_output(result, 0, &out) == NEURIPLO_STATUS_OK) {
        const float* y = (const float*)out->data;
        printf("%g %g %g %g\n", y[0], y[1], y[2], y[3]);   /* 2 4 6 8 */
    }
    neuriplo_result_release(result);
    neuriplo_engine_destroy(engine);
    return status == NEURIPLO_STATUS_OK ? 0 : 1;
}
```

### C++ (wrapper)

```cpp
#include <neuriplo/neuriplo.hpp>

neuriplo::EngineConfig config;
config.backend_id = "FIXTURE_GOOD";
config.model_path = "ok";
config.plugin_dir = plugin_dir;

try {
    neuriplo::Engine engine(config);                    // move-only, destroys on scope exit
    for (const neuriplo::TensorInfo& in : engine.inputs()) { /* name, dtype, shape */ }

    const std::vector<float> x{1, 2, 3, 4};
    const neuriplo::Result result = engine.infer({x});  // or {neuriplo::InputView(ptr, bytes)}
    const neuriplo::TensorView y = result.output(0);
    const float* values = y.data_as<float>();           // throws Error on a dtype mismatch
    // y.shape(), y.element_count(), y.size_bytes()
} catch (const neuriplo::Error& e) {
    // e.status() is the neuriplo_status_t; e.what() is the library's message
}
```

`neuriplo::backends(plugin_dir)` lists the available ids. `neuriplo::check(status)`
turns any raw C status into an exception, so you can mix the wrapper with
direct C calls, for example to `neuriplo_set_log_callback`. A `Result` stays
valid after its `Engine` is destroyed.

### Python (ctypes)

Only the standard library is needed. Declare each struct with its fields in
header order, and set `argtypes` and `restype` for each function.
[test/consumer/python/smoke_ctypes.py](../test/consumer/python/smoke_ctypes.py)
binds the whole API. The core looks like this:

```python
import ctypes

lib = ctypes.CDLL("/opt/neuriplo/lib/libneuriplo.so")
# ... EngineConfig / InputView / TensorView ctypes.Structure definitions and
#     argtypes/restype declarations, as in smoke_ctypes.py ...

config = EngineConfig(struct_size=ctypes.sizeof(EngineConfig),
                      backend_id=b"FIXTURE_GOOD", model_path=b"ok", plugin_dir=plugin_dir)
engine = ctypes.c_void_p()
if lib.neuriplo_engine_create(ctypes.byref(config), ctypes.byref(engine)) != 0:
    raise RuntimeError(lib.neuriplo_last_error().decode())
try:
    x = (ctypes.c_float * 4)(1, 2, 3, 4)
    view = InputView(ctypes.cast(x, ctypes.c_void_p), ctypes.sizeof(x))
    result = ctypes.c_void_p()
    if lib.neuriplo_infer(engine, ctypes.byref(view), 1, ctypes.byref(result)) != 0:
        raise RuntimeError(lib.neuriplo_last_error().decode())
    out = ctypes.POINTER(TensorView)()
    lib.neuriplo_result_output(result, 0, ctypes.byref(out))
    y = ctypes.cast(out.contents.data, ctypes.POINTER(ctypes.c_float))
    print([y[i] for i in range(out.contents.element_count)])      # [2.0, 4.0, 6.0, 8.0]
    lib.neuriplo_result_release(result)
finally:
    lib.neuriplo_engine_destroy(engine)
```

`ctypes` releases the GIL during foreign calls, so inference running in a
Python thread does not block other Python threads.

### C# / Unity

**Project layout.** Put the native libraries under `Assets/Plugins/` for the
target platform and mark them for that platform in the Inspector:

```text
Assets/Plugins/x86_64/neuriplo.dll                 (Windows)
Assets/Plugins/x86_64/libneuriplo.so               (Linux)
Assets/Plugins/x86_64/<backend runtime DLLs>       e.g. onnxruntime.dll, opencv_world*.dll
Assets/StreamingAssets/neuriplo/plugins/           libneuriplo_backend_*.dll/.so (plugin_dir)
Assets/StreamingAssets/neuriplo/models/            model files
```

The backend runtime libraries that `neuriplo` or a plugin depends on must sit
somewhere the OS loader can find them. The folder that holds `neuriplo.dll` in
a built player is the simplest place. Pass
`Path.Combine(Application.streamingAssetsPath, "neuriplo/plugins")` as
`plugin_dir`.

**Bindings.**

```csharp
using System;
using System.Runtime.InteropServices;

static class Native {
    const string Lib = "neuriplo";

    [StructLayout(LayoutKind.Sequential)]
    public struct EngineConfig {
        public uint struct_size; public int use_gpu;
        public IntPtr backend_id; public IntPtr model_path;      // UTF-8, see Utf8()
        public UIntPtr batch_size; public IntPtr input_sizes; public UIntPtr n_input_sizes;
        public IntPtr plugin_dir;
    }
    [StructLayout(LayoutKind.Sequential)] public struct InputView { public IntPtr data; public UIntPtr size_bytes; }
    [StructLayout(LayoutKind.Sequential)]
    public struct TensorView {
        public uint struct_size; public int dtype; public IntPtr data; public UIntPtr size_bytes;
        public UIntPtr element_count; public IntPtr shape; public UIntPtr ndim;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void LogCallback(int level, IntPtr message, IntPtr userData);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern uint neuriplo_api_version();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr neuriplo_last_error();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int neuriplo_engine_create(ref EngineConfig c, out IntPtr engine);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern void neuriplo_engine_destroy(IntPtr engine);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int neuriplo_infer(IntPtr engine, InputView[] inputs, UIntPtr n, out IntPtr result);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int neuriplo_result_output(IntPtr result, UIntPtr index, out IntPtr view);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern void neuriplo_result_release(IntPtr result);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] public static extern int neuriplo_set_log_callback(LogCallback cb, int minLevel, IntPtr userData);

    public static string LastError() => Marshal.PtrToStringUTF8(neuriplo_last_error());
    public static IntPtr Utf8(string s) {           // free with Marshal.FreeHGlobal after create
        var bytes = System.Text.Encoding.UTF8.GetBytes(s + "\0");
        var p = Marshal.AllocHGlobal(bytes.Length); Marshal.Copy(bytes, 0, p, bytes.Length); return p;
    }
}
```

**Rules that matter in Unity.**

- **Keep the log delegate alive.** Store it in a static field. If the GC
  collects the delegate while it is installed, the next log message calls
  freed memory. Before you drop it, for example in `OnApplicationQuit` or on a
  domain reload, remove it with `neuriplo_set_log_callback(null, 0, IntPtr.Zero)`.
  Once that call returns, the delegate may be collected. The callback runs on
  neuriplo's threads, so do not touch Unity objects from it. Queue the message
  and handle it on the main thread.
- **Run inference off the main thread.** Use `Task.Run` or a worker thread,
  pin the input with `GCHandle.Alloc(array, GCHandleType.Pinned)` or `fixed`
  for the duration of the call, and marshal the results back to the main
  thread. The library copies the inputs, so you can unpin the buffer when
  `neuriplo_infer` returns. Use one engine per worker thread for parallelism.
  Calls on one engine are serialised.
- **Copy outputs out before releasing.** Use `Marshal.Copy(view.data, managedArray, 0, count)`,
  then call `neuriplo_result_release`.
- **Wrap handles.** Put each handle in a `SafeHandle` or an `IDisposable` so
  that `neuriplo_engine_destroy` and `neuriplo_result_release` run exactly
  once.
- Check `neuriplo_api_version() == 1` at start-up.

The C#/Unity path is checked manually, not in CI. The Unity and OS versions
it was last run on are recorded in the Phase 7 validation log
(`specs/2026-09-25-consumer-c-abi/validation.md`, [M-3]).

## How it is tested

- `CApiContractTest` (plain C, 33 cases) and `CApiWrapperTest` (C++, 9 cases)
  run against the dependency-free fixture plugins in every build with tests
  enabled: `ctest -R "CApi"`. Build-time targets compile the header as strict
  C99 and C++17, pin layouts, and audit the includes.
- The CI job `capi-consumer` runs the symbol check and its two negative
  controls. It then runs `test/consumer/run.sh`, which installs to a temporary
  prefix and builds and runs the C and C++ consumers (through the CMake
  package and through pkg-config) and the Python `ctypes` smoke test.
- The CI job `capi-tsan` runs the thread and log cases under ThreadSanitizer,
  with no suppressions.
- The existing `sanitizers` CI job (ASan, LSan, and UBSan) runs the full
  `ctest`, including every `CApi*` case, on five backends.
