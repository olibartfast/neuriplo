# Backend Plugins (dlopen)

Backends can be built as standalone shared libraries and loaded at runtime,
so one process can serve models on several frameworks (e.g. ONNX Runtime and
TensorRT) without compiling them all into `libneuriplo.so`.

## Building

```bash
cmake -S . -B build \
  -DDEFAULT_BACKEND=OPENCV_DNN \
  -DNEURIPLO_PLUGIN_BACKENDS="ONNX_RUNTIME;TENSORRT"
cmake --build build
# → build/plugins/libneuriplo_backend_onnx_runtime.so
# → build/plugins/libneuriplo_backend_tensorrt.so
```

- `NEURIPLO_BACKENDS` (semicolon list, default `${DEFAULT_BACKEND}`) selects
  the backends compiled **into** `libneuriplo.so`.
- `NEURIPLO_PLUGIN_BACKENDS` selects backends built as **plugins**. A backend
  may appear in both; at runtime the compiled-in registration wins.
- Dependency validation covers both lists. The `LLAMACPP`+`GGML` pair is only
  banned for the compiled-in set; as plugins they are isolated by
  `RTLD_LOCAL`.

## Loading

Plugins are discovered from `EngineOptions::plugin_dir` and/or the
`NEURIPLO_PLUGIN_DIR` environment variable:

```cpp
EngineOptions options;
options.model_path = "model.onnx";
options.backend_id = "ONNX_RUNTIME";          // registry id
options.plugin_dir = "/opt/neuriplo/plugins"; // scanned for libneuriplo_backend_*.so
auto engine = setup_inference_engine(options);
```

`available_backend_ids()` returns compiled-in plus loaded plugin ids.
Plugins are loaded `RTLD_NOW | RTLD_LOCAL` (each plugin's framework
dependencies stay private), version-checked against
`NEURIPLO_PLUGIN_ABI_VERSION`, and never unloaded for the process lifetime.
Broken or incompatible plugins are skipped with a logged reason.

## ABI

The C ABI lives in `include/neuriplo/plugin_abi.h`. A plugin exports exactly
one symbol, `neuriplo_plugin_get_api_v1`, returning a static descriptor
(`abi_version`, `backend_id`, `create`/`destroy`/`get_metadata`/`infer`/
`release_outputs`). Rules:

- No C++ types, STL, or exceptions cross the boundary.
- Tensors cross as typed contiguous buffers (`neuriplo_dtype_t` + bytes).
- Plugin memory is freed by the plugin; the host never frees it.
- Plugins log through the host-provided callback, never their own glog init.
- Breaking struct changes bump `NEURIPLO_PLUGIN_ABI_VERSION`; the host skips
  mismatched plugins.

Existing backends need no per-backend code to become plugins: the build
generates an entry point from `cmake/plugin_entry.cpp.in` that wraps the
backend's `IBackendRuntimeFactory` via `backends/src/plugin/PluginShim.hpp`
(`NEURIPLO_DEFINE_PLUGIN`). The host side
(`backends/src/plugin/PluginLoader.{hpp,cpp}`) adapts plugins back to
`InferenceInterface`, so `ModelRunner`, decorators, and serving integrations
work unchanged.

Tests: `backends/src/test/PluginLoaderTest.cpp` (built when
`NEURIPLO_PLUGIN_BACKENDS` is set) proves a host without built-in ONNX Runtime
serves an identity model through the ONNX Runtime plugin.
