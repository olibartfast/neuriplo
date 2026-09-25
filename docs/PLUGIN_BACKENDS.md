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

## Packaging layout

A deployment is the host application plus one directory of plugins. The
loader scans that directory (non-recursively) for regular files named
`libneuriplo_backend_*` with the platform's module extension (`.so` on POSIX,
`.dll` on Windows); everything else is ignored.

```text
/opt/myapp/
├── bin/myapp                          # links libneuriplo (compiled-in default backend)
└── plugins/                           # EngineOptions::plugin_dir or NEURIPLO_PLUGIN_DIR
    ├── libneuriplo_backend_onnx_runtime.so
    ├── libneuriplo_backend_tensorrt.so
    └── lib/                           # optional: framework libraries shipped with the plugins
        ├── libonnxruntime.so.1
        └── ...
```

One backend id per plugin. If two plugins advertise the same `backend_id`, the
first one loaded wins and the second is skipped with a warning naming both
paths. A compiled-in backend with the same id beats any plugin.

## Dependency discovery

A plugin's framework libraries (ONNX Runtime, TensorRT, ...) are resolved by
the platform loader when the host opens the plugin. Nothing in neuriplo
searches for them.

- **POSIX: `dlopen(RTLD_NOW | RTLD_LOCAL)`.** `RTLD_NOW` resolves every
  symbol at load, so a missing framework library rejects the plugin at
  discovery ("skipping plugin <path>: <dlerror>") instead of failing
  mid-inference. `RTLD_LOCAL` keeps the plugin's symbols out of the global
  namespace, so two plugins cannot resolve each other's symbols. It does
  **not** control *where* dependencies are found. That follows the normal
  rules: the plugin's own `DT_RUNPATH`, then `LD_LIBRARY_PATH`, the loader
  cache, and the default directories. To ship framework libraries beside a
  plugin, give the plugin a `$ORIGIN`-relative runpath (for the layout above,
  `patchelf --set-rpath '$ORIGIN/lib' libneuriplo_backend_onnx_runtime.so`).
  A plugin built in the tree carries CMake's build-tree runpath, which is
  wrong once the plugin is copied elsewhere. Caveat: the loader
  de-duplicates by SONAME. If the host or another plugin already loaded a
  library with the same SONAME, a later plugin gets that copy, whatever its
  own runpath says. `RTLD_LOCAL` isolates symbols, not SONAMEs.
- **Windows: `LoadLibraryExA(..., LOAD_WITH_ALTERED_SEARCH_PATH)`.** The
  loader passes an absolute path, so the plugin's own directory is searched
  first for its dependent DLLs. Ship framework DLLs *next to* the plugin DLL
  (not in a `lib/` subdirectory), or on `PATH`.

## Compatibility and versioning

- The ABI version is `NEURIPLO_PLUGIN_ABI_VERSION` in
  `include/neuriplo/plugin_abi.h` (currently **2**). The host loads a plugin
  only if its `abi_version` matches exactly. Otherwise it logs
  "ABI version <n> != host <m>" and skips it.
- The version is bumped only for a breaking change to the ABI structs or to
  their meaning. Host-side validation (below) is not a breaking change: a
  conforming v2 plugin built before it keeps loading and behaving the same.
- `struct_size` on `neuriplo_engine_options_t` and `neuriplo_host_services_t`
  lets those structs grow at the end without a version bump. A plugin must
  not read fields beyond the `struct_size` it was given.
- Plugins are never unloaded. The api table and every backend handle a plugin
  returns must stay valid for the process lifetime.

## What the host validates

The host treats everything a plugin returns as untrusted input. A plugin that
breaks a rule below is rejected with a diagnostic. The host never aborts,
asserts, or dereferences the bad value.

**At load** (plugin skipped, logged as `skipping plugin <path>: <reason>`,
never added to `available_backend_ids()`):

| Condition | Reason logged |
| --- | --- |
| library cannot be opened | the `dlerror()` / `FormatMessage` text |
| no `neuriplo_plugin_get_api_v1` symbol | `missing neuriplo_plugin_get_api_v1` |
| entry point returns NULL | `entry point returned null` |
| `abi_version` differs from the host | `ABI version <n> != host <m>` |
| `backend_id` or any function pointer is NULL | `incomplete api table` |
| `backend_id` already loaded | `backend id '<id>' already provided by <path>` |

**At backend creation** (`setup_inference_engine` / `create_plugin_backend`
returns `nullptr` and logs `plugin backend '<id>': ...`). The plugin
instance is destroyed. The message names the plugin path, the layer
(`input layer <i>` / `output layer <i>`) and the field:

- `create` returns NULL. The plugin's own error text is logged.
- `get_metadata` returns non-zero.
- A layer array (`inputs` / `outputs`) is NULL while its count is non-zero.
- A layer's `name` is NULL, its `shape` is NULL with `ndim > 0`, its `ndim`
  exceeds the host's rank bound, or its `element_type` is not a
  `neuriplo_dtype_t` value.

Metadata dimensions may be negative (dynamic), as in ONNX models.

**At inference** (`get_infer_results_raw` / `get_infer_results` throw
`InferenceExecutionException`). The message names the backend id, the
tensor (`output <i>`) and the field:

- `infer` returns non-zero. The plugin's error text is included.
- `infer` returns 0 but `tensors` is NULL with a non-zero count, or a
  tensor's `data` is NULL, its `shape` is NULL with `ndim > 0`, its `ndim`
  exceeds the rank bound, a dimension is negative, or its `dtype` is unknown.
- `size_bytes` differs from `element size × product of shape`. The check is
  strict equality: a buffer that is too short or too long is rejected, not
  truncated.

An unknown dtype rejects the whole call. The host never drops a tensor or
substitutes a type. **Ownership:** for every `infer` that returns 0, the host
calls `release_outputs` exactly once, including when it then rejects the
outputs or fails while copying them. A plugin must not free its output arrays
anywhere else.

## Deployment example

Build the ONNX Runtime backend as a plugin next to an OpenCV DNN host, then
run it from a relocated directory:

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DNEURIPLO_PLUGIN_BACKENDS=ONNX_RUNTIME
cmake --build build

mkdir -p /opt/myapp/plugins/lib
cp build/plugins/libneuriplo_backend_onnx_runtime.so /opt/myapp/plugins/
cp "$ONNXRUNTIME_DIR"/lib/libonnxruntime.so* /opt/myapp/plugins/lib/
patchelf --set-rpath '$ORIGIN/lib' /opt/myapp/plugins/libneuriplo_backend_onnx_runtime.so

# Check that every dependency resolves from the new location:
ldd /opt/myapp/plugins/libneuriplo_backend_onnx_runtime.so | grep -E 'onnxruntime|not found'
```

```cpp
EngineOptions options;
options.model_path = "model.onnx";
options.backend_id = "ONNX_RUNTIME";
options.plugin_dir = "/opt/myapp/plugins";
std::unique_ptr<InferenceInterface> engine = setup_inference_engine(options);
if (!engine) {
    // The log says why: plugin skipped at load, create/metadata rejected, ...
    for (const std::string& id : available_backend_ids(options.plugin_dir)) {
        std::cerr << "available: " << id << "\n";
    }
}
```

`NEURIPLO_PLUGIN_DIR=/opt/myapp/plugins` works in place of `plugin_dir`.

## Tests

- `backends/src/test/PluginAbiContractTest.cpp` is the ABI contract suite.
  It is built in every configuration, needs no vendor SDK, and runs against
  first-party fixture plugins (`backends/src/test/plugin_fixtures/`). It
  covers each load-time rejection, malformed metadata and outputs,
  release-exactly-once ownership, concurrent loading, and isolation of a good
  plugin from broken ones. Run it with
  `ctest --test-dir build -R PluginAbi --output-on-failure`, and under
  `./scripts/quality/sanitizers.sh` for ASan/UBSan.
- `backends/src/test/PluginLoaderTest.cpp` (built when
  `NEURIPLO_PLUGIN_BACKENDS` is set) proves that a host without built-in ONNX
  Runtime serves an identity model through the ONNX Runtime plugin.
