# Adding an Inference Backend

This guide describes the current standard path for adding a backend to neuriplo.
The workflow is intentionally explicit: backend implementations are selected at
CMake configure time through `DEFAULT_BACKEND`, and only the selected backend is
compiled into the `neuriplo` library.

## Backend Contract

Every backend should:

- Derive from `InferenceInterface` (the Adapter role: wrap a vendor SDK behind
  the common contract).
- Implement `get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors)`.
- Populate `inference_metadata_` with input and output shapes.
- Preserve the constructor shape used by `setup_inference_engine`:
  `Backend(model_path, use_gpu, batch_size, input_sizes)`.
- Drive the lifecycle State: set `state_ = BackendState::Ready` after a
  successful load, and `state_ = BackendState::Failed` on any load/inference
  error. **Never call `std::exit`** — throw `ModelLoadException` for load
  failures and `InferenceExecutionException` for runtime failures. The
  `setup_inference_engine` facade catches `InferenceException` and returns
  `nullptr`, which is the failure contract downstream consumers handle.
- Validate unsupported input counts, model formats, device modes, and batch sizes
  with clear exceptions instead of silent fallback.
- Track inference timing through the shared helper methods when practical.

## Required Files

For a backend named `NCNN`, add:

- `backends/ncnn/src/NCNNInfer.hpp`
- `backends/ncnn/src/NCNNInfer.cpp`
- `backends/ncnn/src/NCNNRuntimeFactory.hpp` (Abstract Factory for this backend)
- `backends/ncnn/test/CMakeLists.txt`
- `backends/ncnn/test/NCNNInferTest.cpp`
- `cmake/NCNN.cmake`

Add setup or model-generation helpers when the backend needs them:

- `scripts/setup_ncnn.sh`
- `backends/ncnn/test/generate_model.sh`
- `docker/Dockerfile.ncnn`
- `docker/run_ncnn_tests.sh`

## CMake Registration

Register the backend once in `cmake/BackendRegistry.cmake`:

```cmake
list(APPEND NEURIPLO_BACKEND_IDS NCNN)

set(NEURIPLO_BACKEND_NCNN_MODULE NCNN)
set(NEURIPLO_BACKEND_NCNN_TEST_DIR backends/ncnn/test)
set(NEURIPLO_BACKEND_NCNN_VERSION_VAR NCNN_VERSION)
```

Then add the dependency version:

```bash
NCNN_VERSION=1.0.34
```

in `versions.env`, and expose it in `cmake/versions.cmake`:

```cmake
set(NCNN_VERSION "${NCNN_VERSION}" CACHE STRING "NCNN version")
```

`validate_backend_versions()` will pick up the backend-to-version mapping from
`cmake/BackendRegistry.cmake`.

## Backend CMake Module

The backend module should append implementation sources and define one compile
flag for the selected backend:

```cmake
set(NCNN_SOURCES
    ${INFER_ROOT}/ncnn/src/NCNNInfer.cpp
)

list(APPEND SOURCES ${NCNN_SOURCES})
add_compile_definitions(USE_NCNN)
```

Add backend-specific `find_package`, status messages, and path handling here
when needed.

## Link and Include Rules

Add the backend link block to `cmake/LinkBackend.cmake`:

```cmake
elseif(DEFAULT_BACKEND STREQUAL "NCNN")
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${NCNN_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${INFER_ROOT}/ncnn/src)
    target_link_directories(${PROJECT_NAME} PRIVATE ${NCNN_DIR}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)
```

This is still explicit because backend dependency layouts differ substantially.

## Dependency Validation and Setup

Add a validation function in `cmake/DependencyValidation.cmake` and call it from
`validate_all_dependencies()`:

```cmake
function(validate_ncnn)
    if(DEFAULT_BACKEND STREQUAL "NCNN")
        validate_dependency("NCNN" "${NCNN_DIR}")
        # Check required headers and libraries here.
    endif()
endfunction()
```

Add setup support to `scripts/setup_dependencies.sh`:

- Help text.
- Backend allowlist.
- `setup_ncnn` function or call to `scripts/setup_ncnn.sh`.
- `validate_installation` case.
- Environment exports in `create_env_setup`.

Add matrix support to `scripts/test_backends.sh`:

- `BACKENDS`
- `BACKEND_DIRS`
- `BACKEND_TEST_EXES`
- `check_backend_availability`

## Abstract Factory

Each backend ships a concrete `IBackendRuntimeFactory` that owns construction of
its adapter together with the matching allocator and tensor converter. Create
`backends/ncnn/src/NCNNRuntimeFactory.hpp`:

```cpp
#pragma once
#include "HostTensorConverter.hpp"
#include "IAllocator.hpp"
#include "IBackendRuntimeFactory.hpp"
#include "ITensorConverter.hpp"
#include "NCNNInfer.hpp"

class NCNNRuntimeFactory : public IBackendRuntimeFactory {
  public:
    std::unique_ptr<InferenceInterface> create_backend(const std::string& model_path, bool use_gpu, size_t batch_size,
                                                       const std::vector<std::vector<int64_t>>& input_sizes) override {
        return std::make_unique<NCNNInfer>(model_path, use_gpu, batch_size, input_sizes);
    }
    std::unique_ptr<IAllocator> create_allocator() override { return std::make_unique<HostAllocator>(); }
    std::unique_ptr<ITensorConverter> create_converter() override { return std::make_unique<HostTensorConverter>(); }
    const char* name() const noexcept override { return "NCNNRuntimeFactory"; }
};
```

## Factory Registration

`setup_inference_engine` selects the factory by `#ifdef`, so register the backend
in two places. Add the include in `include/InferenceBackendSetup.hpp`:

```cpp
#elif USE_NCNN
#include "NCNNRuntimeFactory.hpp"
```

and the factory dispatch in `make_runtime_factory()` in
`src/InferenceBackendSetup.cpp`:

```cpp
#elif USE_NCNN
    return std::make_unique<NCNNRuntimeFactory>();
```

The shared facade then handles eager `load()`, the `Failed`-state /
`ModelLoadException` -> `nullptr` translation, and the opt-in decorator chain —
no per-backend wiring is required for any of that.

## Verification

At minimum, run:

```bash
cmake -S . -B build -DDEFAULT_BACKEND=NCNN -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

When dependency setup is automated, also run:

```bash
./scripts/setup_dependencies.sh --backend NCNN
./scripts/test_backends.sh --backend NCNN
```

Update `Readme.md`, `docs/DEPENDENCY_MANAGEMENT.md`, and `CHANGELOG.md` when
the backend changes supported model formats, installation workflow, or public
behavior.
