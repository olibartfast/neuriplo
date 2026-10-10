# Backend Architecture and Design Patterns

Neuriplo exposes runtime implementations and GPU preprocessing through one
`InferenceInterface` contract. The public setup facade returns
`std::unique_ptr<InferenceInterface>`, so consumers can select an implementation
without depending on a vendor SDK type.

## Component relationships

```text
setup_inference_engine / BackendRuntimeRegistry
                    |
                    v
        IBackendRuntimeFactory
                    |
                    v
            InferenceInterface
             /      |       \
            /       |        \
     *Infer adapter |   BackendDecorator
                    |
               ModelRunner
```

The architecture uses five patterns:

- **Adapter:** each `*Infer` class translates a vendor runtime or pipeline API
  into `InferenceInterface`.
- **Bridge:** `ModelRunner` owns an `InferenceInterface` and delegates lifecycle
  and inference without knowing the concrete implementation.
- **Abstract Factory:** each `IBackendRuntimeFactory` creates the backend,
  allocator, and tensor converter for one compiled runtime family.
- **Decorator:** profiling, logging, caching, and quantization wrappers add
  opt-in behavior while retaining the same interface.
- **State:** `BackendState` records the
  `Uninitialized -> Loading -> Ready/Failed` lifecycle.

The compile-time default remains `DEFAULT_BACKEND`. Multi-backend and plugin
builds add runtime lookup without changing the consumer interface. Plugin
ownership and ABI rules are documented in [Plugin Backends](PLUGIN_BACKENDS.md).

## Construction and failure behavior

`setup_inference_engine(model_path, use_gpu, batch_size, input_sizes)` remains
the compatibility facade. It selects the compiled factory, creates the backend,
and optionally applies decorators. Model construction failures use
`ModelLoadException` and are translated according to the facade contract;
request-time failures use `InferenceExecutionException`. Backends must not
terminate the host process.

The facade translates a `ModelLoadException` into a `nullptr` return, so a
consumer checks one value instead of catching backend-specific errors. Each
backend's `*RuntimeFactory` produces a coherent `{backend, allocator,
converter}` family, and `setup_inference_engine` builds through the factory
selected at compile time by `-DDEFAULT_BACKEND`. A factory may pick between
runtimes of one family; `LibtorchRuntimeFactory`, for example, routes `.pt2`
packages to AOTInductor and everything else to TorchScript.

### Opt-in decorators

`ProfilingBackend`, `LoggingBackend`, `CachingBackend`, and `QuantizedBackend`
wrap any backend without changing its interface. They are off by default, so
the production path is unchanged. The profiling and logging chain is enabled at
runtime:

```bash
NEURIPLO_ENABLE_PROFILING=1 NEURIPLO_ENABLE_LOGGING=1 ./your_app
```

Device placement and fallback remain backend-owned until the shared
`EngineOptions` contract is extended. A requested accelerator must be used,
fail clearly, or fall back only when the caller explicitly permits it.

## DALI as a pipeline adapter

DALI is the fourteenth `InferenceInterface` adapter. It wraps a GPU data
pipeline rather than a model inference engine. This lets factories, decorators,
`ModelRunner`, and serving integrations compose DALI through the same contract
as model backends. A serving graph can place DALI before TensorRT or another
model adapter for decode, resize, and normalization without adding a second
orchestration abstraction.

`DALIInfer` accepts a serialized pipeline as `model_path`, maps declared
external sources to input tensors, and returns outputs through the shared typed
metadata and raw-buffer interfaces. Output names are positional unless the
deployment supplies `|outnames=...`.

Dependency layout, CMake configuration, artifact generation, metadata
constraints, and the GPU acceptance command live in
[Dependency Management](DEPENDENCY_MANAGEMENT.md#dali-pipeline-metadata-and-validation).

## Extension points

A new backend supplies an `InferenceInterface` adapter and an
`IBackendRuntimeFactory`, then registers its build and runtime metadata.
[Adding an Inference Backend](ADDING_BACKEND.md) defines the complete contract.

Cross-cutting behavior belongs in decorators when it can preserve the wrapped
backend's output and failure semantics. Runtime-specific device providers,
delegates, and preprocessing operators stay inside their owning adapter rather
than becoming duplicate top-level backend IDs.

_Revision: 2026-10-10 - absorbed the architecture summary from the README
(facade `nullptr` contract, decorator switches, per-format factory routing)._
