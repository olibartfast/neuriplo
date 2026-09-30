# Native Engine

The first-party ONNX inference engine, exposed as the `NATIVE` backend.

It is an interpreter: each model load parses the ONNX graph, infers tensor
shapes for the requested input, plans a single arena allocation, then executes
the nodes one by one in topological order. It is not a plan-building compiler;
there is no graph lowering, fusion, or code generation.

The device is chosen once per loaded graph. A graph that cannot run entirely
on the requested device is rejected when it is loaded; there is no per-operator
fallback to another device at run time.

The CPU kernels are the executable correctness oracle for the future
accelerated path. They are deliberately unoptimized and are not a performance
target — they exist so later device layers can be checked for numerical
agreement rather than re-derive operator semantics.

Nothing under `engine/` includes or links the backend abstraction
(`include/neuriplo/`, `backends/src/`, `src/`). A build guard enforces this
boundary; the adapter that exposes the engine through `InferenceInterface`
lives in `backends/native/` instead.

## Layout

- `engine/include/engine/` — public headers: graph IR, model loader, shape
  inference, memory plan, device seam, and executor.
- `engine/src/` — implementation of the loader, shape inference, memory
  planner, device, and executor.
- `engine/src/kernels/` — the CPU operator kernels behind the device
  kernel table.
- `engine/test/` — unit tests for the loader, shapes, plan, device, kernels,
  and executor.

## Build and test

Configure with the native backend selected and engine tests enabled:

```bash
cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON \
  && cmake --build build-native \
  && ctest --test-dir build-native -R engine
```

Elementwise parity against ONNX Runtime uses a separate tree with both
backends enabled, which provisions a deterministic ResNet-18 fixture and
compares the two outputs:

```bash
cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON \
  && cmake --build build-parity \
  && ctest --test-dir build-parity -R parity
```

The engine links only the C++ standard library.
