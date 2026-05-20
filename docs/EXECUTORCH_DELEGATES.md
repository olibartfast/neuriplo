# ExecuTorch Delegates

ExecuTorch's equivalent of an ONNX Runtime Execution Provider is a **backend**
(a.k.a. **delegate**). Unlike ORT EPs, which partition a graph at runtime, an
ExecuTorch delegate is selected **ahead of time**: the partitioner runs during
`.pte` export and bakes the delegated subgraphs into the file. The runtime then
simply consumes whichever backend library was linked in.

That makes delegate support two-sided in neuriplo:

| Side | Where | What it does |
|------|-------|--------------|
| Export | `backends/executorch/test/export_executorch_classifier.py` | Chooses the partitioner, bakes the delegate into the `.pte`. |
| Build  | `cmake/ExecuTorch.cmake` + `cmake/LinkBackend.cmake` | Links the matching backend library so the delegate self-registers. |

**The delegate baked into the `.pte` must match the backend linked into
neuriplo.** A `.pte` exported for XNNPACK fails to load if neuriplo was built
without the XNNPACK backend, and vice versa.

## Supported delegates

| `EXECUTORCH_DELEGATE` | Backend library | Notes |
|-----------------------|-----------------|-------|
| `xnnpack` (default)   | `xnnpack_backend` | Optimized CPU delegate. Ships inside the ExecuTorch source tree — no external SDK. |
| `portable`            | (none)            | No delegation; runs on the portable kernels. |

SDK-backed delegates — **QNN** (Qualcomm Hexagon), **Vulkan**, **Core ML**,
**MPS**, **Ethos-U** — are intentionally **not wired**. Each requires a new
external dependency (the `new-dependency` change class is forbidden in
`REPO_META.yaml`) and, for some, a non-Linux build host. They slot into the same
seam: a new `EXECUTORCH_DELEGATE` value, a `-DEXECUTORCH_BUILD_<X>=ON` flag in
`setup_executorch.sh` / `docker/Dockerfile.executorch`, a backend library linked
in `cmake/LinkBackend.cmake`, and a partitioner branch in the export script.

## Building

The setup script and Docker image build the XNNPACK delegate by default:

```bash
./scripts/setup_executorch.sh            # builds with -DEXECUTORCH_BUILD_XNNPACK=ON
```

Configure neuriplo with the matching delegate:

```bash
cmake -S . -B build \
  -DDEFAULT_BACKEND=EXECUTORCH \
  -DEXECUTORCH_DIR=$HOME/dependencies/executorch \
  -DEXECUTORCH_DELEGATE=xnnpack \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build
```

`EXECUTORCH_DELEGATE` defaults to `xnnpack`; set it to `portable` to link only
the portable kernels.

## Exporting a matching model

```bash
# from backends/executorch/test/
./generate_model.sh xnnpack      # default; also accepts: portable
```

This writes `resnet18_<delegate>.pte` and a `model_path.txt` pointing at it.
Equivalently, run the exporter directly:

```bash
python export_executorch_classifier.py --delegate xnnpack --output-dir .
```

## Runtime behaviour

`ExecuTorchInfer` logs the configured delegate at construction. The delegate is
fixed in the `.pte`, so the constructor's `use_gpu` flag does not select
hardware — XNNPACK is a CPU delegate, so `is_gpu_available()` reports `false`.
Passing `use_gpu=true` logs a warning that hardware-accelerated delegates are not
built into this configuration.
