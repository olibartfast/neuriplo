# ONNX Runtime Execution Providers

The `ONNX_RUNTIME` backend can select ONNX Runtime Execution Providers (EPs) with
`NEURIPLO_ORT_EP`.

By default, neuriplo preserves the existing behavior:

- `use_gpu=false` uses the default CPU EP.
- `use_gpu=true` tries CUDA, then the legacy ROCm path for compatibility, then CPU.

For explicit selection, set a comma-separated priority list:

```bash
NEURIPLO_ORT_EP=qnn,cpu ./your_app
NEURIPLO_ORT_EP=cuda,cpu ./your_app
NEURIPLO_ORT_EP=cpu ./your_app
```

If the list does not include `cpu`, neuriplo disables ORT CPU EP fallback with
`session.disable_cpu_ep_fallback=1`. Include `cpu` when fallback is intentional.

## Provider Aliases

These aliases are recognized:

| Alias | ORT provider |
|---|---|
| `cpu` | `CPUExecutionProvider` |
| `cuda` | `CUDAExecutionProvider` |
| `tensorrt`, `trt` | `TensorrtExecutionProvider` |
| `openvino` | `OpenVINOExecutionProvider` |
| `directml`, `dml` | `DmlExecutionProvider` |
| `migraphx` | `MIGraphXExecutionProvider` |
| `qnn` | `QNNExecutionProvider` |
| `nnapi` | `NnapiExecutionProvider` |
| `coreml` | `CoreMLExecutionProvider` |
| `xnnpack` | `XnnpackExecutionProvider` |
| `acl` | `ACLExecutionProvider` |
| `armnn` | `ArmNNExecutionProvider` |
| `rknpu` | `RknpuExecutionProvider` |
| `vitisai` | `VitisAIExecutionProvider` |
| `cann` | `CANNExecutionProvider` |
| `azure` | `AzureExecutionProvider` |
| `tvm` | `TvmExecutionProvider` |

ROCm is not an explicit alias because ONNX Runtime marks it deprecated. The old automatic
`use_gpu=true` ROCm fallback remains for compatibility.

## Build Gates

CUDA is available through the existing ORT path. Other explicit EPs require both a provider-enabled
ONNX Runtime package and the matching neuriplo CMake option:

```bash
cmake -S . -B build \
  -DDEFAULT_BACKEND=ONNX_RUNTIME \
  -DORT_ENABLE_QNN_EP=ON
```

Current CMake gates:

- `ORT_ENABLE_TENSORRT_EP`
- `ORT_ENABLE_OPENVINO_EP`
- `ORT_ENABLE_MIGRAPHX_EP`
- `ORT_ENABLE_QNN_EP`
- `ORT_ENABLE_XNNPACK_EP`
- `ORT_ENABLE_CANN_EP`
- `ORT_ENABLE_VITISAI_EP`

Additional aliases are documented so config files can use stable names, but they still need
provider-specific linking and tests before being enabled in neuriplo.

## QNN Notes

For QNN, the default backend library is `libQnnHtp.so`. Override it with:

```bash
NEURIPLO_ORT_QNN_BACKEND_PATH=/path/to/libQnnHtp.so
```

QNN HTP/NPU execution generally requires quantized ONNX models. FP32 models may initialize but
fall back or fail depending on the graph and fallback policy.
