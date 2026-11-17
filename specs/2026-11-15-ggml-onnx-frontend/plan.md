# GGML ONNX-subset frontend — plan

- [x] [T-1] Write the minimal ONNX wire reader (tensors, value infos, nodes,
  attributes, graph, model).
- [x] [T-2] Add static shape tracking and ONNX→ggml axis reversal.
- [x] [T-3] Emit graph inputs, constants, and the supported op set into
  `GGMLGraph`.
- [x] [T-4] Add `Sub`/`Div`/`Sqrt`/`Clip` to the graph executor.
- [x] [T-5] Add regular Conv, depthwise Conv and MaxPool to the executor and
  frontend.
- [x] [T-6] Add tiny fixtures and C++ parity tests for multi-input Add,
  INT64 Gather, attention, shape/activation ops, Conv/MaxPool, depthwise
  Conv, ConstantOfShape, unsupported op, and dynamic shape.
- [x] [T-7] Constant-fold INT64 shape subgraphs so `Shape`/`Concat`/`Slice`/
  `Gather` chains feed `Reshape`, `Expand` and `Tile`. Covered by
  `shape_fold.onnx` and `shape_ops.onnx`.
- [x] [T-8] Add `Slice` (step 1 and axis-0 reverse), `ReduceMean`/`Sum`/`Max`,
  `Pow`, `Abs`, `Log`, `Pad`, `Split`, `Expand`, `Tile`, `Resize` and `Erf`.
- [x] [T-9] Add `TopK`, `GatherElements` and `Resize`. `GridSample` remains.
- [x] [T-10] Run fixed-input raw parity for a static-batch `rtdetrv4`.
  300/300 detections match ORT by label and box; max box L1 2.7e-3,
  max score diff 2.2e-5. Near-tie index order differs.
- [x] [T-11] Add explicit leading-axis offset metadata so Reshape/Unsqueeze/
  Slice/Split/Transpose keep ONNX axis numbering after dropping singleton
  axes.
- [x] [T-12] Add `GridSample` bilinear for deformable attention.
- [ ] [T-13] Sort TopK outputs to satisfy ONNX `sorted=1`, and lift the
  axis-1 restriction in GatherElements.
- [ ] [T-14] Wire the ONNX frontend into `GGMLInfer`/the app and run the
  `rtdetr` app gate (`scripts/ggml_app_gate.py`).
