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
- [ ] [T-7] Constant-fold INT64 shape subgraphs so `Shape`/`Concat`/`Slice`/
  `Gather` chains feed `Reshape`, `Expand` and `Tile`.
- [ ] [T-8] Add `Slice` (step 1 and axis-0 reverse), then `ReduceMean`,
  `Pow`, `Abs`, `Log`, `Pad`, `Split`, `Expand`, `Tile`.
- [ ] [T-9] Add `TopK`, `GatherElements`, `GridSample`, `Resize`, `Erf`.
- [ ] [T-10] Run fixed-input raw parity for a static-batch `rtdetrv4`; only
  then run the `neuriplo-infer` `rtdetr` app gate.
