# GGML multi-input graph executor — plan

- [T-1] Add `GGMLGraph` input/const/op/output API and ggml backend lifecycle.
- [T-2] Implement the op dispatch on top of `GGMLOps`.
- [T-3] Implement host-side I64→I32 materialization for Gather indices.
- [T-4] Add multi-input, multi-output, INT64, transformer-block and
  contract-rejection tests.
- [T-5] Wire into CMake, run the GGML suite and full ctest, format check.
- [T-6] Next packet: ONNX-subset frontend that emits this graph; then add
  `GridSample`/`TopK`/`GatherElements`/dynamic-shape support before an
  RT-DETR raw-parity gate.
