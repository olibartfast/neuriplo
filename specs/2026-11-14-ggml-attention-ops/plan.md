# GGML attention/executor ops — plan

## Group 0 — Semantics

- [T-1] Confirm ggml axis conventions for `ggml_norm` (ne[0]), `ggml_gelu_erf`,
  `ggml_mul_mat` (ne0 contraction), `ggml_get_rows` (row indices), and
  `ggml_top_k` (I32 indices, order unspecified).
- [T-2] Decide the attention layout: q/k/v `[head_dim, seq, batch]`; scores
  `[seq_k, seq_q]`; V transposed to `[seq_k, head_dim]` for the second
  MatMul.

## Group 1 — Op helpers

- [T-3] Add `GGMLOps.{hpp,cpp}` with `layer_norm`, `gelu`, `matmul`,
  `linear`, `scaled_dot_product_attention`, `gather_rows`,
  `top_k_indices`.
- [T-4] Wire `GGMLOps.cpp` into `cmake/GGML.cmake`; expose the header to the
  test target.

## Group 2 — Proof

- [T-5] Add `GGMLOpsParityTest.cpp`: MatMul/Linear, LayerNorm, GELU,
  attention, Gather/TopK, tiny transformer block, and contract-rejection
  tests.
- [T-6] Run the op tests, the existing `GGMLInferTest` suite, and full
  ctest; format check and `git diff --check`.

## Group 3 — Next packet

- [T-7] Specify the RT-DETR path: multi-input/INT64 plumbing + generic
  ONNX-subset importer or purpose-built builder, including `GridSample`,
  `TopK`, `GatherElements`, `Cast`, `ConstantOfShape`, `Expand`, `Tile`.
  Do not claim RT-DETR until raw-output parity exists.
