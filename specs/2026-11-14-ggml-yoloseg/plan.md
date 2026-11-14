# GGML YOLOv8n-seg — plan

## Group 0 — Ground truth

- [x] [T-1] Dump the segmentation ONNX node/tensor structure: confirm
  `output0 [1,116,8400]`, `output1 [1,32,160,160]`, cv4/proto tensor names,
  ConvTranspose weight `[64,64,2,2]`, and that the proto tail is SiLU.
- [x] [T-2] Probe `ggml_conv_transpose_2d_p0` orientation and precision on a
  tiny synthetic kernel/input against numpy before wiring the full branch.

## Group 1 — Multi-output plumbing

- [x] [T-3] Replace the single `output_shape_` with `output_shapes_`
  (`std::vector<std::vector<int64_t>>`); keep `{1,1000}` default.
- [x] [T-4] `setup_input_output_tensors` emits one metadata entry per output
  (`output` for single-output models; `output0`/`output1` for seg).
- [x] [T-5] `load_model` detects seg from exact tensor names, sets both output
  shapes, builds both graph outputs, and stores both tensors in
  `output_tensors_`.
- [x] [T-6] `get_infer_results` returns every output in order with its shape;
  keep the existing single-output per-batch path byte-identical.

## Group 2 — Segmentation head

- [x] [T-7] Add cv4 mask-coefficient branches for P3/P4/P5 (32ch) and concat
  along the flat spatial dim; concat box(64)+class(80)+mask(32) for
  `output0`.
- [x] [T-8] Add proto branch from P3 with F16 ConvTranspose, bias, cv2, cv3
  (SiLU) for `output1`; keep NCHW-flat storage.
- [x] [T-9] Converter: allow the segmentation artifact (it already carries all
  float initializers and requires the DFL weight); no layout change.

## Group 3 — Proof

- [x] [T-10] Generate ORT references for both outputs on gray/zeros/gradient
  with stock onnxruntime 1.32; keep single-output script behavior unchanged.
- [x] [T-11] Add `ParityVsYoloSeg` comparing `output0` (max-cell agreement +
  max-abs-diff) and `output1` (elementwise max-abs-diff); skip cleanly when
  artifacts/refs are absent, hard-fail when present but malformed.
- [x] [T-12] Build, run `GGMLInferTest`, full ctest where practical, format
  check, and record exact commands/results in `validation.md`.
