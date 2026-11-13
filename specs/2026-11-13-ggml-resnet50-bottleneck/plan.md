# GGML ResNet-50 — plan

## Group 0 — Ground truth

- [T-1] Dump `resnet50-v1-7`/`v1-12` initializer names/shapes (download from
  ONNX Zoo if absent); verify BN epsilon uniform 1e-5; confirm FC dims.
  Obtain `resnet50.onnx` locally for conversion + reference.

## Group 1 — Loader

- [T-2] Bottleneck builder + stage loop [3,4,6,3] with downsample on entries;
  dispatch resnet18/resnet50 by FC input dim; `fail_model` otherwise.
- [T-3] Reuse fold/permute/pool/FC helpers; 2048-d GAP→FC verbatim orientation.

## Group 2 — Proof

- [T-4] Converter run on resnet50 (FC gate extended to `[1000,2048]`; rank-skip
  + strict flag validation already in) + ORT reference
  (`ggml_ort_reference.py --model` parameterized if needed) + parity golden
  for both models (resnet18 must stay green).
- [T-5] Full gates (77+ ctest, format, docs check) + reviewer + brev re-gate
  on the resnet50 artifact + teardown + merge.
