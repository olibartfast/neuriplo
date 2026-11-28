# GGML ResNet-50 bottleneck support — requirements

**Date:** 2026-11-13
**Branch:** `feature/ggml-resnet50-bottleneck` from `develop`
**Status:** Specified, not implemented
**Follows:** `2026-11-13-ggml-real-loader` (merged as PR #74).

## Goal

Run ResNet-50 (`resnet50-v1-12.onnx`, ONNX Zoo) through the GGML backend with
ORT parity, matching E6's `1x3x224x224 → 1000` contract. Same executor op set
as ResNet-18; only the topology grows (bottleneck blocks, 2048-d FC).

## In Scope

- Bottleneck block builder in `GGMLInfer.cpp` (1x1 reduce → BN → relu →
  3x3 → BN → relu → 1x1 expand → BN → skip add → relu; 1x1 downsample on
  stage-entry skips), stages [3,4,6,3], 2048-d FC.
- Model dispatch at load: ResNet-18 vs ResNet-50 by FC input dim (512 vs 2048)
  + bottleneck tensor presence; unknown layout fails loud with shapes seen.
- Converter unchanged (carries all float initializers already); parity golden
  extended with the resnet50 reference.

## Out of Scope

- Other models; attention executor; GGUF; GPU; ABI changes; pin moves.

## Decisions

- [D-1] Exact initializer names per model (zoo naming is stable per file);
  structural fallback (dims-only) only where names genuinely vary.
- [D-2] BN fold + layout rules identical to ResNet-18 (eps 1e-5 verified in
  the resnet50 file before coding; record actual value).
- [D-3] Parity bar unchanged: top-5 exact + max-abs-diff ≤ 1e-3, three inputs
  (gray/zeros/gradient), reference from stock onnxruntime.

## Requirements

- [R-1] `resnet50*.nggl` loads: all stage tensors found, BN folded, dims validated.
- [R-2] Full ResNet-50 graph executes: stem + 16 bottleneck blocks + GAP + FC.
- [R-3] Parity vs ORT on gray/zeros/gradient: top-5 exact, diff ≤ 1e-3.
- [R-4] ResNet-18 path unchanged (existing golden still green).
- [R-5] Unknown/unsupported layout fails with path + shapes seen.
