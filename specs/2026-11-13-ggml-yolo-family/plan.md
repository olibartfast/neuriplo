# GGML YOLO family — plan

## Group 0 — Ground truth

- [T-1] Export `yolov8n.onnx` (ultralytics, opset 12, simplify, imgsz 640,
  static) via `/tmp/yoloenv`; dump node types + output spec; decide DFL
  placement [D-3]; record initializer count and head structure.

## Group 1 — Executor extensions

- [T-2] Elementwise/shape ops in the GGML path: SiLU, Concat (multi-input),
  Split (views), Resize-nearest (`ggml_upscale`), Softmax (`ggml_soft_max`
  orientation verified), Reshape/Transpose/Permute, Slice (views),
  Sigmoid iff the head graph contains it. Each helper unit-probed against
  numpy semantics before graph use.

## Group 2 — Topology

- [T-3] YOLOv8n builder: stem convs, C2f blocks (split → bottleneck chain →
  concat), SPPF (maxpools + concat), upsample+concat neck, decoupled head
  (box/cls branches per scale), final concat/permute to `[1,84,8400]`.
  Dispatch resnet vs yolo by tensor-name family; fail loud otherwise.

## Group 3 — Proof

- [T-4] Converter run on `yolov8n.onnx` (extend FC gate to detection heads:
  no single-FC requirement; keep all-float-initializer carriage + rank skip).
- [T-5] ORT reference (same fixed inputs) + parity golden (raw-tensor
  agreement, tolerance adopted from measurement) + resnet no-regression.
- [T-6] Full gates (ctest, format, docs check) + reviewer + brev re-gate
  (real yolov8n conversion + parity) + teardown + merge.
