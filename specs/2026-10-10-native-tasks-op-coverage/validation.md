# Validation — neuriplo-tasks Operator Coverage on the NERT

Written before each tier is implemented. Nothing is checked unless it ran.

```text
# Engine and adapter tests
cmake -S . -B build-parity -DDEFAULT_BACKEND=NERT -DNEURIPLO_BACKENDS=ONNX_RUNTIME \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-parity --parallel 2
NERT_FIXTURE=<resnet18.onnx> ctest --test-dir build-parity -R "engine|NertInfer|parity"
```

```text
# Model-zoo parity: one manifest row per export
#   name  path  dims[;dims]  parity|e2e|load  atol  [input.bin]
NEURIPLO_MODEL_ZOO_MANIFEST=<manifest> \
  build-parity/backends/nert/test/nert_model_zoo_parity_test
```

- [ ] [V-1] → [R-1]: every tier operator has a hand-computed unit case in
  `engine/test/KernelsTest.cpp`, off-default attributes included, and a
  negative case for a rejected shape.
- [ ] [V-2] → [R-2], [R-4]: every gated export loads on `NERT` (`load` rows).
- [ ] [V-3] → [R-3]: every gated export matches `ONNX_RUNTIME` (`parity` or
  `e2e` rows), max scaled difference recorded below.
- [ ] [V-4] → [D-4]: constant folding folds constant-only chains, leaves
  graph-output producers and unknown ops (`engine_constant_folding`).
- [ ] [V-5] → [D-5], [D-6]: DOUBLE constants and value-info load as float32;
  identical duplicate value-info loads, a conflicting one is rejected
  (`engine_loader`).
- [ ] [V-6]: the existing ResNet-18 and YOLO26 parity tests still pass.

## Evidence

Tier 1 (2026-10-10, CPU, Release, ORT 1.19.2). Inputs: seeded uniform [0,1)
for `parity` rows, `bus.jpg` (640×640, /255) for YOLO rows marked *.

| Export | Opset (source) | Mode | Max scaled diff |
| --- | --- | --- | --- |
| YOLOv5x | 12 (`~/model_repository`) | parity* | 7.7e-05 |
| YOLOv6l | 13 (`~/model_repository`) | parity* | 2.4e-05 |
| YOLOv7 | 12 (`~/model_repository`) | parity* | 2.3e-05 |
| YOLOv8x | 12 (`~/model_repository`) | parity* | 5.2e-05 |
| YOLOv9c | 12 (`~/model_repository`) | parity* | 4.2e-05 |
| YOLOv10n | 12 (`~/model_repository`) | e2e*, 6 rows | 6.2e-06 |
| YOLOv12s | 17 (`~/model_repository`) | parity* | 1.7e-05 |
| YOLO11n-seg | 17 (Ultralytics 8.3) | parity | 2.1e-05 |
| YOLO11n-pose | 17 (Ultralytics 8.3) | parity | 1.5e-04 |
| YOLO26n | 17 (Ultralytics 8.4) | e2e*, 4 rows | 9.2e-05 |
| YOLO26n-seg | 17 (Ultralytics 8.4) | parity | 5.4e-05 |
| YOLO26n-pose | 17 (Ultralytics 8.4) | e2e*, 40 rows | 1.9e-04 |

Every export above was converted to opset 18 ([D-2]) and also loads (`load`
rows). Existing tests on the same tree: `engine_*` 12/12, `NertInferTest`
9/9, ResNet-18 parity max abs diff 2.2e-05, YOLO26 fixture parity (re-provisioned
with the pinned script) 6 confident rows, min IoU 0.999999.

Tier 1 status: [V-1] (tier 1 operators), [V-2] and [V-3] (YOLO families),
[V-4], [V-5], [V-6] pass. 

Tier 2 (2026-10-10, same tree). Seeded uniform [0,1) input; full shapes in
`input_sizes` for the symbolic-batch exports ([D-8]).

| Export | Opset (source) | Input | Max scaled diff |
| --- | --- | --- | --- |
| ViT (base) | 17 (`~/model_repository`) | 1×3×224×224 | 2.6e-06 |
| ViTPose (base) | 14 (`huggingface_playground`) | 1×3×256×192 | 5.3e-07 |
| Depth Anything V2 (small) | 18 (neuriplo-tasks recipe) | 1×3×518×518 | 2.7e-06 |
| VideoMAE (base) | 18 (neuriplo-tasks recipe) | 1×16×3×224×224 | 1.0e-06 |
| ViViT (b-16x2) | 18 (neuriplo-tasks recipe) | 1×32×3×224×224 | 2.6e-05 |
| TimeSformer | 14 (`~/model_repository`) | 1×8×3×224×224 | 3.4e-06 |

Tier 2 status: [V-1] (tier 2 operators), [V-2] and [V-3] (transformer
families) pass. Tiers 3–5 open.

Regression after [D-15] (double accumulation, P3f; 2026-10-10): all 19
tier 1 and tier 2 exports still pass, most with smaller differences:
YOLO11n-pose 4.9e-05 (was 1.5e-04), YOLO26n-pose 9.3e-05 (1.9e-04), YOLOv5x
2.9e-05 (7.7e-05), ViViT 4.5e-06 (2.6e-05), ResNet-18 3.8e-06 (2.2e-05).

Tier 3 (2026-10-10, [D-16], real image `bus.jpg`):

| Export | Input | Confident | Min box IoU | Max score diff |
| --- | --- | --- | --- | --- |
| RF-DETR (base) | 1x3x560x560, ImageNet-normalized | 5 | 0.999999 | 4.7e-08 |
| RF-DETR-Seg | 1x3x432x432, ImageNet-normalized | 5 (mask IoU 1.0) | 0.999996 | 1.7e-06 |
| RT-DETR v4 | 1x3x640x640, /255; sizes 640 | 5 | 0.999999 | 5.4e-07 |
| D-FINE/DEIM | 1x3x640x640, /255; sizes 640 | 5 | 0.999999 | 2.4e-07 |
| DEIMv2 | 1x3x640x640, /255; sizes 640 | 5 | 1.0 | 3.0e-07 |

Tier 3 status: complete. Tiers 4-5 continue in `olibartfast/nert`.

Tier 4 (2026-10-10, real image pair `bus.jpg` 256×320 in [-1,1] and a 4 px
shifted copy; `parity`, atol 1e-3 scaled). Manifest input column extended to
accept several `;`-separated files, filling the float32 graph inputs in order.

| Export | Input | Mode | Max scaled diff |
| --- | --- | --- | --- |
| RAFT-large (torchvision) | 1x3x256x320;1x3x256x320 | parity | 6.1e-05 |

Tier 4 status: complete (Phase A of `olibartfast/nert`
`specs/2026-10-10-raft-open-vocab`). Tier 5 continues in `olibartfast/nert`.
