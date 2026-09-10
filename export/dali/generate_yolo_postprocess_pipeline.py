#!/usr/bin/env python3
"""Serialize the YOLO segmentation GPU postprocessing pipeline.

Decodes the model's raw outputs into the platform ensemble contract's
packed-mask envelope on the GPU, so postprocessing never touches the host.

The decode itself is a custom CUDA DALI operator supplied as a plugin: DALI has
no built-in NMS or mask assembly. The plugin must be loaded before the pipeline
is serialized here, and again before it is deserialized at serve time -- the
neuriplo DALI backend takes it through the model_path suffix
"pipeline.dali|plugin=<path.so>".

The plugin and the serialized pipeline are tied to one DALI version. Generate
inside the container (export/dali/generate_pipelines.sh) so the Python DALI
matches the C++ libdali the backend links.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nvidia.dali import fn, pipeline_def, plugin_manager, types


@pipeline_def
def pipeline(confidence_threshold: float, mask_threshold: float):
    # Names match what the ensemble graph maps the model's outputs onto.
    detections = fn.external_source(name="DETECTIONS", device="gpu", ndim=2, dtype=types.FLOAT)
    prototypes = fn.external_source(name="PROTOTYPES", device="gpu", ndim=3, dtype=types.FLOAT)
    original_size = fn.external_source(name="ORIGINAL_SIZE", device="gpu", ndim=1, dtype=types.INT64)

    outputs = fn.yolo26_seg_mask_postprocess(
        detections,
        prototypes,
        original_size,
        device="gpu",
        confidence_threshold=confidence_threshold,
        mask_threshold=mask_threshold,
    )
    return tuple(outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True, help="DALI operator plugin (.so)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    plugin_manager.load_library(args.plugin)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline(
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
        batch_size=1,
        num_threads=1,
        device_id=args.device_id,
    ).serialize(filename=str(output))
    print(f"wrote {output} (inputs DETECTIONS, PROTOTYPES, ORIGINAL_SIZE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
