#!/usr/bin/env python3
"""Serialize the YOLO preprocessing pipeline consumed by the neuriplo DALI backend.

The output is a .dali artifact loaded at run time by DALIInfer through the DALI
C API. Nothing here runs during inference: this is an offline build step, and the
serving path is pure C++.

This must reproduce neuriplo-tasks' YoloPreprocessor
(src/object_detection/detection_preprocessor.cpp) exactly. Three of its choices
are easy to get wrong, and every one of them fails silently -- boxes land in
slightly the wrong place rather than anything crashing:

  1. CENTERED padding. scale = min(W/w, H/h), then the resized image is pasted at
     ((W - new_w) / 2, (H - new_h) / 2). Padding at the bottom-right instead
     (DALI's fn.pad default) shifts every box by half the pad.

  2. ZERO fill, not 114. The C++ path allocates the letterbox with
     vision::Image::zeros. Most YOLO implementations pad with grey 114; this one
     does not, and matching the wrong constant changes what the network sees in
     the padded band.

  3. antialias=False. DALI antialiases when downscaling by default; the C++
     reference is a plain bilinear resize. With antialiasing on, tensors diverge
     on every image larger than the model input, worst on the largest ones.

Normalization is /255 alone (mean 0, std 255) into CHW float RGB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nvidia.dali import fn, pipeline_def, types

# Must match DALIInfer::kEncodedInputName.
INPUT_NAME = "IMAGE"


@pipeline_def(batch_size=1, num_threads=2, device_id=0)
def yolo_encoded_pipeline(size: int):
    """encoded JPEG bytes -> GPU decode -> letterbox -> /255 -> CHW float."""
    encoded = fn.external_source(name=INPUT_NAME, device="cpu", dtype=types.UINT8, ndim=1)

    # "mixed" decodes on the GPU via nvJPEG and returns a GPU tensor.
    image = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)

    # Aspect-preserving resize so the long side lands on `size`; this is
    # scale = min(size/w, size/h) expressed the way DALI spells it.
    resized = fn.resize(
        image,
        device="gpu",
        size=[size, size],
        mode="not_larger",
        interp_type=types.INTERP_LINEAR,
        antialias=False,
    )

    # Source dimensions, read from the JPEG header without decoding. This is a
    # second pipeline output because postprocessing has to map boxes back onto
    # the original frame, and the decode is the only place that knows its size.
    #
    # INT64 (height, width) is the convention the GPU postprocessing operators
    # expect, so preprocessing emits exactly that and one convention serves both
    # the CPU and GPU postprocess paths.
    full_shape = fn.peek_image_shape(encoded, dtype=types.INT64)
    source_shape = fn.slice(full_shape, 0, 2, axes=[0])

    # Centering and padding happen inside crop_mirror_normalize: a crop of
    # size x size positioned at the image centre, on an image that is smaller
    # than the crop in one axis, pads that axis symmetrically. That is exactly
    # the centered letterbox, expressed without any per-sample shape arithmetic
    # (fn.paste would need its ratio as a CPU node, which a GPU image cannot
    # supply). fill_values=0 matches vision::Image::zeros.
    preprocessed = fn.crop_mirror_normalize(
        resized,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(size, size),
        crop_pos_x=0.5,
        crop_pos_y=0.5,
        out_of_bounds_policy="pad",
        fill_values=0.0,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    )
    return preprocessed, source_shape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=640, help="square model input size")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dali/yolo_preprocess_640.dali"),
        help="path to write the serialized pipeline to",
    )
    args = parser.parse_args()

    pipe = yolo_encoded_pipeline(size=args.size, device_id=args.device_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pipe.serialize(filename=str(args.output))
    print(f"wrote {args.output} (input '{INPUT_NAME}', output 1x3x{args.size}x{args.size} FP32 CHW)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
