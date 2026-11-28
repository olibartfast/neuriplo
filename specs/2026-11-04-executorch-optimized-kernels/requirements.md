# ExecuTorch optimized kernel registration

Maintainer requests resolving RF-DETR blocked cells and Slurm parallel validation.
R1: Build the pinned v1.2.0 SDK with its existing optimized kernels, including aten::linear.out. No runtime version or mathematical kernel changes.
R2: Link the SDK-provided combined optimized/portable registration library when available; preserve compatibility with portable-only SDK installations. Never link overlapping registration libraries together.
R3: Validate detection, segmentation and keypoint artifacts against existing ORT references; do not infer numerical correctness from load success.
