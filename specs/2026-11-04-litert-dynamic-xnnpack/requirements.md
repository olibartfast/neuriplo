# LiteRT dynamic XNNPACK correctness

The maintainer requested fixing all KNOWN-BAD coverage cases, following explicit authorization for cross-repo inference correctness work.

R1: Enable XNNPACK subgraph reshaping for the LiteRT CPU interpreter so reconverted EdgeCrafter matches ONNX on identical inputs.
R2: Delegate lifetime must exceed interpreter lifetime. Allocation/delegation failures must be explicit.
R3: Preserve public input/output layouts, dtype and GPU-warning behavior. No runtime version bump, external dependency or custom-kernel rewrite.

Evidence: TF 2.19 default delegate rejects dynamic tensor #324. Python LiteRT builtin path reproduces deflated scores. Explicit C++ XNNPACK with ENABLE_SUBGRAPH_RESHAPING returns ONNX scores 0.929990/0.927981/0.780462.
