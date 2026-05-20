import argparse
import os

import torch
import torchvision.models as models
from executorch.exir import to_edge, to_edge_transform_and_lower

# The delegate is selected at export time and baked into the .pte. neuriplo links
# the matching ExecuTorch backend library at build time (see cmake/ExecuTorch.cmake).
# `xnnpack` is the default optimized-CPU delegate; `portable` lowers with no
# delegation and runs on the portable kernels. SDK-backed delegates (qnn, vulkan,
# coreml, mps, ethos-u) need their own export environments and are not wired here.
SUPPORTED_DELEGATES = ("xnnpack", "portable")

parser = argparse.ArgumentParser(description="Export a ResNet-18 classifier to ExecuTorch .pte")
parser.add_argument(
    "--delegate",
    default=os.environ.get("NEURIPLO_EXECUTORCH_DELEGATE", "xnnpack"),
    choices=SUPPORTED_DELEGATES,
    help="ExecuTorch delegate to lower the model for (default: xnnpack)",
)
parser.add_argument(
    "--output-dir",
    default="/workspace",
    help="Directory for the .pte and model_path.txt (default: /workspace)",
)
args = parser.parse_args()

weights = getattr(models, "ResNet18_Weights", None)
model = models.resnet18(weights=weights.DEFAULT if weights is not None else None).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)
exported_program = torch.export.export(model, sample_inputs)

if args.delegate == "xnnpack":
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
else:  # portable: no delegation, runs on the portable kernels
    executorch_program = to_edge(exported_program).to_executorch()

pte_name = f"resnet18_{args.delegate}.pte"
with open(os.path.join(args.output_dir, pte_name), "wb") as f:
    f.write(executorch_program.buffer)

with open(os.path.join(args.output_dir, "model_path.txt"), "w") as f:
    f.write(pte_name)

print(f"Exported {pte_name} (delegate: {args.delegate})")
