import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower


weights = getattr(models, "ResNet18_Weights", None)
model = models.resnet18(weights=weights.DEFAULT if weights is not None else None).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)

exported_program = torch.export.export(model, sample_inputs)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
).to_executorch()

with open("/workspace/resnet18_xnnpack.pte", "wb") as f:
    f.write(executorch_program.buffer)

with open("/workspace/model_path.txt", "w") as f:
    f.write("resnet18_xnnpack.pte")
