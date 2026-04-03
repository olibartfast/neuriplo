"""Export a torchvision ResNet-18 classifier to ONNX for MIGraphX tests."""
import torch
import torchvision.models as models

model = models.resnet18(weights=None)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "resnet18.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11,
)
print("Exported resnet18.onnx")

with open("model_path.txt", "w") as f:
    f.write("resnet18.onnx\n")
print("Wrote model_path.txt")
