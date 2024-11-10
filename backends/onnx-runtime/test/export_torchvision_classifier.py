import torch
import torchvision.models as models
import torch.onnx as onnx


# Load pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()


# Example input to trace the model
example_input = torch.rand(1, 3, 224, 224)


# Export the model to ONNX
model_name = "resnet18.onnx"
model_path = f"/workspace/{model_name}"
with torch.no_grad():
    onnx.export(
        model,
        example_input,
        model_path,
        opset_version=12,  # You can adjust the opset version as needed
        input_names=["input"],
        output_names=["output"],
    )