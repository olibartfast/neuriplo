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
with torch.no_grad():
    onnx.export(
        model,
        example_input,
        model_name,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )

# Write the model path to a file
with open("model_path.txt", "w") as f:
    f.write(model_name)

print(f"Model exported to {model_name}")
