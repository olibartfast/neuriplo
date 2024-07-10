import torch
import torchvision.models as models

# Load pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Example input to trace the model
example_input = torch.rand(1, 3, 224, 224)

# Trace and save the model
traced_model = torch.jit.trace(model, example_input)
model_name = "resnet18_traced.pt"
model_path = f"/workspace/{model_name}"
traced_model.save(model_path)

# Write the model path to a file
with open("/workspace/model_path.txt", "w") as f:
    f.write(model_name)