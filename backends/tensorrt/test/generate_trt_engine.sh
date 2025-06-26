#!/bin/bash

# Generate TensorRT Engine from ONNX model for testing

set -e

echo "Generating TensorRT Engine for testing..."

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "trtexec not found. Please ensure TensorRT is properly installed and in PATH."
    exit 1
fi

# First, check if we have an ONNX model, if not generate one
if [ ! -f "resnet18.onnx" ]; then
    echo "ONNX model not found. Generating ResNet-18 ONNX model..."
    
    if [ -f "export_torchvision_classifier.py" ]; then
        python3 export_torchvision_classifier.py
    else
        echo "Cannot find model generation script. Creating a simple one..."
        cat > temp_export.py << 'EOF'
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

print(f"Model exported to {model_name}")
EOF
        python3 temp_export.py
        rm temp_export.py
    fi
fi

# Generate TensorRT engine
echo "Converting ONNX to TensorRT engine..."
trtexec --onnx=resnet18.onnx \
        --saveEngine=resnet18.engine \
        --fp16 \
        --workspace=1024 \
        --verbose

if [ -f "resnet18.engine" ]; then
    echo "TensorRT engine generated successfully: resnet18.engine"
else
    echo "Failed to generate TensorRT engine"
    exit 1
fi
