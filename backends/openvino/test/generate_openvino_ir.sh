#!/bin/bash

# Generate OpenVINO IR from ONNX model for testing

set -e

echo "Generating OpenVINO IR for testing..."

# ovc ships with the openvino wheel; the legacy Model Optimizer (mo) ended with
# openvino-dev 2024.6 and no longer works against the pinned runtime.
if ! command -v ovc &> /dev/null; then
    echo "OpenVINO converter (ovc) not found. Install the openvino wheel matching OPENVINO_VERSION."
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

# Convert ONNX to OpenVINO IR
echo "Converting ONNX to OpenVINO IR using ovc..."
ovc resnet18.onnx \
    --output_model resnet18 \
    --input "[1,3,224,224]" \
    --compress_to_fp16=True

if [ -f "resnet18.xml" ] && [ -f "resnet18.bin" ]; then
    echo "OpenVINO IR generated successfully:"
    echo "  - resnet18.xml"
    echo "  - resnet18.bin"
else
    echo "Failed to generate OpenVINO IR files"
    exit 1
fi
