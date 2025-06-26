#!/bin/bash

# Generate OpenVINO IR from ONNX model for testing

set -e

echo "Generating OpenVINO IR for testing..."

# Check if mo (Model Optimizer) is available
if ! command -v mo &> /dev/null; then
    echo "OpenVINO Model Optimizer (mo) not found. Trying ovc..."
    if ! command -v ovc &> /dev/null; then
        echo "OpenVINO conversion tools not found. Please ensure OpenVINO is properly installed and sourced."
        exit 1
    fi
    CONVERTER="ovc"
else
    CONVERTER="mo"
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
echo "Converting ONNX to OpenVINO IR using $CONVERTER..."

if [ "$CONVERTER" = "mo" ]; then
    # Using legacy Model Optimizer
    mo --input_model resnet18.onnx \
       --output_dir . \
       --model_name resnet18 \
       --input_shape [1,3,224,224] \
       --compress_to_fp16
else
    # Using new OpenVINO Converter (ovc)
    ovc resnet18.onnx \
        --output_model resnet18 \
        --input_shape [1,3,224,224] \
        --compress_to_fp16
fi

if [ -f "resnet18.xml" ] && [ -f "resnet18.bin" ]; then
    echo "OpenVINO IR generated successfully:"
    echo "  - resnet18.xml"
    echo "  - resnet18.bin"
else
    echo "Failed to generate OpenVINO IR files"
    exit 1
fi
