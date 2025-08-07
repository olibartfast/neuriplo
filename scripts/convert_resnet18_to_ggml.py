#!/usr/bin/env python3
"""
Convert ResNet18 PyTorch model to GGML format for inference testing.
This script downloads a pretrained ResNet18 model and converts it to GGML format.
"""

import os
import sys
import torch
import torchvision.models as models
import numpy as np
import struct
import argparse
from pathlib import Path

def create_ggml_header(f, model_info):
    """Write GGML model header"""
    # GGML file format header (simplified version)
    f.write(b'ggml')  # Magic number
    f.write(struct.pack('<I', 1))  # Version
    f.write(struct.pack('<I', model_info['num_layers']))
    f.write(struct.pack('<I', model_info['input_size']))
    f.write(struct.pack('<I', model_info['output_size']))

def convert_conv_layer(f, conv_layer, layer_name):
    """Convert a convolutional layer to GGML format"""
    # Write layer type
    f.write(struct.pack('<I', 1))  # CONV layer type
    
    # Write layer name
    name_bytes = layer_name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)
    
    # Write layer parameters
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size[0] if isinstance(conv_layer.kernel_size, tuple) else conv_layer.kernel_size
    stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
    padding = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
    
    f.write(struct.pack('<IIII', in_channels, out_channels, kernel_size, stride))
    f.write(struct.pack('<I', padding))
    
    # Write weights (transposed for GGML format)
    weights = conv_layer.weight.data.numpy().astype(np.float32)
    f.write(struct.pack('<I', weights.size))
    f.write(weights.tobytes())
    
    # Write bias
    if conv_layer.bias is not None:
        bias = conv_layer.bias.data.numpy().astype(np.float32)
        f.write(struct.pack('<I', bias.size))
        f.write(bias.tobytes())
    else:
        f.write(struct.pack('<I', 0))

def convert_bn_layer(f, bn_layer, layer_name):
    """Convert a batch normalization layer to GGML format"""
    # Write layer type
    f.write(struct.pack('<I', 2))  # BN layer type
    
    # Write layer name
    name_bytes = layer_name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)
    
    # Write layer parameters
    num_features = bn_layer.num_features
    f.write(struct.pack('<I', num_features))
    
    # Write running mean and variance
    running_mean = bn_layer.running_mean.data.numpy().astype(np.float32)
    running_var = bn_layer.running_var.data.numpy().astype(np.float32)
    
    f.write(struct.pack('<I', running_mean.size))
    f.write(running_mean.tobytes())
    f.write(struct.pack('<I', running_var.size))
    f.write(running_var.tobytes())
    
    # Write weight and bias
    if bn_layer.weight is not None:
        weight = bn_layer.weight.data.numpy().astype(np.float32)
        f.write(struct.pack('<I', weight.size))
        f.write(weight.tobytes())
    else:
        f.write(struct.pack('<I', 0))
    
    if bn_layer.bias is not None:
        bias = bn_layer.bias.data.numpy().astype(np.float32)
        f.write(struct.pack('<I', bias.size))
        f.write(bias.tobytes())
    else:
        f.write(struct.pack('<I', 0))

def convert_linear_layer(f, linear_layer, layer_name):
    """Convert a linear layer to GGML format"""
    # Write layer type
    f.write(struct.pack('<I', 3))  # LINEAR layer type
    
    # Write layer name
    name_bytes = layer_name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)
    
    # Write layer parameters
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    f.write(struct.pack('<II', in_features, out_features))
    
    # Write weights (transposed for GGML format)
    weights = linear_layer.weight.data.numpy().astype(np.float32)
    f.write(struct.pack('<I', weights.size))
    f.write(weights.tobytes())
    
    # Write bias
    if linear_layer.bias is not None:
        bias = linear_layer.bias.data.numpy().astype(np.float32)
        f.write(struct.pack('<I', bias.size))
        f.write(bias.tobytes())
    else:
        f.write(struct.pack('<I', 0))

def convert_resnet18_to_ggml(model, output_path):
    """Convert ResNet18 model to GGML format"""
    print(f"Converting ResNet18 to GGML format: {output_path}")
    
    # Count layers for header
    num_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):
            num_layers += 1
    
    model_info = {
        'num_layers': num_layers,
        'input_size': 3 * 224 * 224,  # ResNet18 input size
        'output_size': 1000,  # ImageNet classes
    }
    
    with open(output_path, 'wb') as f:
        # Write header
        create_ggml_header(f, model_info)
        
        # Convert each layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                convert_conv_layer(f, module, name)
                print(f"  Converted Conv2d layer: {name}")
            elif isinstance(module, torch.nn.BatchNorm2d):
                convert_bn_layer(f, module, name)
                print(f"  Converted BatchNorm2d layer: {name}")
            elif isinstance(module, torch.nn.Linear):
                convert_linear_layer(f, module, name)
                print(f"  Converted Linear layer: {name}")
    
    print(f"✓ ResNet18 converted to GGML format: {output_path}")
    print(f"  Model size: {os.path.getsize(output_path)} bytes")
    print(f"  Number of layers: {num_layers}")

def download_resnet18_model():
    """Download pretrained ResNet18 model"""
    print("Downloading pretrained ResNet18 model...")
    model = models.resnet18(pretrained=True)
    model.eval()
    print("✓ ResNet18 model downloaded successfully")
    return model

def create_test_script(output_path):
    """Create a test script to verify the converted model"""
    test_script = f"""#!/bin/bash
# Test script for GGML ResNet18 model

MODEL_PATH="{output_path}"
echo "Testing GGML model: $MODEL_PATH"

if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model file exists"
    echo "  Size: $(stat -c%s "$MODEL_PATH") bytes"
else
    echo "✗ Model file not found"
    exit 1
fi

# Write model path for tests
echo "$MODEL_PATH" > model_path.txt
echo "✓ Model path written to model_path.txt"
"""
    
    script_path = os.path.join(os.path.dirname(output_path), "test_ggml_model.sh")
    with open(script_path, 'w') as f:
        f.write(test_script)
    
    os.chmod(script_path, 0o755)
    print(f"✓ Test script created: {script_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert ResNet18 PyTorch model to GGML format')
    parser.add_argument('--output', '-o', default='resnet18.ggml', 
                       help='Output GGML model path (default: resnet18.ggml)')
    parser.add_argument('--test-dir', '-t', default='backends/ggml/test',
                       help='Test directory to copy model to (default: backends/ggml/test)')
    
    args = parser.parse_args()
    
    try:
        # Download model
        model = download_resnet18_model()
        
        # Convert to GGML format
        convert_resnet18_to_ggml(model, args.output)
        
        # Copy to test directory if specified
        if args.test_dir and os.path.exists(args.test_dir):
            test_model_path = os.path.join(args.test_dir, 'resnet18.ggml')
            import shutil
            shutil.copy2(args.output, test_model_path)
            print(f"✓ Model copied to test directory: {test_model_path}")
            
            # Create test script in test directory
            create_test_script(test_model_path)
        
        print("\n=== Conversion Complete ===")
        print(f"GGML model: {args.output}")
        print("You can now use this model with the GGML backend in InferenceEngines")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
