#!/usr/bin/env python3
"""
Simple script to create a test model for GGML backend testing.
This is a placeholder - in practice, you would convert real models to GGML format.
"""

import numpy as np
import struct
import os

def create_simple_ggml_model(output_path):
    """
    Create a simple placeholder GGML model file.
    This is just for testing the infrastructure - not a real model.
    """
    print(f"Creating placeholder GGML model at: {output_path}")
    
    # Create a simple binary file that looks like a GGML model
    # In practice, this would be a real GGML model file
    with open(output_path, 'wb') as f:
        # Write a simple header (this is not a real GGML format)
        f.write(b'GGML')  # Magic number
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', 1000))  # Output size
        f.write(struct.pack('<I', 224*224*3))  # Input size
        
        # Write some dummy weights (not real model weights)
        dummy_weights = np.random.randn(1000, 224*224*3).astype(np.float32)
        f.write(dummy_weights.tobytes())
    
    print(f"Created placeholder model with size: {os.path.getsize(output_path)} bytes")
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "/tmp/test_ggml_model.bin"
    
    try:
        model_path = create_simple_ggml_model(output_path)
        
        # Write the model path to a file for the test
        with open("model_path.txt", "w") as f:
            f.write(model_path + "\n")
        
        print(f"Model path written to model_path.txt: {model_path}")
        print("Note: This is a placeholder model for testing infrastructure only.")
        print("For real testing, you would need to convert an actual model to GGML format.")
        
    except Exception as e:
        print(f"Error creating test model: {e}")
        sys.exit(1)
