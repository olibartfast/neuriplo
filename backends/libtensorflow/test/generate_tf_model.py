#!/usr/bin/env python3
"""
Generate a simple TensorFlow SavedModel for testing purposes.
Creates a basic ResNet-18 equivalent model for classification.
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_classifier():
    """Create a simple CNN classifier similar to ResNet-18 structure."""
    
    # Input layer: NHWC format (Batch, Height, Width, Channels)
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    
    # Simple CNN layers (ResNet-18 inspired)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks (simplified)
    for filters in [64, 128, 256, 512]:
        # Basic residual block
        residual = x
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        
        # Adjust residual connection if needed
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv2D(filters, 1, padding='same')(residual)
            
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Global average pooling and classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000, activation='softmax', name='output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def main():
    """Main function to create and save the model."""
    
    print("Creating TensorFlow test model...")
    
    # Create the model
    model = create_simple_classifier()
    
    # Compile the model (needed for SavedModel format)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("Model summary:")
    model.summary()
    
    # Save as SavedModel format
    save_path = "saved_model"
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    
    print(f"Saving model to {save_path}...")
    tf.saved_model.save(model, save_path)
    
    # Verify the saved model
    print("Verifying saved model...")
    loaded_model = tf.saved_model.load(save_path)
    
    # Test with dummy input
    dummy_input = tf.random.normal((1, 224, 224, 3))
    try:
        # Get the inference function
        infer = loaded_model.signatures["serving_default"]
        output = infer(input=dummy_input)
        print(f"Test inference successful. Output shape: {list(output.values())[0].shape}")
    except Exception as e:
        print(f"Warning: Could not test inference: {e}")
    
    print(f"Model successfully saved to {save_path}")
    print("TensorFlow test model creation completed!")

if __name__ == "__main__":
    main()
