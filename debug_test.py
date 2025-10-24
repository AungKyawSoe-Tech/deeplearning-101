#!/usr/bin/env python3
"""
Simple test to debug the 3-class classification issue
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Clear session
tf.keras.backend.clear_session()

# Create simple test data
image_size = (180, 180)
num_classes = 3
batch_size = 16

# Create dummy data for testing
np.random.seed(42)
dummy_images = np.random.random((batch_size, 180, 180, 3)).astype('float32')
dummy_labels = np.random.randint(0, num_classes, (batch_size,))

print(f"Dummy images shape: {dummy_images.shape}")
print(f"Dummy labels shape: {dummy_labels.shape}")
print(f"Label values: {dummy_labels}")

# Create simple model
model = keras.Sequential([
    keras.layers.Input(shape=image_size + (3,)),
    keras.layers.Rescaling(1.0 / 255),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(num_classes)  # 3 outputs for 3 classes
])

print(f"Model output shape: {model.output_shape}")

# Compile with sparse categorical crossentropy
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print("Model compiled successfully!")

# Test prediction
try:
    prediction = model.predict(dummy_images[:1])
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction values: {prediction}")
    
    # Test training on a small batch
    print("\nTesting training...")
    history = model.fit(dummy_images, dummy_labels, epochs=1, verbose=1)
    print("Training successful!")
    
except Exception as e:
    print(f"Error during prediction/training: {e}")
    import traceback
    traceback.print_exc()