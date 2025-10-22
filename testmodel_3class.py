# Test script for 3-class classification model
# Loads the trained 3-class model and makes predictions

import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

# Augment image data
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Build model - Same architecture as training script
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)

    return keras.Model(inputs, outputs)

# Configuration
image_size = (180, 180)
num_classes = 3
class_names = ["Bird", "Cat", "Dog"]  # Alphabetical order as Keras will assign

# Create model
model = make_model(input_shape=image_size + (3,), num_classes=num_classes)

# Load weights
filepath = 'saved_3class.weights.h5'
if os.path.exists(filepath):
    model.load_weights(filepath)
    print(f"Loaded model weights from {filepath}")
else:
    print(f"Warning: {filepath} not found. Please train the 3-class model first.")
    exit()

# Prediction function
def predict_image(image_path, model, class_names):
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found!")
        return
        
    img = keras.utils.load_img(image_path, target_size=image_size)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicting: {os.path.basename(image_path)}")
    plt.axis("off")
    plt.show()

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = tf.nn.softmax(predictions[0])
    
    print(f"\nPrediction for {os.path.basename(image_path)}:")
    for i, class_name in enumerate(class_names):
        confidence = float(predicted_class[i]) * 100
        print(f"  {class_name}: {confidence:.2f}%")
    
    predicted_label = np.argmax(predicted_class)
    max_confidence = float(predicted_class[predicted_label]) * 100
    print(f"  -> Prediction: {class_names[predicted_label]} ({max_confidence:.2f}%)")
    print("-" * 60)

# Test with available images
print("Testing 3-class classification model...")
print(f"Classes: {class_names}")
print("=" * 60)

test_images = [
    "TestImages/image1.jpg", 
    "TestImages/image2.jpg",
    "TestImages/image3.jpg",
    "TestImages/image4.jpg"
]

# Also test with some images from PetImages if they exist
pet_test_images = [
    "PetImages/Cat/6779.jpg",
    "PetImages/Dog/0.jpg",
    "PetImages/Bird/0.jpg"  # If bird images exist
]

all_test_images = test_images + pet_test_images

found_images = []
for img_path in all_test_images:
    if os.path.exists(img_path):
        found_images.append(img_path)

if found_images:
    for img_path in found_images:
        predict_image(img_path, model, class_names)
else:
    print("No test images found. Please add some images to test with.")
    print("Expected locations:")
    for img_path in all_test_images:
        print(f"  - {img_path}")

print(f"\nModel ready for 3-class classification: {class_names}")