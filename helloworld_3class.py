# Extended from original helloworld.py to support 3-class classification
# Adapted from https://keras.io/examples/vision/image_classification_from_scratch/

import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Build model - Modified for 3-class classification
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
    
    # For multi-class classification, we need num_classes output units
    outputs = layers.Dense(num_classes, activation=None)(x)

    return keras.Model(inputs, outputs)


# Data cleanup - Modified to handle 3 classes (Cat, Dog, Bird)
# You'll need to add a "Bird" folder to your PetImages directory
num_skipped = 0
class_names = ["Cat", "Dog", "Bird"]  # Define your 3 classes here

for folder_name in class_names:
    folder_path = os.path.join("PetImages", folder_name)
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist. Please create this folder and add images.")
        continue
        
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as fobj:
                is_jfif = b"JFIF" in fobj.read(10)
        except Exception:
            is_jfif = False

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

# Generate Datasets
image_size = (180, 180)
batch_size = 16

train_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Print class names for verification
print("Class names:", train_ds.class_names)

# Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(f"Class: {train_ds.class_names[int(labels[i])]}")
        plt.axis("off")

# Augment image data
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

# Visualize augmented samples
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    augmented_images = data_augmentation(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[i]).astype("uint8"))
        plt.axis("off")

# Apply augmentation to training set
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Make model - Now with 3 classes
num_classes = 3
model = make_model(input_shape=image_size + (3,), num_classes=num_classes)
keras.utils.plot_model(model, show_shapes=True)

# Train model - Modified for multi-class classification
epochs = 25
callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}_3class.keras")]

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Changed from BinaryCrossentropy
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],  # Changed from BinaryAccuracy
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# Save the weights of the trained model
filepath = 'saved_3class.weights.h5'
model.save_weights(filepath)

# Prediction function for 3 classes
def predict_image(image_path, model, class_names):
    img = keras.utils.load_img(image_path, target_size=image_size)
    plt.imshow(img)
    plt.title(f"Predicting: {image_path}")
    plt.axis("off")
    plt.show()

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = tf.nn.softmax(predictions[0])
    
    print(f"\nPrediction for {image_path}:")
    for i, class_name in enumerate(class_names):
        confidence = float(predicted_class[i]) * 100
        print(f"  {class_name}: {confidence:.2f}%")
    
    predicted_label = np.argmax(predicted_class)
    print(f"  -> Prediction: {class_names[predicted_label]} ({float(predicted_class[predicted_label]) * 100:.2f}%)")
    print("-" * 50)

# Test predictions with existing images (if they exist)
test_images = [
    "PetImages/Cat/6779.jpg",
    "TestImages/image1.jpg", 
    "TestImages/image4.jpg"
]

for img_path in test_images:
    if os.path.exists(img_path):
        predict_image(img_path, model, train_ds.class_names)
    else:
        print(f"Image {img_path} not found, skipping...")

print("\nTraining completed! The model now classifies 3 classes:")
print(f"Classes: {train_ds.class_names}")
print(f"Model weights saved to: {filepath}")