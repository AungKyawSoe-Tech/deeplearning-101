#  Predict

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

# Build model
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
    units = 1 if num_classes == 2 else num_classes
    outputs = layers.Dense(units, activation=None)(x)

    return keras.Model(inputs, outputs)


image_size = (180, 180)

model = make_model(input_shape=image_size + (3,), num_classes=2)

filepath = 'saved.weights.h5'
model.load_weights(filepath)


img = keras.utils.load_img("TestImages/image1.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(tf.sigmoid(predictions[0][0]))
print(f"This image #1 is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

img = keras.utils.load_img("TestImages/image2.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(tf.sigmoid(predictions[0][0]))
print(f"This image #2 is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

img = keras.utils.load_img("TestImages/image3.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(tf.sigmoid(predictions[0][0]))
print(f"This image #3 is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

img = keras.utils.load_img("TestImages/image4.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(tf.sigmoid(predictions[0][0]))
print(f"This image #4 is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")



