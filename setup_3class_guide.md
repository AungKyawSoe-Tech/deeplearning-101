# 3-Class Classification Setup Guide

## Overview
This guide explains how to extend the original 2-class (Cat vs Dog) classification to support 3 classes.

## Files Created
1. **helloworld_3class.py** - Main training script for 3-class classification
2. **testmodel_3class.py** - Testing script to load and use the trained 3-class model
3. **setup_3class_guide.md** - This setup guide

## Key Changes from 2-Class to 3-Class

### 1. Model Architecture Changes
- **Output layer**: Changed from 1 unit (binary) to 3 units (multi-class)
- **Loss function**: Changed from `BinaryCrossentropy` to `SparseCategoricalCrossentropy`
- **Metrics**: Changed from `BinaryAccuracy` to `SparseCategoricalAccuracy`

### 2. Data Structure
The original setup expects:
```
PetImages/
├── Cat/
└── Dog/
```

For 3-class classification, you need:
```
PetImages/
├── Bird/
├── Cat/
└── Dog/
```

### 3. Prediction Logic
- **Original**: Binary classification with sigmoid activation, outputs probability for one class
- **New**: Multi-class classification with softmax activation, outputs probability for all 3 classes

## Setup Instructions

### Step 1: Prepare Data Directory
Create a Bird folder in your PetImages directory:
```bash
mkdir PetImages/Bird
```

### Step 2: Add Bird Images
You need to add bird images to the `PetImages/Bird/` folder. You can:

1. **Download bird images from a dataset** (recommended):
   - Use Kaggle's bird species dataset
   - Use CIFAR-10 bird class
   - Download from Google Images (ensure proper licensing)

2. **Use existing bird images** you have

3. **For testing purposes**, you can use any images temporarily

### Step 3: Verify Data Structure
Your directory should look like this:
```
PetImages/
├── Bird/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Cat/
│   ├── existing_cat_images...
└── Dog/
    ├── existing_dog_images...
```

### Step 4: Train the 3-Class Model
Run the training script:
```bash
python helloworld_3class.py
```

This will:
- Clean up invalid images
- Create training and validation datasets
- Train the model for 25 epochs
- Save the trained weights to `saved_3class.weights.h5`

### Step 5: Test the Model
Run the testing script:
```bash
python testmodel_3class.py
```

## Expected Output Format

### Original 2-Class Output:
```
This image is 15.23% cat and 84.77% dog.
```

### New 3-Class Output:
```
Prediction for image1.jpg:
  Bird: 12.45%
  Cat: 78.32%
  Dog: 9.23%
  -> Prediction: Cat (78.32%)
```

## Technical Details

### Model Changes
```python
# Original (2-class)
units = 1 if num_classes == 2 else num_classes
outputs = layers.Dense(units, activation=None)(x)

# New (3-class)
outputs = layers.Dense(num_classes, activation=None)(x)  # Always num_classes units
```

### Compilation Changes
```python
# Original (2-class)
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

# New (3-class)
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
```

### Prediction Changes
```python
# Original (2-class)
score = float(tf.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

# New (3-class)
predicted_class = tf.nn.softmax(predictions[0])
for i, class_name in enumerate(class_names):
    confidence = float(predicted_class[i]) * 100
    print(f"  {class_name}: {confidence:.2f}%")
```

## Troubleshooting

### Common Issues:
1. **"Bird folder not found"** - Create the Bird directory and add images
2. **"No images in Bird folder"** - Add at least 100+ bird images for good training
3. **Poor performance** - Ensure bird images are of similar quality to cat/dog images
4. **Class imbalance** - Try to have roughly equal numbers of images per class

### Recommendations:
- Use at least 500+ images per class for better results
- Ensure images are diverse (different backgrounds, angles, lighting)
- Consider data augmentation if you have limited images
- Monitor training/validation accuracy to detect overfitting

## Next Steps
Once you have the 3-class model working, you can:
1. Add more classes (4, 5, etc.) by following the same pattern
2. Experiment with different architectures
3. Try transfer learning with pre-trained models
4. Implement data augmentation techniques