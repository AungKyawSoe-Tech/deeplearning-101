"""
Comparison between 2-class and 3-class classification approaches
This script demonstrates the key differences in code structure
"""

import numpy as np
import tensorflow as tf
from keras import layers
import keras

def demonstrate_differences():
    print("=" * 60)
    print("2-CLASS vs 3-CLASS CLASSIFICATION COMPARISON")
    print("=" * 60)
    
    # Simulate some prediction outputs
    print("\n1. MODEL OUTPUT LAYER DIFFERENCES:")
    print("-" * 40)
    
    print("2-Class (Binary Classification):")
    print("  - Output units: 1")
    print("  - Activation: sigmoid (applied during prediction)")
    print("  - Interpretation: Single probability score")
    
    print("\n3-Class (Multi-class Classification):")
    print("  - Output units: 3")
    print("  - Activation: softmax (applied during prediction)")
    print("  - Interpretation: Probability distribution over all classes")
    
    # Demonstrate prediction differences
    print("\n2. PREDICTION OUTPUT EXAMPLES:")
    print("-" * 40)
    
    # Simulate 2-class predictions
    binary_logit = np.array([2.3])  # Raw logit output
    binary_prob = 1 / (1 + np.exp(-binary_logit[0]))  # Sigmoid
    
    print("2-Class Prediction:")
    print(f"  Raw logit: {binary_logit[0]:.2f}")
    print(f"  Sigmoid output: {binary_prob:.3f}")
    print(f"  Cat probability: {(1-binary_prob)*100:.1f}%")
    print(f"  Dog probability: {binary_prob*100:.1f}%")
    
    # Simulate 3-class predictions
    multiclass_logits = np.array([1.2, 2.8, 0.5])  # Raw logits for [Bird, Cat, Dog]
    multiclass_probs = np.exp(multiclass_logits) / np.sum(np.exp(multiclass_logits))  # Softmax
    class_names = ["Bird", "Cat", "Dog"]
    
    print("\n3-Class Prediction:")
    print(f"  Raw logits: {multiclass_logits}")
    print(f"  Softmax output: {multiclass_probs}")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name} probability: {multiclass_probs[i]*100:.1f}%")
    
    # Show loss function differences
    print("\n3. LOSS FUNCTION DIFFERENCES:")
    print("-" * 40)
    
    print("2-Class (Binary Crossentropy):")
    print("  Loss = -[y*log(p) + (1-y)*log(1-p)]")
    print("  Where y ∈ {0,1} and p is predicted probability")
    
    print("\n3-Class (Sparse Categorical Crossentropy):")
    print("  Loss = -log(p_true_class)")
    print("  Where p_true_class is probability of the correct class")
    
    # Show code structure differences
    print("\n4. CODE STRUCTURE DIFFERENCES:")
    print("-" * 40)
    
    print("2-Class Model Definition:")
    print("  units = 1 if num_classes == 2 else num_classes")
    print("  outputs = layers.Dense(units, activation=None)(x)")
    print("  loss = 'binary_crossentropy'")
    print("  metrics = ['binary_accuracy']")
    
    print("\n3-Class Model Definition:")
    print("  outputs = layers.Dense(num_classes, activation=None)(x)")
    print("  loss = 'sparse_categorical_crossentropy'")
    print("  metrics = ['sparse_categorical_accuracy']")
    
    # Show practical implications
    print("\n5. PRACTICAL IMPLICATIONS:")
    print("-" * 40)
    
    print("2-Class Limitations:")
    print("  ✗ Can only distinguish between two categories")
    print("  ✗ Not easily extensible to more classes")
    print("  ✗ Decision boundary is a single threshold")
    
    print("\n3-Class Advantages:")
    print("  ✓ Can distinguish between multiple categories")
    print("  ✓ Easily extensible to more classes")
    print("  ✓ Provides probability distribution over all classes")
    print("  ✓ Better representation of real-world scenarios")
    
    print("\n6. WHEN TO USE EACH APPROACH:")
    print("-" * 40)
    
    print("Use 2-Class (Binary) When:")
    print("  • You have exactly two mutually exclusive categories")
    print("  • The problem is naturally binary (yes/no, pass/fail)")
    print("  • You want to optimize for a single decision threshold")
    
    print("\nUse 3+ Class (Multi-class) When:")
    print("  • You have multiple distinct categories")
    print("  • Categories are mutually exclusive")
    print("  • You want probabilities for all possible outcomes")
    print("  • Future expansion to more classes is likely")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    demonstrate_differences()