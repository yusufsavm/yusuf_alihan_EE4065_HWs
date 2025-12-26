# EE 4065 - Embedded Digital Image Processing: Homework 4

**Course:** EE 4065 - Embedded Digital Image Processing
**Assignment:** Homework 4
**Due Date:** December 26, 2025

## Student Information

| Name | Student ID |
| :--- | :--- |
| Yusuf Oruç | 150720036 |
| Alihan Kocaakman | 150720065 |

## Project Overview

This repository contains the solutions for **Homework 4**, based on the applications described in the textbook *Embedded Machine Learning with Microcontrollers*. The project focuses on **Handwritten Digit Recognition** using the MNIST dataset.

We implemented two different approaches to classify handwritten digits by extracting **Hu Moments** (7 invariant features) from images:

1.  **Q1 (Section 10.9):** Binary classification (Zero vs. Not-Zero) using a **Single Neuron**.
2.  **Q2 (Section 11.8):** Multi-class classification (Digits 0-9) using a **Multilayer Perceptron (MLP)**.

---

## Q1 - Section 10.9: Single Neuron Classifier

**Objective:** Distinguish between the digit "0" and "Not 0" (all other digits).

* **Model:** Single Neuron (Perceptron) with Sigmoid activation.
* **Feature Extraction:** 7 Hu Moments.
* **Result:** The confusion matrix below shows the classification performance.

### Confusion Matrix (Q1)
![Figure_1](https://github.com/user-attachments/assets/482cbd20-6963-412f-9c9e-37e4a69ebf27)

### Source Code (Q1)

```python
import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import struct
from matplotlib import pyplot as plt

# --- Manual Data Loading Functions ---
def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# File Paths
train_img_path = os.path.join("train-images.idx3-ubyte")
train_label_
```
---

## Q2 - Section 11.8: Neural Network Classifier (Multiclass)

**Objective:** Classify handwritten digits into 10 separate classes (0-9) using a Multilayer Perceptron (MLP).

### Methodology
* **Model Architecture:** A Feed-Forward Neural Network with 3 layers.
    * **Input:** 7 Hu Moments (Shape features).
    * **Hidden Layer 1:** 100 Neurons, ReLU activation.
    * **Hidden Layer 2:** 100 Neurons, ReLU activation.
    * **Output Layer:** 10 Neurons, Softmax activation (representing probabilities for digits 0-9).
* **Training:**
    * **Optimizer:** Adam (Learning rate: 1e-4).
    * **Loss Function:** Sparse Categorical Cross-Entropy.
    * **Callbacks:** `ModelCheckpoint` (to save the best model) and `EarlyStopping` (to prevent overfitting).
* **Preprocessing:** Z-score normalization was applied to the Hu Moments to ensure stable convergence.

### Results (Confusion Matrix)
The confusion matrix below demonstrates the model's performance on the test set. The diagonal elements represent correctly classified digits.

![Neural Network Confusion Matrix](https://github.com/user-attachments/assets/8ab4a9fb-60b7-4c10-8060-065eaf737602)

### Source Code (Q2)

```python
import os
import numpy as np
import cv2
import struct 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

# --- Manual Data Loading Functions ---
def load_images(path):
    with open(path, 'rb') as f:
        # Magic number, image count, rows, cols (Big Endian)
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        # Magic number, label count
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# File Paths
train_img_path = os.path.join("train-images.idx3-ubyte")
train_label_path = os.path.join("train-labels.idx1-ubyte")
test_img_path = os.path.join("t10k-images.idx3-ubyte")
test_label_path = os.path.join("t10k-labels.idx1-ubyte")

print("Loading Data...")
train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images = load_images(test_img_path)
test_labels = load_labels(test_label_path)

train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

print("Calculating Hu Moments...")
for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True)
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# --- Normalization (Crucial Step) ---
features_mean = np.mean(train_huMoments, axis=0)
