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

Q2-) (50 points) Section 11.8 Application: Handwritten Digit Recognition from Digital Images


confusion table: <img width="1000" height="615" alt="Figure_2" src="https://github.com/user-attachments/assets/8ab4a9fb-60b7-4c10-8060-065eaf737602" />


Code

import os
import numpy as np
import cv2
import struct # Dosya okuma için gerekli
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

# --- MANUEL VERİ YÜKLEME FONKSİYONLARI (Hata almamak için) ---
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
# -----------------------------------------------------------

# Dosya yolları
train_img_path = os.path.join("train-images.idx3-ubyte")
train_label_path = os.path.join("train-labels.idx1-ubyte")
test_img_path = os.path.join("t10k-images.idx3-ubyte")
test_label_path = os.path.join("t10k-labels.idx1-ubyte")

print("Veriler yükleniyor...")
train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images = load_images(test_img_path)
test_labels = load_labels(test_label_path)

train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

print("Öznitelikler (Hu Moments) hesaplanıyor...")
for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True)
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# --- ÖNEMLİ EKLEME: NORMALİZASYON ---
# Kitaptaki Listing 11.6'da bazen atlanmış olsa da, Neural Network'lerin
# Hu momentleri gibi çok küçük sayılarla düzgün çalışması için bu şarttır.
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# Model Mimarisi (Kitaptaki gibi: 100 -> 100 -> 10)
model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=[7], activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax") # 10 sınıf için Softmax
])

# Model Derleme
# SparseCategoricalCrossentropy: Etiketler integer (0,1,2...) olduğu için kullanılır.
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(1e-4), # Learning rate 1e-4
              metrics=['accuracy'])

# Callbacks: En iyi modeli kaydet ve eğitim iyileşmezse erken durdur
mc_callback = ModelCheckpoint("mlp_mnist_model.h5", save_best_only=True)
es_callback = EarlyStopping(monitor="loss", patience=5)

print("Model eğitiliyor...")
# labels olduğu gibi veriliyor (0-9 arası), binary çevirmeye gerek yok
history = model.fit(train_huMoments, train_labels,
                    epochs=1000, # Early stopping olduğu için yüksek verilebilir
                    verbose=1,
                    callbacks=[mc_callback, es_callback])

# Tahmin ve Sonuçlar
print("Test yapılıyor...")
nn_preds = model.predict(test_huMoments)
predicted_classes = np.argmax(nn_preds, axis=1) # En yüksek olasılıklı sınıfı seç

categories = np.unique(test_labels) # 0, 1, 2... 9

conf_matrix = confusion_matrix(test_labels, predicted_classes)
# Büyük matris olduğu için görselleştirmeyi biraz büyütelim
fig, ax = plt.subplots(figsize=(10, 10))
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
cm_display.plot(ax=ax, cmap='viridis') # Okunabilirliği artırmak için renk haritası
cm_display.ax_.set_title("Neural Network Confusion Matrix (Multiclass)")
plt.show()

print("İşlem tamamlandı. 'mlp_mnist_model.h5' kaydedildi.")
