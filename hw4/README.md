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

**Based on the Figure **(Confusion Matrix):

High Sensitivity for Class "0": The model demonstrates strong performance in identifying the target class. Out of 980 actual "0" digits in the test set, the model correctly classified 945 (True Negatives in the context of "0" vs "Not-0"). This indicates that the extracted Hu Moments successfully capture the circular geometric features of the digit zero.

High False Negative Rate: The most significant error is observed in the bottom-left quadrant (1476 instances). This represents the "False Negative" count where the model incorrectly classified non-zero digits (1-9) as "0".

### Source Code (Q1)

```python
import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import struct  # <--- YENİ EKLENDİ: Dosya okumak için gerekli
from matplotlib import pyplot as plt

# Veri yükleme fonksiyonlarını manuel tanımlamak daha güvenli olabilir 
# veya 'python-mnist' kütüphanesinin 'MNIST' sınıfını kullanabilirsin.
# Senin kodundaki 'load_images' fonksiyonu 'mnist' kütüphanesinden geliyor.
def load_images(path):
    with open(path, 'rb') as f:
        # Magic number, image count, rows, cols (Big Endian formatında okunur)
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Veriyi numpy array olarak oku ve yeniden şekillendir
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        # Magic number, label count
        magic, num = struct.unpack(">II", f.read(8))
        # Etiketleri oku
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Dosya yolları
train_img_path = os.path.join("train-images.idx3-ubyte")
train_label_path = os.path.join("train-labels.idx1-ubyte")
test_img_path = os.path.join("t10k-images.idx3-ubyte")
test_label_path = os.path.join("t10k-labels.idx1-ubyte")

# Veriyi yükle
train_images = load_images(train_img_path)
train_labels = np.array(load_labels(train_label_path)) # Numpy array'e çevirmek önemli
test_images = load_images(test_img_path)
test_labels = np.array(load_labels(test_label_path))

# Hu Momentleri Hesapla
train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

# Görüntüleri reshape edip momentleri çıkarma (load_images düz liste döndürebilir)
for train_idx, train_img in enumerate(train_images):
    # Görüntüyü 28x28 formatına getir (kütüphaneye göre değişebilir)
    img_reshaped = np.array(train_img).reshape(28, 28).astype(np.uint8)
    train_moments = cv2.moments(img_reshaped, True)
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    img_reshaped = np.array(test_img).reshape(28, 28).astype(np.uint8)
    test_moments = cv2.moments(img_reshaped, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# Normalizasyon
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# Etiketleri Düzenle (0 -> 0, Diğerleri -> 1)
train_labels[train_labels != 0] = 1
test_labels[test_labels != 0] = 1

# Model Oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[7], activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Eğitim
model.fit(train_huMoments, train_labels,
          batch_size=128, epochs=50,
          class_weight={0: 8, 1: 1}, # Veri dengesizliği için ağırlık
          verbose=1)

# Test ve Sonuç Gösterimi
perceptron_preds = model.predict(test_huMoments)
conf_matrix = confusion_matrix(test_labels, perceptron_preds > 0.5)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot()
cm_display.ax_.set_title("Single Neuron Classifier Confusion Matrix")
plt.show()

model.save("mnist_single_neuron.h5")
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

**Based on the Figure **(Confusion Matrix):

Overall Improvement: The strong diagonal line (high values like 829, 1102, 722) indicates that the Multilayer Perceptron (MLP) successfully learned non-linear relationships, significantly outperforming the single neuron approach.

Class-Specific Performance:

Best Performance (Digit 1): The digit "1" has the highest classification accuracy (1102 correct predictions). Its distinct geometric shape (a simple vertical line) produces a unique Hu Moment signature that is easily separable from other digits.

Weak Performance (Digits 2, 4, 5): The model struggles significantly with these digits. For example, the digit "2" was correctly classified only 364 times. It was frequently misclassified as "4" (211 times) or "5" (137 times), suggesting that their moment invariants are clustered very closely in the feature space.

Specific Confusion Clusters:

4 vs. 9: There is notable confusion between "4" and "9" (132 misclassifications). In handwriting, the closed upper loops of these digits often result in similar geometric moments.

Conclusion: While the hidden layers of the MLP allowed for non-linear decision boundaries, the accuracy is ultimately bottlenecked by the feature extraction method. Reducing a 784-pixel image to just 7 Hu Moments results in significant information loss, making it difficult to distinguish between geometrically similar digits (like 2, 4, and 5) regardless of the classifier's complexity.
### Source Code (Q2)

```python
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
```
