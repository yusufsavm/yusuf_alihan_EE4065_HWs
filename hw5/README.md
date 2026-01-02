question 1

result
Predicted Digit and its probability <img width="499" height="187" alt="Ekran görüntüsü 2026-01-01 230803" src="https://github.com/user-attachments/assets/0eaa6c0b-fbb7-48e6-9928-45bcb72d36db" />

training code

# Dosya adı: train_keyword_spotting.py
import os
import numpy as np
import scipy.signal as sig
from mfcc_func import create_mfcc_features # Az önce oluşturduğumuz dosya
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Veri seti klasörü
RECORDINGS_DIR = "recordings"

# Klasör kontrolü
if not os.path.exists(RECORDINGS_DIR):
    print("HATA: 'recordings' klasörü bulunamadı. Lütfen oluşturup içine .wav dosyalarını atın.")
    exit()

# Tüm .wav dosyalarını listele
recordings_list = [(RECORDINGS_DIR, f) for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav")]

# Parametreler (STM32 tarafında C kodu ile uyumlu olmalı)
FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13  # MFCC katsayı sayısı

# Test ve Eğitim setlerini ayırma (Kitaptaki gibi 'yweweler' test için ayrılıyor)
test_list = {record for record in recordings_list if "yweweler" in record[1]}
train_list = set(recordings_list) - test_list

print("Eğitim verisi hazırlanıyor (Bu işlem biraz sürebilir)...")
train_mfcc_features, train_labels = create_mfcc_features(list(train_list), FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs)

print("Test verisi hazırlanıyor...")
test_mfcc_features, test_labels = create_mfcc_features(list(test_list), FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs)

# Model Mimarisi [cite: 11]
# Giriş katmanı: 26 nöron (13 MFCC + 13 Delta)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(26,)), 
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax") # 0-9 arası rakamlar için çıkış
])

# Etiketleri One-Hot formatına çevirme
ohe = OneHotEncoder()
train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1)).toarray()

# Modeli Derleme
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
              optimizer=tf.keras.optimizers.Adam(1e-3), 
              metrics=['accuracy'])

# Eğitimi Başlat
model.fit(train_mfcc_features, train_labels_ohe, epochs=100, verbose=1)

# Test Sonuçlarını Göster
nn_preds = model.predict(test_mfcc_features)
predicted_classes = np.argmax(nn_preds, axis=1)

# Confusion Matrix
categories = np.unique(test_labels)
conf_matrix = confusion_matrix(test_labels, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
cm_display.plot()
plt.title("Neural Network Confusion Matrix")
plt.show()

# MODELİ KAYDET (En Önemli Adım)
model.save("mlp_fsdd_model.h5")
print("Model başarıyla 'mlp_fsdd_model.h5' olarak kaydedildi.")


conversion file from h5 to tflite

import tensorflow as tf
import os

# Model dosya yolunu kontrol et
model_path = "mlp_fsdd_model.h5"

if not os.path.exists(model_path):
    print("HATA: .h5 dosyası bulunamadı!")
    exit()

try:
    # 1. Mevcut modeli yükle
    model = tf.keras.models.load_model(model_path)
    
    # 2. TFLite Dönüştürücü oluştur
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # (Opsiyonel) STM32 için optimizasyonları açar
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    
    # 3. Dönüştür
    tflite_model = converter.convert()

    # 4. Kaydet (.tflite olarak)
    tflite_path = "mlp_fsdd_model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Başarılı! Model '{tflite_path}' olarak kaydedildi.")
    print("Şimdi STM32CubeMX'te bu .tflite dosyasını seç.")

except Exception as e:
    print(f"Dönüştürme Hatası: {e}")


tflite to test_input_h

import os

# Dosya isimlerini tanımla
tflite_path = 'mlp_fsdd_model.tflite'
output_header_path = 'model_data.h'
array_name = 'mlp_fsdd_model_tflite'

# Binary dosyayı oku
with open(tflite_path, 'rb') as f:
    data = f.read()

# C header dosyasını yaz
with open(output_header_path, 'w') as f:
    f.write(f'// Bu dosya Python scripti ile otomatik oluşturuldu.\n\n')
    f.write(f'const unsigned char {array_name}[] = {{\n')
    
    for i, byte in enumerate(data):
        f.write(f'0x{byte:02x}, ')
        if (i + 1) % 12 == 0: # Okunabilirlik için her 12 byteda bir alt satıra geç
            f.write('\n')
            
    f.write('};\n\n')
    f.write(f'const unsigned int {array_name}_len = {len(data)};\n')

print(f"{output_header_path} başarıyla oluşturuldu!")


app_c-cube-ai.c


/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "hw5_last.h"
#include "hw5_last_data.h"

/* USER CODE BEGIN includes */
 /* USER CODE BEGIN Includes */
 #include <test_input_audio.h>  // Python ile oluşturduğumuz MFCC verisi
 #include <stdio.h>       // Gerekirse printf için

 /* USER CODE BEGIN 0 */
 // Debug için Global Değişkenler
 float max_prob = 0.0f;
 int predicted_digit = -1;
 /* USER CODE END 0 */
 /* USER CODE END Includes */
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_HW5_LAST_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_HW5_LAST_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_HW5_LAST_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_HW5_LAST_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_HW5_LAST_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_HW5_LAST_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_HW5_LAST_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_HW5_LAST_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_HW5_LAST_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle hw5_last = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  /*if (fct)
   printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);
*/
  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_hw5_last_create_and_init(&hw5_last, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_hw5_last_create_and_init");
    return -1;
  }

  ai_input = ai_hw5_last_inputs_get(hw5_last, NULL);
  ai_output = ai_hw5_last_outputs_get(hw5_last, NULL);

#if defined(AI_HW5_LAST_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_HW5_LAST_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_HW5_LAST_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_HW5_LAST_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_HW5_LAST_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_HW5_LAST_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_hw5_last_run(hw5_last, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_hw5_last_get_error(hw5_last),
        "ai_hw5_last_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
int acquire_and_process_data(ai_i8* data[])
{
  /* fill the inputs of the c-model
  for (int idx=0; idx < AI_HW5_LAST_IN_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}

int post_process(ai_i8* data[])
{
  /* process the predictions
  for (int idx=0; idx < AI_HW5_LAST_OUT_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  //printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  int res = -1;

  //printf("TEMPLATE - run - main loop\r\n");

  if (hw5_last) {

      // --- BİZİM EKLEDİĞİMİZ KISIM BAŞLIYOR ---

      // 1. Buffer Erişimleri
      ai_float *in_data = (ai_float *)ai_input[0].data;
      ai_float *out_data = (ai_float *)ai_output[0].data;

      // 2. Test Verisini Yükle
      //printf("Test verisi yukleniyor...\r\n");
      for (int i = 0; i < 26; i++) {
          in_data[i] = (ai_float)test_input_mfcc[i];
      }

      // 3. Modeli Çalıştır
      //printf("Model calistiriliyor (Inference)...\r\n");
      res = ai_run();

      // 4. Sonucu Analiz Et
      if (res == 0) {
          // NOT: Buradaki "float" ve "int" kelimelerini sildik çünkü
          // yukarıda global olarak tanımladık. Sadece değerlerini sıfırlıyoruz.
          max_prob = 0.0f;
          predicted_digit = -1;

          for (int i = 0; i < 10; i++) {
              if (out_data[i] > max_prob) {
                  max_prob = out_data[i];
                  predicted_digit = i;
              }
          }

          // Konsol Çıktıları
         // printf("--------------------------------\r\n");
         // printf("TAHMIN EDILEN RAKAM: %d\r\n", predicted_digit);
         // printf("GUVEN ORANI: %f\r\n", max_prob);
         // printf("--------------------------------\r\n");
      }
      else {
         // printf("Hata kodu: %d\r\n", res);
      }

      HAL_Delay(2000);
      // --- BİZİM KISIM BİTTİ ---
  }

  if (res) {
    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
    ai_log_err(err, "Process has FAILED");
  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif


create_real_input.py

import numpy as np
import os
# Senin eğitimde kullandığın fonksiyonu çağırıyoruz
from mfcc_func import create_mfcc_features 

# --- AYARLAR ---
# Test etmek istediğin dosyanın tam adı (Klasörde bu dosya olmalı!)
# Model 0-9 arası rakamları bildiği için "0" ile başlayan bir dosya seçtik.
TEST_FILE_DIR = "recordings"
TEST_FILE_NAME = "0_jackson_0.wav" 

# Eğitimdeki parametrelerin AYNISI olmalı (Listing 11.5'ten alındı)
FFT_SIZE = 1024
SAMPLE_RATE = 8000
NUM_OF_MEL_FILTERS = 20
NUM_OF_DCT_OUTPUTS = 13

def generate_header():
    file_path = os.path.join(TEST_FILE_DIR, TEST_FILE_NAME)
    
    if not os.path.exists(file_path):
        print(f"HATA: '{file_path}' dosyasi bulunamadi! Lutfen dosya yolunu kontrol et.")
        return

    print(f"Islem basliyor: {TEST_FILE_NAME} dosyasi isleniyor...")

    # create_mfcc_features fonksiyonu genelde bir liste bekler.
    # Bizim elimizde tek dosya var, o yüzden onu tek elemanlı bir liste gibi gösteriyoruz.
    # Yapı: [(KlasörYolu, DosyaAdi)]
    single_file_list = [(TEST_FILE_DIR, TEST_FILE_NAME)]

    # Fonksiyonu çağırarak MFCC özelliklerini çıkarıyoruz
    # Bu fonksiyon hem özellikleri (features) hem etiketleri (labels) döndürür.
    features, labels = create_mfcc_features(
        single_file_list, 
        FFT_SIZE, 
        SAMPLE_RATE, 
        NUM_OF_MEL_FILTERS, 
        NUM_OF_DCT_OUTPUTS
    )

    # features değişkeni muhtemelen [[...26 değer...]] şeklinde bir listedir.
    # İlk (ve tek) elemanı alıyoruz.
    real_mfcc_input = features[0]

    # --- KONTROL ---
    print(f"Elde edilen ozellik boyutu: {len(real_mfcc_input)}")
    if len(real_mfcc_input) != 26:
        print("HATA: Model 26 giris bekliyor ama fonksiyon farkli sayida uretti!")
        return

    # --- C HEADER DOSYASINI YAZMA ---
    header_content = f"""
// Bu dosya create_real_input.py tarafindan olusturuldu.
// Kaynak ses dosyasi: {TEST_FILE_NAME}
// Beklenen Tahmin: 0 (Sifir)

#ifndef TEST_INPUT_H
#define TEST_INPUT_H

// Yapay zeka modelinin girisine kopyalanacak 26 adet MFCC degeri:
const float test_input_mfcc[26] = {{
"""
    
    # Sayıları virgüle ayırarak yaz
    count = 0
    for val in real_mfcc_input:
        header_content += f"    {val:.6f}f, "
        count += 1
        if count % 5 == 0: # Okunabilirlik için satır atla
            header_content += "\n"

    header_content += "\n};\n\n#endif // TEST_INPUT_H"

    with open("test_input.h", "w") as f:
        f.write(header_content)

    print("-" * 30)
    print("BASARILI! 'test_input.h' dosyasi olusturuldu.")
    print("Simdi bu dosyayi STM32 projesindeki Core/Inc klasorune kopyala.")
    print("-" * 30)

if __name__ == "__main__":
    generate_header()






question 2

result
<img width="504" height="239" alt="hw5_q2" src="https://github.com/user-attachments/assets/e6d36028-9f3f-46c4-b318-325310b4e2b0" />


training code

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


h5_to_tflite

import tensorflow as tf

# 1. Modeli yükle
model_path = 'mlp_mnist_model.h5'
print(f"Model yukleniyor: {model_path}...")
model = tf.keras.models.load_model(model_path)

# Modelin giriş boyutunu kontrol edelim (Bunu bir sonraki adımda kullanacağız)
input_shape = model.input_shape
print(f"Model Giris Boyutu: {input_shape}")

# 2. TFLite dönüştürücüyü hazırla
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Modeli dönüştür
tflite_model = converter.convert()

# 4. Kaydet
output_path = 'q2_mlp_mnist.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"BASARILI! '{output_path}' dosyasi olusturuldu.")


h5_to_c

import os
import numpy as np
import cv2
import struct
import tensorflow as tf
from tensorflow import keras

# --- AYARLAR ---
MODEL_PATH = "mlp_mnist_model.h5"     # Elindeki model dosyası
OUTPUT_HEADER = "hw5_q2_data.h"        # Çıktı dosyası
DATASET_PATH = "."                    # MNIST dosyalarının olduğu klasör (aynı dizindeyse nokta)

# --- 1. VERİ YÜKLEME FONKSİYONLARI ---
# (Mean ve Std hesaplamak için veriye ihtiyacımız var)
def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def extract_hu_moments(images):
    print(f"   -> {len(images)} resimden Hu Momentler çıkarılıyor...")
    hu_list = np.empty((len(images), 7))
    for idx, img in enumerate(images):
        moments = cv2.moments(img, True)
        hu_list[idx] = cv2.HuMoments(moments).reshape(7)
    return hu_list

# --- 2. ANA İŞLEM ---
def export_model_to_c():
    print("1. MNIST Eğitim verisi yükleniyor (Mean/Std hesaplamak için)...")
    train_img_path = os.path.join(DATASET_PATH, "train-images.idx3-ubyte")
    
    if not os.path.exists(train_img_path):
        print(f"HATA: {train_img_path} bulunamadı! Mean/Std hesaplanamıyor.")
        return

    train_images = load_images(train_img_path)
    
    # Hu Momentleri hesapla
    train_huMoments = extract_hu_moments(train_images)
    
    # Mean ve Std Hesapla (Eğitimdeki aynı işlemi tekrarlıyoruz)
    features_mean = np.mean(train_huMoments, axis=0)
    features_std = np.std(train_huMoments, axis=0)
    
    print("\n2. Normalizasyon Parametreleri Hesaplandı:")
    print(f"   Mean: {features_mean}")
    print(f"   Std : {features_std}")

    # --- MODELİ YÜKLE ---
    print(f"\n3. Model Yükleniyor: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("HATA: .h5 dosyası bulunamadı!")
        return
        
    model = keras.models.load_model(MODEL_PATH)
    print("   Model başarıyla yüklendi.")

    # --- C HEADER OLUŞTUR ---
    print(f"\n4. C Header dosyası oluşturuluyor: {OUTPUT_HEADER}")
    with open(OUTPUT_HEADER, "w") as f:
        f.write("/* STM32 Neural Network Model Data */\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        
        # A) Normalizasyon Değerlerini Yaz
        f.write("// --- Normalization Parameters (Mean & Std) ---\n")
        f.write(f"const float features_mean[7] = {{\n")
        for val in features_mean:
            f.write(f"    {val:.6f}f, \n")
        f.write("};\n\n")

        f.write(f"const float features_std[7] = {{\n")
        for val in features_std:
            f.write(f"    {val:.6f}f, \n")
        f.write("};\n\n")

        # B) Ağırlıklar ve Biaslar
        f.write("// --- Model Weights & Biases ---\n")
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if len(weights) > 0: # Sadece ağırlığı olan katmanlar (Dense)
                w, b = weights
                
                # Weights (Flatten edilmiş)
                f.write(f"// Layer {i} ({layer.name}) Weights - Shape: {w.shape}\n")
                f.write(f"const float layer{i}_weights[] = {{\n")
                for val in w.flatten():
                    f.write(f"{val:.6f}f, ")
                f.write("};\n\n")
                
                # Biases
                f.write(f"// Layer {i} ({layer.name}) Biases - Shape: {b.shape}\n")
                f.write(f"const float layer{i}_biases[] = {{\n")
                for val in b.flatten():
                    f.write(f"{val:.6f}f, ")
                f.write("};\n\n")
        
        f.write("#endif // MODEL_DATA_H\n")
    
    print(f"\nBAŞARILI! '{OUTPUT_HEADER}' dosyası oluşturuldu.")
    print("Bu dosyayı STM32 projenin 'Core/Inc' klasörüne kopyalayabilirsin.")

if __name__ == "__main__":
    export_model_to_c()


test_data_generate

import os
import numpy as np
import cv2
import struct
import random

# --- DOSYA YOLLARI ---
# Eğer dosyalar aynı klasördeyse "." kalsın
DATASET_PATH = "." 
OUTPUT_HEADER = "hw5_q2_test_data.h"
NUM_SAMPLES = 20  # Kaç tane test verisi gömmek istiyorsun?

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

def create_test_header():
    print("1. MNIST Test verileri yükleniyor...")
    img_path = os.path.join(DATASET_PATH, "t10k-images.idx3-ubyte")
    lbl_path = os.path.join(DATASET_PATH, "t10k-labels.idx1-ubyte")
    
    if not os.path.exists(img_path):
        print(f"HATA: {img_path} bulunamadı!")
        return

    test_images = load_images(img_path)
    test_labels = load_labels(lbl_path)
    
    # Rastgele seçim yapmayalım, her seferinde aynılarını alalım ki kıyaslaması kolay olsun (ilk N tanesi)
    # İstersen random.sample kullanabilirsin ama sıralı gitmek debug için iyidir.
    indices = range(NUM_SAMPLES) 
    
    print(f"2. İlk {NUM_SAMPLES} verinin Hu Momentleri hesaplanıyor...")
    
    with open(OUTPUT_HEADER, "w") as f:
        f.write(f"#ifndef TEST_DATA_H\n#define TEST_DATA_H\n\n")
        f.write(f"// MNIST Test Setinden ilk {NUM_SAMPLES} ornek\n")
        f.write(f"#define NUM_TEST_SAMPLES {NUM_SAMPLES}\n\n")
        
        # 1. Labels (Gerçek Cevaplar)
        f.write(f"const int test_labels[{NUM_SAMPLES}] = {{\n    ")
        for i in indices:
            f.write(f"{test_labels[i]}, ")
        f.write("\n};\n\n")
        
        # 2. Inputs (Raw Hu Moments)
        f.write(f"const float test_samples[{NUM_SAMPLES}][7] = {{\n")
        
        for i in indices:
            img = test_images[i]
            moments = cv2.moments(img, True)
            hu = cv2.HuMoments(moments).reshape(7)
            
            f.write("    {")
            for val in hu:
                f.write(f"{val:.6f}f, ")
            f.write("}, \n")
            
        f.write("};\n\n")
        f.write("#endif // TEST_DATA_H\n")
        
    print(f"BAŞARILI! '{OUTPUT_HEADER}' oluşturuldu.")
    print("Bu dosyayı 'hw5_q2_data.h' yanına (Core/Inc klasörüne) kopyala.")

if __name__ == "__main__":
    create_test_header()


app_x-cube-ai.c


/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "hw5_q2_network.h"
#include "hw5_q2_network_data.h"

/* USER CODE BEGIN includes */
#include "test_input_image.h"
 float max_prob = 0.0f;
 int predicted_digit = -1;
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_HW5_Q2_NETWORK_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_HW5_Q2_NETWORK_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_HW5_Q2_NETWORK_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_HW5_Q2_NETWORK_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_HW5_Q2_NETWORK_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_HW5_Q2_NETWORK_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_HW5_Q2_NETWORK_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_HW5_Q2_NETWORK_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_HW5_Q2_NETWORK_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle hw5_q2_network = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_hw5_q2_network_create_and_init(&hw5_q2_network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_hw5_q2_network_create_and_init");
    return -1;
  }

  ai_input = ai_hw5_q2_network_inputs_get(hw5_q2_network, NULL);
  ai_output = ai_hw5_q2_network_outputs_get(hw5_q2_network, NULL);

#if defined(AI_HW5_Q2_NETWORK_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_HW5_Q2_NETWORK_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_HW5_Q2_NETWORK_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_HW5_Q2_NETWORK_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_HW5_Q2_NETWORK_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_HW5_Q2_NETWORK_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_hw5_q2_network_run(hw5_q2_network, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_hw5_q2_network_get_error(hw5_q2_network),
        "ai_hw5_q2_network_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
int acquire_and_process_data(ai_i8* data[])
{
  /* fill the inputs of the c-model
  for (int idx=0; idx < AI_HW5_Q2_NETWORK_IN_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}

int post_process(ai_i8* data[])
{
  /* process the predictions
  for (int idx=0; idx < AI_HW5_Q2_NETWORK_OUT_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
	    int res = -1;

	    // 1. Model Hazır mı Kontrol Et
	    if (hw5_q2_network) { // (Senin model degiskenin farkliysa duzelt, genelde hw5_last veya network olur)

	        // Giris ve Cikis Hafiza Adreslerini Al
	        ai_float *in_data = (ai_float *)ai_input[0].data;
	        ai_float *out_data = (ai_float *)ai_output[0].data;

	        // ---------------------------------------------
	        // SORU 2: MNIST RAKAM TANIMA (Giris: 7 Hu Moment)
	        // ---------------------------------------------

	        // 2. Veriyi Yukle (test_input.h dosyasindan)
	        // DİKKAT: Döngü sayısı artık 7!
	        for (int i = 0; i < 7; i++) {
	            in_data[i] = (ai_float)test_input_image[i];
	        }

	        // 3. Modeli Calistir
	        res = ai_run();

	        // 4. Sonucu Analiz Et (En yuksek olasilik hangisi?)
	        if (res == 0) {

	            // Global degiskenleri sifirla (Debug ekrani icin)
	            max_prob = 0.0f;
	            predicted_digit = -1;

	            // 0'dan 9'a kadar tum cikislara bak
	            for (int i = 0; i < 10; i++) {
	                if (out_data[i] > max_prob) {
	                    max_prob = out_data[i];
	                    predicted_digit = i;
	                }
	            }
	        }

	        // Islemciyi yormamak icin bekle
	        HAL_Delay(2000);
	    }

	    if (res) {
	        ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
	        ai_log_err(err, "Process has FAILED");
	    }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif


