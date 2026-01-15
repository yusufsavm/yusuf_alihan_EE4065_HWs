# EE 4065 - Embedded Digital Image Processing: Homework 6

**Course:** EE 4065 - Embedded Digital Image Processing
**Assignment:** Homework 6

## Student Information

| Name | Student ID |
| :--- | :--- |
| Yusuf Oruç | 150720036 |
| Alihan Kocaakman | 150720065 |


### Results, Predicted Digits and their probabilities

| Model | Output |
| :--- | :--- |
| squeezenet | <img width="651" height="64" alt="squeeznet" src="https://github.com/user-attachments/assets/1a4b1368-5f94-4bec-a6c6-7780baeee092" /> |
| resnet | Got an error |
| mobilenet | <img width="654" height="64" alt="mobilenet" src="https://github.com/user-attachments/assets/9c1fd51d-71e2-4cc3-958d-b278b37ee73d" /> |
| efficientnet | Got an error |
| resnet | Got an error |


### Training Code

```python
import tensorflow as tf
import os

# Modellerin (Eğittiğin .h5 dosyalarının isimleri buraya)
model_names = ["mobilenet_mnist", "squeezenet_mnist", "efficientnet_mnist", "shufflenet_mnist","resnet_mnist"]

def manual_convert_tflite2cc(tflite_path, cc_path):
    """tflite2cc kütüphanesinin yaptığı işi manuel yapar."""
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    # Değişken ismini dosya adından türet (örn: mobilenet_mnist_tflite)
    var_name = os.path.basename(tflite_path).replace('.', '').replace('-', '')
    
    with open(cc_path, 'w') as f:
        f.write(f'// Model: {var_name}\n')
        f.write(f'unsigned char {var_name}[] = {{\n')
        
        # Byte'ları hex formatında yaz (0x00 formatında)
        for i, byte in enumerate(data):
            f.write(f' 0x{byte:02x},')
            if (i + 1) % 12 == 0:  # Her 12 byte'da bir alt satıra geç (okunabilirlik için)
                f.write('\n')
        
        f.write('\n};\n\n')
        f.write(f'unsigned int {var_name}_len = {len(data)};\n')

# Ana Döngü
for name in model_names:
    h5_path = f"{name}.h5"
    tflite_path = f"{name}.tflite"
    cc_path = f"{name}.cc"
    
    if os.path.exists(h5_path):
        print(f"İşleniyor: {h5_path}")
        
        # 1. .h5 -> .tflite Dönüşümü (Default Optimization ile) 
        model = tf.keras.models.load_model(h5_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        # 2. .tflite -> .cc Dönüşümü
        manual_convert_tflite2cc(tflite_path, cc_path)
        print(f"Tamamlandı: {cc_path}")
    else:
        print(f"Hata: {h5_path} bulunamadı! Lütfen dosya adını kontrol et.")
```
---

### quantization_mobilenet.py

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Veri Hazırla
print("Veri hazırlanıyor...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 32x32 RGB Formatı
train_images = tf.expand_dims(train_images, axis=-1)
train_images = tf.image.grayscale_to_rgb(train_images)
train_images = tf.image.resize(train_images, [32, 32]) / 255.0

# Test verisini de hazırla (Doğruluk testi için)
test_images = tf.expand_dims(test_images, axis=-1)
test_images = tf.image.grayscale_to_rgb(test_images)
test_images = tf.image.resize(test_images, [32, 32]) / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Veriyi biraz azaltalım (Hızlı dönüşüm için)
train_images_subset = train_images[:1000]

# 2. MOBILENET (Nano Alpha=0.1 - Hala MobileNet!)
def get_mobilenet():
    # alpha=0.1 ile en küçük MobileNetV2
    base = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
        alpha=0.1 
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    output = layers.Dense(10, activation='softmax')(x)
    return models.Model(base.input, output)

print("MobileNet Eğitiliyor...")
model = get_mobilenet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3, batch_size=32)

# ==========================================
# 3. QUANTIZATION (RAM KURTARICI ADIM)
# ==========================================
print("Quantization işlemi yapılıyor...")

# Dönüştürücü için örnek veri seti fonksiyonu (Representative Dataset)
# Bu fonksiyon, modelin sayı aralıklarını öğrenmesini sağlar
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images_subset).batch(1).take(100):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Bu satırlar modeli INT8 formatına zorlar (RAM 4 kat rahatlar)
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # Giriş verisi de byte olacak
converter.inference_output_type = tf.uint8 # Çıkış verisi de byte olacak

tflite_model_quant = converter.convert()

# Kaydet
dosya_adi = "mobilenet_int8.tflite"
with open(dosya_adi, "wb") as f:
    f.write(tflite_model_quant)

print(f"✅ {dosya_adi} oluşturuldu! Boyut: {len(tflite_model_quant)/1024:.2f} KB")
print("Bunu X-CUBE-AI'ya yükleyince RAM sorunu bitecek.")
```
---

### quantization_for_other_models.py

```python
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

# ==========================================
# 1. VERİ HAZIRLA (MNIST - 32x32 RGB)
# ==========================================
print("Veri hazırlanıyor...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Model girişine uygun hale getirme
train_images = tf.expand_dims(train_images, axis=-1)
train_images = tf.image.grayscale_to_rgb(train_images)
train_images = tf.image.resize(train_images, [32, 32]) / 255.0

test_images = tf.expand_dims(test_images, axis=-1)
test_images = tf.image.grayscale_to_rgb(test_images)
test_images = tf.image.resize(test_images, [32, 32]) / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Quantization için örnek veri seti
train_images_subset = train_images[:1000]

# ==========================================
# 2. MODEL MİMARİLERİ
# ==========================================

# --- A. MICRO SHUFFLENET V2 (DÜZELTİLMİŞ) ---
def channel_shuffle(x, groups):
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = channels // groups
    x = layers.Reshape((height, width, groups, channels_per_group))(x)
    x = layers.Permute((1, 2, 4, 3))(x)
    x = layers.Reshape((height, width, channels))(x)
    return x

def shuffle_block(x, in_channels, out_channels, stride):
    if stride == 2:
        x1 = layers.DepthwiseConv2D(3, strides=2, padding='same')(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Conv2D(out_channels // 2, 1)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        
        x2 = layers.Conv2D(out_channels // 2, 1)(x)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x2 = layers.DepthwiseConv2D(3, strides=2, padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Conv2D(out_channels // 2, 1)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x = layers.Concatenate()([x1, x2])
    else:
        channels = x.shape[-1]
        c_hat = channels // 2
        x1 = layers.Lambda(lambda z: z[..., :c_hat])(x)
        x2 = layers.Lambda(lambda z: z[..., c_hat:])(x)
        
        x2 = layers.Conv2D(out_channels // 2, 1)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x2 = layers.DepthwiseConv2D(3, padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Conv2D(out_channels // 2, 1)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x = layers.Concatenate()([x1, x2])

    return channel_shuffle(x, groups=2)

def get_micro_shufflenet():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(24, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    x = shuffle_block(x, 24, 48, stride=2) 
    x = shuffle_block(x, 48, 48, stride=1)
    x = shuffle_block(x, 48, 96, stride=2)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return models.Model(inputs, outputs, name="MicroShuffleNetV2")

# --- B. MICRO SQUEEZENET ---
def get_micro_squeezenet():
    def fire_module(x, s, e):
        squeeze = layers.Conv2D(s, (1, 1), activation='relu')(x)
        e1 = layers.Conv2D(e, (1, 1), activation='relu')(squeeze)
        e3 = layers.Conv2D(e, (3, 3), padding='same', activation='relu')(squeeze)
        return layers.Concatenate()([e1, e3])

    inputs = Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, 3, strides=2, activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    x = fire_module(x, 8, 16) 
    x = fire_module(x, 8, 16)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = fire_module(x, 16, 32)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return models.Model(inputs, outputs, name="MicroSqueezeNet")

# --- C. MICRO RESNET ---
def get_micro_resnet():
    def res_block(x, filters):
        shortcut = x
        if x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1)(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.Add()([x, shortcut])
        return layers.Activation('relu')(x)

    inputs = Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = res_block(x, 16)
    x = layers.MaxPooling2D(2)(x)
    x = res_block(x, 32)
    x = layers.MaxPooling2D(2)(x)
    x = res_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return models.Model(inputs, outputs, name="MicroResNet")

# --- D. MICRO EFFICIENTNET ---
def get_micro_efficientnet():
    # EfficientNet'in yapı taşı MBConv bloğunun basitleştirilmiş hali
    def mbconv_block(x_in, filters, expansion=2):
        # x_in: Bloğa giren veri (Shortcut için bunu saklıyoruz)
        input_ch = x_in.shape[-1]
        expanded_ch = input_ch * expansion
        
        x = x_in # İşlem yapılacak değişken
        
        # 1. Expand (Genişlet)
        if expansion != 1:
            x = layers.Conv2D(expanded_ch, 1, padding='same', activation='relu')(x)
            
        # 2. Depthwise Conv
        x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
        
        # 3. Project (Daralt)
        x = layers.Conv2D(filters, 1, padding='same')(x)
        
        # Residual bağlantı (Sadece giriş ve çıkış boyutları/kanalları aynıysa)
        # HATA BURADAYDI: 'inputs' yerine 'x_in' kullanmalıyız.
        if input_ch == filters:
            x = layers.Add()([x, x_in]) 
            
        return x

    inputs = Input(shape=(32, 32, 3))
    
    # Başlangıç
    x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)
    
    # Bloklar
    x = mbconv_block(x, 16, expansion=1)
    x = layers.MaxPooling2D(2)(x)
    
    x = mbconv_block(x, 32, expansion=4)
    x = layers.MaxPooling2D(2)(x)
    
    x = mbconv_block(x, 64, expansion=4)
    
    # Çıkış
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name="MicroEfficientNet")

# ==========================================
# 3. OTOMATİK ÇALIŞTIRMA DÖNGÜSÜ
# ==========================================

# Çalıştırmak istediğin tüm modelleri bu listeye ekle
model_listesi = [
    get_micro_shufflenet,
    get_micro_squeezenet,
    get_micro_resnet,
    get_micro_efficientnet
]

# Quantization Veri Üreteci
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images_subset).batch(1).take(100):
        yield [input_value]

print(f"\nToplam {len(model_listesi)} model eğitilecek ve dönüştürülecek...")
print("="*60)

for model_func in model_listesi:
    # 1. Modeli Oluştur
    model = model_func()
    print(f"\n>>> İŞLENİYOR: {model.name}")
    
    # 2. Compile ve Eğit
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Hız için 2 epoch yaptım. İstersen artır.
    model.fit(train_images, train_labels, epochs=2, batch_size=64, validation_split=0.1, verbose=1)
    
    # 3. TFLite Dönüştür (INT8)
    print(f"[{model.name}] TFLite dönüşümü yapılıyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # 4. Kaydet
    dosya_adi = f"{model.name}_int8.tflite"
    with open(dosya_adi, "wb") as f:
        f.write(tflite_model)
        
    boyut = len(tflite_model) / 1024
    print(f"✅ TAMAMLANDI: {dosya_adi} ({boyut:.2f} KB)")
    
print("\n" + "="*60)
print("TÜM İŞLEMLER BİTTİ! 4 Dosya da hazır.")
```
---


### main.c for both squeezenet and mobilenet

```c
/* USER CODE BEGIN Header */
/**
  **************************
  * @file           : main.c
  * @brief          : Main program body
  **************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ai_platform.h"
#include "network.h"       // Eğer ağ adını değiştirdiysen bu dosya adı da değişir
#include "network_data.h"
#include <stdio.h>         // printf için
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
ai_handle network = AI_HANDLE_NULL;
AI_ALIGNED(32) ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
AI_ALIGNED(32) ai_u8 in_data[AI_NETWORK_IN_1_SIZE_BYTES];
AI_ALIGNED(32) ai_u8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES];

ai_buffer *ai_input;
ai_buffer *ai_output;

// --- BURAYA EKLE (Global Yapıyoruz) ---
volatile int tahmin = 0;     // 'volatile' ekledik ki derleyici silmesin
volatile float guven = 0.0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int ai_init(void) {
    ai_error err;

    /* 1. Ağı Oluştur (Create) - GÜNCELLENDİ */
    err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        return -1;
    }

    /* 2. Parametreleri Hazırla */
    const ai_network_params params = {
        AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        AI_NETWORK_DATA_ACTIVATIONS(activations)
    };

    /* 3. Ağı Başlat (Init) - GÜNCELLENDİ */
    if (!ai_network_init(network, &params)) {
        return -1;
    }

    /* 4. Bufferları Ayarla */
    ai_input = ai_network_inputs_get(network, NULL);
    ai_output = ai_network_outputs_get(network, NULL);

    return 0;
}

int ai_run(void) {
    ai_i32 batch;

    // Buffer adreslerini güncelle
    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].data = AI_HANDLE_PTR(out_data);

    // Modeli çalıştır
    batch = ai_network_run(network, ai_input, ai_output);
    if (batch != 1) {
        return -1; // Hata
    }
    return 0;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */
  ai_init(); // Beyni hazırlar
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
    {
        // ---------------------------------------------------------
        // ADIM 1: Girişe "Gri" Renk Verelim (0 yerine 128)
        // ---------------------------------------------------------
        // Simsiyah ekran yerine gri verelim ki modelin çalıştığını görelim.
        for (int i=0; i < AI_NETWORK_IN_1_SIZE_BYTES; i++) {
            in_data[i] = 128; // Gri Piksel (Test için)
        }

        // ---------------------------------------------------------
        // ADIM 2: Modeli Çalıştır ve HATA KONTROLÜ Yap
        // ---------------------------------------------------------
        int hata_durumu = ai_run();
        // Live Watch'a 'hata_durumu' değişkenini de ekle!
        // Eğer bu değişken -1 ise model çalışmıyor demektir.

        if (hata_durumu == 0) {
            // Model Başarılı Çalıştıysa Sonucu Oku
            ai_u8 *predictions = out_data;

            int max_score = -1;
            tahmin = -1;

            // En yüksek puanı alan kutuyu bul
            for(int i=0; i<10; i++) {
                if(predictions[i] > max_score) {
                    max_score = predictions[i];
                    tahmin = i;
                }
            }

            // Skoru yüzdeye çevir
            guven = (float)max_score / 255.0f;
        }
        else {
            // Hata varsa tahmin'i -99 yap ki anlayalım
            tahmin = -99;
            guven = -1.0f;
        }

        // ---------------------------------------------------------
        // ADIM 3: Bekle
        // ---------------------------------------------------------
        HAL_Delay(100);

    /* USER CODE END WHILE */

  MX_X_CUBE_AI_Process();
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
```
---


