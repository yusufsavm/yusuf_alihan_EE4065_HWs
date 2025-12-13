# EE 4065 - Embedded Digital Image Processing: Homework 3

**Students:**
* Yusuf Oruç (150720036)
* Alihan Kocaakman (150720065)

---

## Question 1: Otsu's Thresholding

This section demonstrates the implementation of Otsu's thresholding method on an STM32 microcontroller. The image is sent from a PC via UART, processed on the MCU, and the result is sent back.

### Results

| Original Image (64x64) | Otsu Applied | Threshold Value Graph |
| :---: | :---: | :---: |
| ![Original](https://github.com/user-attachments/assets/b8f60f85-d335-4e86-b402-e4dd956ab626) | ![Otsu Result](https://github.com/user-attachments/assets/e39ebb74-aae2-4fee-901f-fec545625443) | ![Graph](https://github.com/user-attachments/assets/2fd43cc9-413c-4aaa-b739-5af9cb71575a) |

---

### STM32 C Code Implementation
The following code handles UART communication and the implementation of the Otsu algorithm in C.

```c
/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - Otsu Implementation
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define WIDTH  64   // Resim genisligi (Python koduyla ayni olmali)
#define HEIGHT 64   // Resim yuksekligi (Python koduyla ayni olmali)
#define IMG_SIZE (WIDTH * HEIGHT)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint8_t image_buffer[IMG_SIZE];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
#include <stdint.h>
#include <math.h>
#include "image.h"


// Otsu Eşik Değerini Bulan Fonksiyon
uint8_t calculate_otsu_threshold(uint8_t *image, int size) {
    int histogram[256] = {0};

    // 1. Histogram Oluşturma
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }

    float total_pixels = size;
    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * histogram[i];

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float varMax = 0;
    uint8_t threshold = 0;


    for (int t = 0; t < 256; t++) {
        wB += histogram[t];
        if (wB == 0) continue;
        wF = total_pixels - wB;
        if (wF == 0) break;

        sumB += (float)(t * histogram[t]);

        float mB = sumB / wB;             // Arka plan ortalaması
        float mF = (sum - sumB) / wF;     // Ön plan ortalaması

        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);


        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }
    return threshold;
}


void apply_threshold(uint8_t *image, int size, uint8_t threshold) {
    for(int i=0; i<size; i++){
        if(image[i] > threshold) image[i] = 255;
        else image[i] = 0;
    }
}
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
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
        // 1. Veriyi PC'den al
        // image_buffer artik tanimli oldugu icin hata vermeyecek
        HAL_UART_Receive(&huart2, image_buffer, IMG_SIZE, HAL_MAX_DELAY); // Timeout'u genislettim

        // 2. Otsu hesapla
        uint8_t best_thresh = calculate_otsu_threshold(image_buffer, IMG_SIZE);

        // 3. Eşiği uygula (Fonksiyon ismini duzelttim)
        apply_threshold(image_buffer, IMG_SIZE, best_thresh);

        // 4. Geri gönder
        HAL_UART_Transmit(&huart2, image_buffer, IMG_SIZE, HAL_MAX_DELAY);
        HAL_UART_Transmit(&huart2, &best_thresh, 1, HAL_MAX_DELAY);

      /* USER CODE END WHILE */

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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
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



Embeded Image Processing Homework 3
Yusuf Oruç 150720036
Alihan Kocaakman 150720065

Question 1
grayscaled mainimage for question 1 (64x64)
![mainimg](https://github.com/user-attachments/assets/b8f60f85-d335-4e86-b402-e4dd956ab626)
Otsu's tresholding applied
<img width="64" height="64" alt="otsu_sonuc" src="https://github.com/user-attachments/assets/e39ebb74-aae2-4fee-901f-fec545625443" />
threshold value
<img width="1068" height="142" alt="otsu_threshold" src="https://github.com/user-attachments/assets/2fd43cc9-413c-4aaa-b739-5af9cb71575a" />
STM32 C code part 
/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define WIDTH  64   // Resim genisligi (Python koduyla ayni olmali)
#define HEIGHT 64   // Resim yuksekligi (Python koduyla ayni olmali)
#define IMG_SIZE (WIDTH * HEIGHT)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint8_t image_buffer[IMG_SIZE];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
#include <stdint.h>
#include <math.h>
#include "image.h"


// Otsu Eşik Değerini Bulan Fonksiyon
uint8_t calculate_otsu_threshold(uint8_t *image, int size) {
    int histogram[256] = {0};

    // 1. Histogram Oluşturma
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }

    float total_pixels = size;
    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * histogram[i];

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float varMax = 0;
    uint8_t threshold = 0;


    for (int t = 0; t < 256; t++) {
        wB += histogram[t];
        if (wB == 0) continue;
        wF = total_pixels - wB;
        if (wF == 0) break;

        sumB += (float)(t * histogram[t]);

        float mB = sumB / wB;             // Arka plan ortalaması
        float mF = (sum - sumB) / wF;     // Ön plan ortalaması

        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);


        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }
    return threshold;
}


void apply_threshold(uint8_t *image, int size, uint8_t threshold) {
    for(int i=0; i<size; i++){
        if(image[i] > threshold) image[i] = 255;
        else image[i] = 0;
    }
}
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
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
        // 1. Veriyi PC'den al
        // image_buffer artik tanimli oldugu icin hata vermeyecek
        HAL_UART_Receive(&huart2, image_buffer, IMG_SIZE, HAL_MAX_DELAY); // Timeout'u genislettim

        // 2. Otsu hesapla
        uint8_t best_thresh = calculate_otsu_threshold(image_buffer, IMG_SIZE);

        // 3. Eşiği uygula (Fonksiyon ismini duzelttim)
        apply_threshold(image_buffer, IMG_SIZE, best_thresh);

        // 4. Geri gönder
        HAL_UART_Transmit(&huart2, image_buffer, IMG_SIZE, HAL_MAX_DELAY);
        HAL_UART_Transmit(&huart2, &best_thresh, 1, HAL_MAX_DELAY);

      /* USER CODE END WHILE */

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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
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

Python Code Part
import serial
import cv2
import numpy as np
import time

# --- AYARLAR ---
COM_PORT = 'COM8'      # STM32'nin bagli oldugu portu buraya yaz
BAUD_RATE = 115200     # STM32'deki huart2.Init.BaudRate ile ayni olmali
IMG_WIDTH = 64        # C kodundaki WIDTH ile ayni olmali
IMG_HEIGHT = 64       # C kodundaki HEIGHT ile ayni olmali
IMAGE_PATH = '.emb_hw\Include\mainimg.png' # Test edecegin resmin adi

try:
    # 1. Seri Portu Ac
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    print(f"Baglanti acildi: {COM_PORT}")
    time.sleep(2) # Baglantinin oturmasi icin bekleme

    # 2. Resmi Yukle ve Hazirla
    # Resmi gri tonlamali (grayscale) olarak oku
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("HATA: Resim bulunamadi! Dosya yolunu kontrol et.")
        exit()

    # Resmi STM32'nin bekledigi boyuta getir
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Resmi tek boyutlu byte dizisine cevir (Flatten)
    data_to_send = img_resized.flatten().tobytes()
    print(f"Gonderilen veri boyutu: {len(data_to_send)} bytes")

    # 3. Veriyi STM32'ye Gonder
    ser.write(data_to_send)
    print("Veri gonderildi, islenmesi bekleniyor...")

    # 4. Cevabi Bekle ve Oku
    # Gonderdigimiz kadar veriyi geri bekliyoruz
    received_data = ser.read(len(data_to_send))
    thresh_byte = ser.read(1)
    
    if thresh_byte:
        # Gelen byte verisini tam sayıya (integer) çevir
        thresh_val = int.from_bytes(thresh_byte, byteorder='little')
        print(f"STM32 Tarafindan Hesaplanan Otsu Esik Degeri: {thresh_val}")
    else:
        print("Uyari: Esik degeri okunamadi!)")

    if len(received_data) != len(data_to_send):
        print(f"HATA: Eksik veri geldi! Beklenen: {len(data_to_send)}, Gelen: {len(received_data)}")
        # Yine de ne geldigini gormek icin devam edebiliriz veya durabiliriz
    else:
        print("Veri basariyla alindi!")

    # 5. Gelen Veriyi Gorsele Cevir
    # Byte verisini numpy dizisine cevir
    processed_img = np.frombuffer(received_data, dtype=np.uint8)
    
    # Diziyi tekrar kare (resim) formatina sok
    processed_img = processed_img.reshape((IMG_HEIGHT, IMG_WIDTH))

    # 6. Sonuclari Ekrana Bas
    cv2.imshow("Orijinal (Kucultulmus)", img_resized)
    cv2.imshow("STM32 Islenmis (Otsu)", processed_img)

    kayit_adi = "otsu_sonuc.png"
    cv2.imwrite(kayit_adi, processed_img)
    print(f"Islenen resim '{kayit_adi}' olarak kaydedildi!")

    print("Cikmak icin bir tusa basin...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ser.close()

except serial.SerialException:
    print(f"HATA: {COM_PORT} portuna baglanilamadi. Portun baska programda acik olmadigindan emin ol.")
except Exception as e:
    print(f"Bir hata olustu: {e}")


QUestion 2
rgb main image for question 2 (64x64)
![resized2](https://github.com/user-attachments/assets/057656e9-9469-4b98-88cb-499f0715bcf0)
otsu's tresholding applied 
<img width="64" height="64" alt="renkli_otsu_sonuc" src="https://github.com/user-attachments/assets/486bda5c-740c-4bcc-a178-1002dd0bf751" />
threshold value
<img width="1111" height="97" alt="colorimgaeth" src="https://github.com/user-attachments/assets/8f4287b8-045b-40de-81b8-e69923bd7520" />

STM32 Code PArt
/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"  // <--- KANKA İŞTE EKSİK OLAN KRİTİK SATIR BU!
#include <stdint.h>
#include <math.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define WIDTH  64
#define HEIGHT 64
#define PIXEL_COUNT (WIDTH * HEIGHT)
#define RGB_SIZE (PIXEL_COUNT * 3) // 12288 bytes
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2; // main.h eklenince bu tip artik taninacak

/* USER CODE BEGIN PV */
uint8_t rgb_buffer[RGB_SIZE];
uint8_t gray_buffer[PIXEL_COUNT];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// RGB -> Gray Dönüşümü
void rgb_to_gray(uint8_t *rgb_img, uint8_t *gray_img, int pixel_count) {
    for (int i = 0; i < pixel_count; i++) {
        uint8_t r = rgb_img[i * 3];
        uint8_t g = rgb_img[i * 3 + 1];
        uint8_t b = rgb_img[i * 3 + 2];
        gray_img[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Otsu Hesaplama
uint8_t calculate_otsu_threshold(uint8_t *image, int size) {
    int histogram[256] = {0};
    for (int i = 0; i < size; i++) histogram[image[i]]++;

    float total = size;
    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * histogram[i];

    float sumB = 0;
    int wB = 0, wF = 0;
    float varMax = 0;
    uint8_t threshold = 0;

    for (int t = 0; t < 256; t++) {
        wB += histogram[t];
        if (wB == 0) continue;
        wF = total - wB;
        if (wF == 0) break;

        sumB += (float)(t * histogram[t]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }
    return threshold;
}

// Renkli resme eşik uygulama
void apply_threshold_to_rgb(uint8_t *rgb_img, uint8_t *gray_img, int count, uint8_t thresh) {
    for(int i = 0; i < count; i++) {
        if(gray_img[i] <= thresh) {
            rgb_img[i * 3]     = 0;
            rgb_img[i * 3 + 1] = 0;
            rgb_img[i * 3 + 2] = 0;
        }
    }
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* Infinite loop */
  while (1)
  {
        // 1. Renkli resmi al
        if (HAL_UART_Receive(&huart2, rgb_buffer, RGB_SIZE, HAL_MAX_DELAY) == HAL_OK)
        {
            // 2. Gri kopyasını oluştur
            rgb_to_gray(rgb_buffer, gray_buffer, PIXEL_COUNT);

            // 3. Eşik değerini bul
            uint8_t best_thresh = calculate_otsu_threshold(gray_buffer, PIXEL_COUNT);

            // 4. Eşiği uygula
            apply_threshold_to_rgb(rgb_buffer, gray_buffer, PIXEL_COUNT, best_thresh);

            // 5. Geri gönder
            HAL_UART_Transmit(&huart2, rgb_buffer, RGB_SIZE, HAL_MAX_DELAY);

            // 6. Eşik değerini gönder
            HAL_UART_Transmit(&huart2, &best_thresh, 1, HAL_MAX_DELAY);
        }
  }
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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
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
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

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
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

Python Code PArt
import serial
import cv2
import numpy as np
import time

# --- AYARLAR ---
COM_PORT = 'COM8'     
BAUD_RATE = 115200 
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMAGE_PATH = 'resized2.jpg' 

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=5)
    print(f"Baglanti: {COM_PORT}")
    time.sleep(2)

    # 1. Resmi Renkli Oku
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # BGR -> RGB Dönüşümü (STM32'de renklerin doğru görünmesi için)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 2. Gönder
    data_to_send = img_rgb.flatten().tobytes()
    ser.write(data_to_send)
    print("Veri gonderildi...")

    # 3. Cevabı Bekle (ARTIK RENKLİ BEKLİYORUZ -> 64*64*3 byte)
    expected_size = IMG_WIDTH * IMG_HEIGHT * 3 
    received_data = ser.read(expected_size)
    thresh_byte = ser.read(1)
    
    if thresh_byte:
        val = int.from_bytes(thresh_byte, 'little')
        print(f"Otsu Threshold: {val}")

    if len(received_data) == expected_size:
        # 4. Gelen veriyi işle
        processed_img = np.frombuffer(received_data, dtype=np.uint8)
        
        # Reshape yaparken artık (64, 64, 3) yapıyoruz
        processed_img = processed_img.reshape((IMG_HEIGHT, IMG_WIDTH, 3))
        
        # OpenCV göstermek için BGR ister, RGB -> BGR yapalım
        processed_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

        # 5. Göster
        cv2.imshow("Orijinal", img_resized)
        cv2.imshow("STM32 Renkli Otsu", processed_bgr)
        
        cv2.imwrite("renkli_otsu_sonuc.png", processed_bgr)
        print("Kaydedildi.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Hata: Eksik veri geldi.")
    
    ser.close()

except Exception as e:
    print(f"Hata: {e}")
    
Question 3
dilation applied 
<img width="64" height="64" alt="dilation_morfoloji" src="https://github.com/user-attachments/assets/466e3233-9d24-4f63-a920-71076ebf97ba" />
erosion applied
<img width="64" height="64" alt="erode_morfoloji" src="https://github.com/user-attachments/assets/0ee6b145-b0c7-41b5-9e50-cd14ac901e4b" />
oppenşng applied
<img width="64" height="64" alt="openin_morfoloji" src="https://github.com/user-attachments/assets/d0c5927f-0896-404a-8b2b-5cfab6a77372" />
closing applied
<img width="64" height="64" alt="closing2_morfoloji" src="https://github.com/user-attachments/assets/76d39abf-78e2-4f98-b561-c310b83eafd6" />

STM32 Code Part
/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body (Only Q3 - Morphology)
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdint.h>
#include <string.h> // memcpy icin

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define WIDTH  64
#define HEIGHT 64
#define IMG_SIZE (WIDTH * HEIGHT)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint8_t img_buffer[IMG_SIZE];   // Gelen ve giden resim
uint8_t temp_buffer[IMG_SIZE];  // İşlem sırasındaki yedek alan
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void MX_GPIO_Init(void);          // static kaldirildi
void MX_USART2_UART_Init(void);   // static kaldirildi
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// --- MORFOLOJİK OPERASYONLAR ---

// 1. EROSION (Aşındırma): 3x3 alanda bir tane bile SİYAH (0) varsa, merkez SİYAH olur.
void morphology_erode(uint8_t *input, uint8_t *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Kenarları koruma (siyah bırak)
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                output[y * width + x] = 0;
                continue;
            }

            uint8_t min_val = 255;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    // Komşuya bak
                    int pixel_val = input[(y + ky) * width + (x + kx)];
                    // Eğer komşu siyahsa (0), biz de siyah olacağız
                    if (pixel_val == 0) min_val = 0;
                }
            }
            output[y * width + x] = min_val;
        }
    }
}

// 2. DILATION (Yayma): 3x3 alanda bir tane bile BEYAZ (255) varsa, merkez BEYAZ olur.
void morphology_dilate(uint8_t *input, uint8_t *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Kenar kontrolü
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                output[y * width + x] = 0;
                continue;
            }

            uint8_t max_val = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel_val = input[(y + ky) * width + (x + kx)];
                    // Eğer komşu beyazsa, biz de beyaz olacağız
                    if (pixel_val == 255) max_val = 255;
                }
            }
            output[y * width + x] = max_val;
        }
    }
}

// 3. OPENING (Açma): Önce Erode, Sonra Dilate
// Gürültüyü (küçük beyaz noktaları) temizler.
void morphology_opening(uint8_t *img, uint8_t *temp, int width, int height) {
    // 1. Adım: Erode yap -> Sonuç temp'e yazılır
    morphology_erode(img, temp, width, height);

    // 2. Adım: Temp'i al, Dilate yap -> Sonuç tekrar img'ye (ana buffera) yazılır
    morphology_dilate(temp, img, width, height);
}

// 4. CLOSING (Kapama): Önce Dilate, Sonra Erode
// Nesne içindeki delikleri kapatır.
void morphology_closing(uint8_t *img, uint8_t *temp, int width, int height) {
    // 1. Adım: Dilate yap -> Sonuç temp'e
    morphology_dilate(img, temp, width, height);

    // 2. Adım: Temp'i al, Erode yap -> Sonuç img'ye
    morphology_erode(temp, img, width, height);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* Infinite loop */
  while (1)
  {
        // 1. Resmi Al (4096 Byte - Binary/Grayscale)
        if (HAL_UART_Receive(&huart2, img_buffer, IMG_SIZE, HAL_MAX_DELAY) == HAL_OK)
        {
            // Ödevde "Use binary image" dediği için gelen veriyi netleştirelim.
            // Eğer gri tonlu geldiyse >127 olanları 255 (Beyaz), altını 0 (Siyah) yapalım.
            for(int i=0; i<IMG_SIZE; i++) {
                if(img_buffer[i] > 140) img_buffer[i] = 255;
                else img_buffer[i] = 0;
            }

            // 2. Morfolojik İşlemi Uygula
            // Şuan "Opening" aktif. Diğerlerini denemek için yorum satırlarını değiştirebilirsin.
            morphology_opening(img_buffer, temp_buffer, WIDTH, HEIGHT);

            // Alternatifler:
           // morphology_closing(img_buffer, temp_buffer, WIDTH, HEIGHT);

            //morphology_erode(img_buffer, temp_buffer, WIDTH, HEIGHT);
           // memcpy(img_buffer, temp_buffer, IMG_SIZE);

           // morphology_dilate(img_buffer, temp_buffer, WIDTH, HEIGHT);
           // memcpy(img_buffer, temp_buffer, IMG_SIZE);

            // 3. Sonucu Geri Gönder
            HAL_UART_Transmit(&huart2, img_buffer, IMG_SIZE, HAL_MAX_DELAY);
        }
  }
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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
void MX_USART2_UART_Init(void)
{
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
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

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
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

Python Code Part
import serial
import cv2
import numpy as np
import time

# --- AYARLAR (Burayı Kendi Bilgisayarına Göre Düzenle) ---
COM_PORT = 'COM8'       # Aygıt Yöneticisi'nden STM32'nin portuna bak
BAUD_RATE = 115200      # STM32 kodundaki ile aynı olmalı
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMAGE_PATH = '.emb_hw\Include\mainimg.png' # İşlenecek resmin adı

try:
    # 1. Seri Port Bağlantısını Başlat
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=5)
    print(f"Bağlantı sağlandı: {COM_PORT}")
    time.sleep(2) # Bağlantının oturması için kısa bekleme

    # 2. Resmi Yükle ve Hazırla
    # Q3 için sadece siyah-beyaz/gri veriye ihtiyacımız var, o yüzden GRAYSCALE okuyoruz.
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("HATA: Resim dosyası bulunamadı!")
        exit()

    # Resmi 64x64 boyutuna getir
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Opsiyonel: Göndermeden önce Python'da binary yapıp gönderebiliriz
    # ama STM32 kodu zaten gelen veriyi binary'ye çeviriyor, o yüzden direkt yolluyoruz.
    
    # 3. Veriyi Gönder (4096 Byte)
    data_to_send = img_resized.flatten().tobytes()
    ser.write(data_to_send)
    print(f"Veri gönderildi ({len(data_to_send)} byte)... İşlem bekleniyor.")

    # 4. Cevabı Bekle
    # STM32 sadece işlenmiş resmi (4096 byte) geri yolluyor.
    expected_size = IMG_WIDTH * IMG_HEIGHT
    received_data = ser.read(expected_size)

    if len(received_data) == expected_size:
        print("Sonuç başarıyla alındı!")

        # 5. Gelen Veriyi Görsele Çevir
        processed_img = np.frombuffer(received_data, dtype=np.uint8)
        processed_img = processed_img.reshape((IMG_HEIGHT, IMG_WIDTH))

        # 6. Sonuçları Göster
        # Orijinal vs İşlenmiş
        cv2.imshow("Orijinal (64x64)", img_resized)
        cv2.imshow("STM32 Morfoloji Sonucu", processed_img)

        # Sonucu kaydet
        cv2.imwrite("openin_morfoloji.png", processed_img)
        print("İşlenmiş resim 'morfoloji_sonuc.png' olarak kaydedildi.")

        print("Çıkış için bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print(f"HATA: Eksik veri geldi veya zaman aşımı! Beklenen: {expected_size}, Gelen: {len(received_data)}")

    ser.close()

except serial.SerialException:
    print(f"HATA: {COM_PORT} portuna bağlanılamadı. Portu kontrol et veya kapatıp aç.")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")



