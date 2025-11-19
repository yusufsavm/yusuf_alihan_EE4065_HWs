## EE 4065 - Embedded Digital Image Processing: Homework 2
---



| Name | Student ID |
| :--- | :--- |
| Yusuf Oruç | 150720036 |
| Alihan Kocaakman | 150720065 |

---



![Language](https://img.shields.io/badge/language-C-blue.svg) ![Platform](https://img.shields.io/badge/platform-STM32-green.svg) ![IDE](https://img.shields.io/badge/IDE-STM32CubeIDE-blueviolet.svg)

## 📂 Project Overview

* **Input:** 64x64 Grayscale Image (Generated as C Header)
* **Core Features:**
    * Histogram Calculation & Equalization
    * Low Pass Filtering (Smoothing)
    * High Pass Filtering (Edge Detection)
    * Median Filtering (Noise Reduction)

---

## 1. Input Image & Histogram Analysis (Q1 & Q2)

The input is a custom 64x64 grayscale image. We analyzed its intensity distribution and applied **Histogram Equalization** to improve contrast.

| Original Input Image | Histogram Data (Memory View) |
| :---: | :---: |
| ![mainimg](https://github.com/user-attachments/assets/c9e4f99c-5d31-426a-9894-96cd0376dcff) | ![histogran_values](https://github.com/user-attachments/assets/5331aae1-bda3-4daf-8d68-ceef0230b679) |

### Histogram Equalization Results

After applying the equalization algorithm, the pixel intensities are redistributed.

| Equalized Histogram Data |
| :---: |
| ![equalized_image](https://github.com/user-attachments/assets/01639020-37ad-4d54-b2a5-6e24c6ddc781) |

---

## 2. Spatial Filtering: Convolution (Q3)

We implemented a generic 3x3 convolution kernel function to apply both Low Pass and High Pass filters.

### A. Low Pass Filter (Blurring)
* **Kernel:** 3x3 Matrix (All 1s)
* **Divisor:** 9
* **Effect:** Softens edges and reduces noise.

| Filtered Image (LPF) | Memory View (Result Data) |
| :---: | :---: |
| ![lpf_tested](https://github.com/user-attachments/assets/9371886a-6ac8-4d4a-b7e2-b13c6e080595) | ![lpf](https://github.com/user-attachments/assets/6118586e-bf78-484e-8013-4f9610f0fabe) |

### B. High Pass Filter (Edge Detection)
* **Kernel:** Laplacian Style (-1 surrounding, 8 center)
* **Effect:** Highlights high-frequency components (edges).

| Filtered Image (HPF) | Memory View (Result Data) |
| :---: | :---: |
| ![hpf_tested](https://github.com/user-attachments/assets/6b3077db-9c87-4662-a565-1ced956b52af) | ![hpf](https://github.com/user-attachments/assets/84b37f22-fc80-43a4-8626-9a2ef25294f2) |

---

## 3. Non-Linear Filtering: Median Filter (Q4)

A 3x3 Median Filter was implemented using a sorting algorithm (Bubble Sort) to remove noise (salt-and-pepper) while preserving edges better than LPF.

| Filtered Image (Median) | Memory View (Result Data) |
| :---: | :---: |
| ![median_tested](https://github.com/user-attachments/assets/3ca6522f-3af5-4f06-99c8-d2465bd8feec) | ![median](https://github.com/user-attachments/assets/985cacae-5a2a-4a61-b26b-b122441459c6) |

---

## 💻 Implementation Details

The core algorithms are implemented in C without using external image processing libraries.

```c
#include "main.h"
#include <stdint.h>
#include "image.h"

// --- Global Variables ---
uint32_t histogram[256];
uint32_t histogram_original[256];
uint8_t equalized_image[WIDTH * HEIGHT];
uint32_t histogram_equalized[256];
uint8_t image_median[WIDTH * HEIGHT];
uint8_t image_lpf[WIDTH * HEIGHT];
uint8_t image_hpf[WIDTH * HEIGHT];

// --- Kernels ---
const int8_t kernel_lpf[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
const int8_t kernel_hpf[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };

// --- Functions ---

// Median Filter Implementation
void median_filter_3x3(const uint8_t* src, uint8_t* dst, uint32_t width, uint32_t height) {
    int x, y, i, j;
    uint8_t window[9];
    uint8_t temp;

    for(i = 0; i < width * height; i++) dst[i] = 0;

    for (y = 1; y < height - 1; y++) {
        for (x = 1; x < width - 1; x++) {
            int k = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    window[k++] = src[(y + ky) * width + (x + kx)];
                }
            }
            // Bubble Sort
            for (i = 0; i < 9 - 1; i++) {
                for (j = 0; j < 9 - i - 1; j++) {
                    if (window[j] > window[j + 1]) {
                        temp = window[j];
                        window[j] = window[j + 1];
                        window[j + 1] = temp;
                    }
                }
            }
            dst[y * width + x] = window[4];
        }
    }
}

// Convolution Implementation
void apply_convolution_3x3(const uint8_t* src, uint8_t* dst, uint32_t width, uint32_t height, const int8_t* kernel, int divisor) {
    int x, y, kx, ky, sum, pixel_idx, kernel_idx;

    for(int i=0; i < width*height; i++) dst[i] = 0;

    for (y = 1; y < height - 1; y++) {
        for (x = 1; x < width - 1; x++) {
            sum = 0;
            for (ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    pixel_idx = (y + ky) * width + (x + kx);
                    kernel_idx = (ky + 1) * 3 + (kx + 1);
                    sum += src[pixel_idx] * kernel[kernel_idx];
                }
            }
            if (divisor != 0) sum = sum / divisor;
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            dst[y * width + x] = (uint8_t)sum;
        }
    }
}

void calculate_histogram(const uint8_t* image, uint32_t width, uint32_t height, uint32_t* hist) {
    uint32_t i;
    uint32_t total_pixels = width * height;

    for (i = 0; i < 256; i++) {
        hist[i] = 0;
    }

    for (i = 0; i < total_pixels; i++) {
        uint8_t pixel_val = image[i];
        hist[pixel_val]++;
    }
}

void histogram_equalization(const uint8_t* input_img, uint8_t* output_img, uint32_t width, uint32_t height) {
    uint32_t hist[256] = {0};
    uint32_t cdf[256] = {0};
    uint8_t map[256] = {0};
    uint32_t i;
    uint32_t total_pixels = width * height;

    for (i = 0; i < total_pixels; i++) {
        hist[input_img[i]]++;
    }

    cdf[0] = hist[0];
    for (i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    for (i = 0; i < 256; i++) {
        map[i] = (uint8_t)((cdf[i] * 255) / total_pixels);
    }

    for (i = 0; i < total_pixels; i++) {
        output_img[i] = map[input_img[i]];
    }
}
int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();


  //calculate_histogram(image, WIDTH, HEIGHT, histogram);
  //histogram_equalization(image, equalized_image, WIDTH, HEIGHT);
  //calculate_histogram(equalized_image, WIDTH, HEIGHT, histogram_equalized);
  //apply_convolution_3x3(image, image_lpf, WIDTH, HEIGHT, kernel_lpf, 9);
  //apply_convolution_3x3(image, image_hpf, WIDTH, HEIGHT, kernel_hpf, 1);
  median_filter_3x3(image, image_median, WIDTH, HEIGHT);

  while (1)
  {

  }
}



void SystemClock_Config(void)
{

}

static void MX_GPIO_Init(void)
{

}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}
