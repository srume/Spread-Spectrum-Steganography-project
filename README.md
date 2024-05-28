# Spread Spectrum Steganography using Discrete Wavelet Transform (DWT)

## Overview

This project implements a spread spectrum steganography technique enhanced with Discrete Wavelet Transform (DWT) to embed secret information within digital images. The goal is to maintain high imperceptibility and security while manipulating the embedding strength parameter (alpha value). Evaluation metrics such as Peak Signal-to-Noise Ratio (PSNR), Mean Squared Error (MSE), and Structural Similarity Index (SSI) are used to assess the quality and fidelity of the stego images.

## Features

- Embed secret information within digital images using DWT.
- Adjustable alpha value for embedding strength.
- Evaluation of image quality using PSNR, MSE, and SSI metrics.
- Visualization of results through plots and graphs.

## Tools Used

- **Python**: The primary programming language for the project.
- **NumPy**: For numerical computations and array manipulation.
- **PyWavelets**: For applying Discrete Wavelet Transform (DWT).
- **Pillow (PIL)**: For image opening, manipulation, and saving.
- **Matplotlib**: For plotting images and graphs.
- **scikit-image**: For calculating evaluation metrics (MSE, SSI).

## What is Steganography?

Steganography is a technique of hiding secret data within non-secret files or messages to avoid detection. Unlike cryptography, which makes the data unreadable to unauthorized parties, steganography aims to conceal the existence of the data altogether. Image steganography involves embedding secret information within digital images, making it a powerful tool for secure communication.

## Spread Spectrum Steganography

Spread spectrum steganography is a method that spreads the secret information across a wide frequency spectrum. This makes the hidden data more resilient to detection and interference. By using the Discrete Wavelet Transform (DWT), we can effectively embed the secret image within the cover image's wavelet coefficients, thus enhancing the security and imperceptibility of the steganographic process.

## Discrete Wavelet Transform (DWT)

DWT is a mathematical transformation that decomposes an image into different frequency sub-bands. This allows for the separation of the image into high and low-frequency components. The high-frequency components typically contain detailed information, while the low-frequency components hold the approximation of the image. By embedding the secret data into the high-frequency components, we can ensure that the modifications remain imperceptible to human eyes.

## Implementation

**Image Preparation**:

Load the cover image and the secret image.
Resize the secret image to match the dimensions of the cover image.

**Embedding Process**:

Apply DWT to decompose both the cover image and the secret image into their wavelet coefficients.
Embed the secret image's wavelet coefficients into the cover image's coefficients using the specified alpha value.
Perform the inverse DWT to reconstruct the stego image from the modified coefficients.

<img src="https://github.com/srume/Spread-Spectrum-Steganography-project/blob/main/cover-image.jpg?raw=true" alt="cover-image" width="125px" height="125px">



**Evaluation Metrics**:

**Peak Signal-to-Noise Ratio (PSNR)**: 

Measures the quality of the stego image compared to the original cover image. Higher PSNR indicates better quality.

**Mean Squared Error (MSE)**: 

Measures the average squared difference between the original and stego images. Lower MSE indicates better quality.

**Structural Similarity Index (SSI)**:

Evaluates the perceived quality of the stego image by comparing it with the original image. Higher SSI indicates better quality.

**Analysis**:

By varying the alpha value, we analyze the trade-off between imperceptibility and embedding strength.
Plotting the evaluation metrics against different alpha values helps in understanding the impact of the alpha value on the stego image quality.




