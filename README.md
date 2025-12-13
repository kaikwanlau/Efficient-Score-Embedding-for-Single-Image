# Efficient Score Embedding for Single-Image Denoising

[![arXiv](https://img.shields.io/badge/arXiv-2511.17634-b31b1b.svg)](https://arxiv.org/abs/2511.17634)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img width="3060" height="448" alt="miku2024_denoising_process" src="https://github.com/user-attachments/assets/bdfd981b-4c59-49bc-b964-eb76ba48ba8d" />

<img width="3060" height="448" alt="miku2025_denoising_process" src="https://github.com/user-attachments/assets/5b7a7027-079c-4d99-92bd-254f9b9c67ca" />


This repository contains the official PyTorch implementation of the **Single-Image Denoising Experiment** (Experiment 1) from the paper:

**"Efficient Score Pre-computation for Diffusion Models via Cross-Matrix Krylov Projection"**
*Kaikwan Lau, Andrew S. Na, Justin W.L. Wan* (2025)

## Overview

This code reproduces the "Our Method" baseline results presented in **Table 1** of the paper. It demonstrates the **Score Embedding Framework**, a technique that significantly accelerates diffusion model training for single images by leveraging numerical PDE solvers.

**Note:** This repository specifically implements the **Score Embedding** technique for single-image generation. The *Cross-Matrix Krylov Projection* (used for accelerating multi-image pre-computation) is part of a separate experimental setup described in the paper.

### The Method
Instead of learning score functions from scratch (like standard DDPM), this approach:
1.  **Pre-computes exact scores** by numerically solving the Fokker-Planck equation using a finite-difference solver.
2.  **Embeds the scores** directly into the image data via the Probability Flow ODE (Transport Equation).
3.  **Trains the model** to match these pre-computed scores, allowing for rapid convergence.
4.  **Optimizes training time** using an SSIM-based early stopping mechanism.

## Experimental Results

Our method achieves orders-of-magnitude faster training convergence compared to DDPM while maintaining high image quality.

| Dataset | Resolution | Target SSIM | Time (DDPM) | **Time (Ours)** | **Speedup** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | 32 × 32 | 0.99 | 1634.93s | **60.27s** | **27.13×** |
| **CelebA** | 64 × 64 | 0.99 | 14131.53s | **122.45s** | **115.40×** |
| **CelebA** | 128 × 128 | 0.90 | 11319.86s | **186.09s** | **60.83×** |

> **Key Result:** On CelebA ($64 \times 64$), our method reduces training time from **~4 hours** (14,131s) to just **~2 minutes** (122s).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kaikwanlau/efficient-score-embedding-single-image.git
    cd efficient-score-embedding-single-image
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy scipy matplotlib tqdm pillow
    ```

## Usage

1.  **Prepare your image:**
    Place a 32x32 image named `miku.jpg` in the root directory (or update the `IMAGE_PATH` variable in the script).

2.  **Run the experiment:**
    ```bash
    python efficient-score-embedding-single-image.py
    ```

    The script will:
    * Pre-compute the scores using the Fokker-Planck solver.
    * Embed the scores into the image.
    * Train the U-Net model until the target SSIM (0.95) is reached.
    * Save the results (plots, denoised images, and metrics) to the `Results/` directory.
    * ![Adobe Express - 10](https://github.com/user-attachments/assets/528cc80a-9a69-43b6-ba08-ff0ae77d0565)

## Repository Structure

* `efficient-score-embedding-single-image.py`: The main script implementing the Fokker-Planck solver, U-Net, and training loop.
* `miku2024.jpg`: An image demonstration of Magical Mirai 2024.
* `miku2025.jpg`: An image demonstration of Magical Mirai 2025.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{lau2025efficient,
  title={Efficient Score Pre-computation for Diffusion Models via Cross-Matrix Krylov Projection},
  author={Lau, Kaikwan and Na, Andrew S and Wan, Justin WL},
  journal={arXiv preprint arXiv:2511.17634},
  year={2025}
}
