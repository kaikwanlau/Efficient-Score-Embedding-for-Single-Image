# Efficient Score Embedding for Single-Image Denoising

[![arXiv](https://img.shields.io/badge/arXiv-2511.17634-b31b1b.svg)](https://arxiv.org/abs/2511.17634)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Results

As reported in the paper (Table 1), this method achieves a **~27x speedup** compared to a standard DDPM baseline on CIFAR-10 (32x32) while maintaining high image quality.

| Method | Target SSIM | Training Time (s) | Speedup |
| :--- | :---: | :---: | :---: |
| Standard DDPM | 0.99 | ~1634.93s | 1x |
| **Our Method (This Code)** | **0.99** | **~60.27s** | **~27.13x** |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/efficient-score-embedding-single-image.git](https://github.com/YOUR_USERNAME/efficient-score-embedding-single-image.git)
    cd efficient-score-embedding-single-image
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy scipy matplotlib tqdm pillow
    ```

## Usage

1.  **Prepare your image:**
    Place a 32x32 image named `image_CIFAR-10_32x32.png` in the root directory (or update the `IMAGE_PATH` variable in the script).

2.  **Run the experiment:**
    ```bash
    python main_score_embedding.py
    ```

    The script will:
    * Pre-compute the scores using the Fokker-Planck solver.
    * Embed the scores into the image.
    * Train the U-Net model until the target SSIM (0.99) is reached.
    * Save the results (plots, denoised images, and metrics) to the `Results/` directory.
    * <img width="3060" height="448" alt="miku2024_denoising_process" src="https://github.com/user-attachments/assets/b471650c-850a-4cc4-8885-5f1754c34a7c" />
    * <img width="3060" height="448" alt="miku2025_denoising_process" src="https://github.com/user-attachments/assets/172748bc-8461-43d1-94ad-c44d8cfff3b1" />


## Repository Structure

* `main_score_embedding.py`: The main script implementing the Fokker-Planck solver, U-Net, and training loop.
* `assets/`: Contains sample images for testing.
* `results/`: (Generated at runtime) Stores training logs, loss curves, and generated images.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{lau2025efficient,
  title={Efficient Score Pre-computation for Diffusion Models via Cross-Matrix Krylov Projection},
  author={Lau, Kaikwan and Na, Andrew S and Wan, Justin WL},
  journal={arXiv preprint arXiv:2511.17634},
  year={2025}
}
