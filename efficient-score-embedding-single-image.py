# ===================================================================================
#
# This script implements the Single-Image Denoising experiment using Score Embedding,
# reproducing the "Our Method" baseline from Experiment 1 of the paper:
# "Efficient Score Pre-computation for Diffusion Models via Cross-Matrix Krylov Projection"
# (arXiv:2511.17634).
#
# Actions:
# 1. Load a single custom image (e.g., 32x32).
# 2. Pre-compute exact scores by numerically solving the Fokker-Planck equation
#    [cite_start]using a finite difference solver [cite: 38-39, 41, 71].
# [cite_start]3. Embed these scores into the image via the probability flow ODE (Transport Equation) [cite: 209-210].
# 4. Train a U-Net model to match these scores with SSIM-based early stopping.
# 5. Generate and save training metrics (MSE, PSNR) and the final denoised result.
#
# Citation:
# If you use this code, please cite the following paper:
#
# @article{lau2025efficient,
#   title={Efficient Score Pre-computation for Diffusion Models via Cross-Matrix Krylov Projection},
#   author={Lau, Kaikwan and Na, Andrew S and Wan, Justin WL},
#   journal={arXiv preprint arXiv:2511.17634},
#   year={2025}
# }
#
# ===================================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from PIL import Image

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============= BASIC FOKKER-PLANCK SOLVER (PAPER-ALIGNED) =============

class BasicFokkerPlanckSolver:
    """Basic FP solver following the paper's methodology."""

    def __init__(self, H, W, T, N_time, g_func, f_func=None, tol=1e-6):
        self.H = H
        self.W = W
        self.T = T
        self.N_time = N_time
        self.dt = T / N_time
        self.g_func = g_func
        self.f_func = f_func if f_func is not None else lambda x, t: np.zeros((x.shape[0], 2))
        self.tol = tol

        self.dx = 1.0 / (H - 1)
        self.dy = 1.0 / (W - 1)
        self.N_space = H * W

        print(f"Basic FP Solver initialized:")
        print(f"  Grid: {H}x{W}, Time steps: {N_time}")
        print(f"  dx={self.dx:.4f}, dy={self.dy:.4f}, dt={self.dt:.4f}")

    def initial_density(self, x_samples):
        """Basic initial density estimation."""
        x_samples = np.clip(x_samples, 0, 1)
        kde = gaussian_kde(x_samples.T, bw_method='scott')

        x = np.linspace(0, 1, self.H)
        y = np.linspace(0, 1, self.W)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        positions = np.vstack([xx.ravel(), yy.ravel()])

        density = kde(positions).reshape(self.H, self.W)
        density = np.maximum(density, 1e-10)
        density = density / (density.sum() * self.dx * self.dy)
        log_density = np.log(density)

        return log_density.ravel()

    def construct_system_matrix(self, n, m_tilde, f_vals):
        """
        CORRECTED matrix construction based on the finite difference scheme
        from the provided PDF document (pages 7-9).
        The variable `m_tilde` corresponds to `m_current` in the solve loop,
        representing the guess for the current time step `n`, which is `m_tilde` in the PDF.
        """
        t = n * self.dt
        g_n = self.g_func(t)
        g2 = g_n * g_n

        m_tilde_2d = m_tilde.reshape(self.H, self.W)
        f_x = f_vals[:, 0].reshape(self.H, self.W)
        f_y = f_vals[:, 1].reshape(self.H, self.W)

        # Gradient of m_tilde, used for the non-linear coefficients
        grad_m_tilde_x = np.zeros((self.H, self.W))
        grad_m_tilde_y = np.zeros((self.H, self.W))

        for i in range(self.H):
            for j in range(self.W):
                if 0 < i < self.H - 1:
                    grad_m_tilde_x[i, j] = (m_tilde_2d[i + 1, j] - m_tilde_2d[i - 1, j]) / (2 * self.dx)
                elif i == 0:
                    grad_m_tilde_x[i, j] = (m_tilde_2d[i + 1, j] - m_tilde_2d[i, j]) / self.dx
                else:  # i == H - 1
                    grad_m_tilde_x[i, j] = (m_tilde_2d[i, j] - m_tilde_2d[i - 1, j]) / self.dx

                if 0 < j < self.W - 1:
                    grad_m_tilde_y[i, j] = (m_tilde_2d[i, j + 1] - m_tilde_2d[i, j - 1]) / (2 * self.dy)
                elif j == 0:
                    grad_m_tilde_y[i, j] = (m_tilde_2d[i, j + 1] - m_tilde_2d[i, j]) / self.dy
                else:  # j == W - 1
                    grad_m_tilde_y[i, j] = (m_tilde_2d[i, j] - m_tilde_2d[i, j - 1]) / self.dy

        row_ind = []
        col_ind = []
        data = []

        # Assuming dx = dy as per the PDF
        dx2 = self.dx * self.dx

        for i in range(self.H):
            for j in range(self.W):
                k = i * self.W + j

                # Diagonal coefficient for m_i,j
                # In the PDF, a factor of 2g^2 is used. Since dx=dy, it becomes 4g^2, but the Laplacian stencil
                # uses 1/dx^2, and the coefficients for neighbors are divided by 2dx, leading to this form.
                # The PDF derivation shows (1/dt + 2g^2/dx^2)m_i,j after simplification.
                c_diag = (1.0 / self.dt) + (2.0 * g2 / dx2)
                row_ind.append(k)
                col_ind.append(k)
                data.append(c_diag)

                # Off-diagonal coefficients
                # Using coefficient definitions from pages 8 and 9 of the PDF

                # East neighbor (m_i+1,j)
                if i < self.H - 1:
                    fx_ij = f_x[i, j]
                    grad_x_term = 0.25 * g2 * (m_tilde_2d[i + 1, j] - m_tilde_2d[i - 1, j] if i > 0 else 2 * (
                            m_tilde_2d[i + 1, j] - m_tilde_2d[i, j])) / self.dx
                    c_east = (1.0 / (2.0 * self.dx)) * (-g2 / self.dx + fx_ij - grad_x_term)
                    row_ind.append(k)
                    col_ind.append(k + self.W)
                    data.append(c_east)

                # West neighbor (m_i-1,j)
                if i > 0:
                    fx_ij = f_x[i, j]
                    # Note the sign change for f_x and the gradient term
                    grad_x_term = 0.25 * g2 * (m_tilde_2d[i + 1, j] - m_tilde_2d[i - 1, j] if i < self.H - 1 else 2 * (
                            m_tilde_2d[i, j] - m_tilde_2d[i - 1, j])) / self.dx
                    c_west = (1.0 / (2.0 * self.dx)) * (-g2 / self.dx - fx_ij + grad_x_term)
                    row_ind.append(k)
                    col_ind.append(k - self.W)
                    data.append(c_west)

                # North neighbor (m_i,j+1)
                if j < self.W - 1:
                    fy_ij = f_y[i, j]
                    grad_y_term = 0.25 * g2 * (m_tilde_2d[i, j + 1] - m_tilde_2d[i, j - 1] if j > 0 else 2 * (
                            m_tilde_2d[i, j + 1] - m_tilde_2d[i, j])) / self.dy
                    c_north = (1.0 / (2.0 * self.dy)) * (-g2 / self.dy + fy_ij - grad_y_term)
                    row_ind.append(k)
                    col_ind.append(k + 1)
                    data.append(c_north)

                # South neighbor (m_i,j-1)
                if j > 0:
                    fy_ij = f_y[i, j]
                    # Note the sign change for f_y and the gradient term
                    grad_y_term = 0.25 * g2 * (m_tilde_2d[i, j + 1] - m_tilde_2d[i, j - 1] if j < self.W - 1 else 2 * (
                            m_tilde_2d[i, j] - m_tilde_2d[i, j - 1])) / self.dy
                    c_south = (1.0 / (2.0 * self.dy)) * (-g2 / self.dy - fy_ij + grad_y_term)
                    row_ind.append(k)
                    col_ind.append(k - 1)
                    data.append(c_south)

        A = sp.csr_matrix((data, (row_ind, col_ind)), shape=(self.N_space, self.N_space))
        return A

    def construct_rhs(self, m_prev, f_vals):
        """
        CORRECTED RHS vector 'b' construction.
        Based on equation (115) from the PDF.
        The variable `m_prev` corresponds to `m^{n-1}`.
        """
        f_x = f_vals[:, 0].reshape(self.H, self.W)
        f_y = f_vals[:, 1].reshape(self.H, self.W)

        # Divergence of f
        div_f = np.zeros((self.H, self.W))

        for i in range(self.H):
            for j in range(self.W):
                # Central difference for divergence
                if 0 < i < self.H - 1:
                    dfx_dx = (f_x[i + 1, j] - f_x[i - 1, j]) / (2 * self.dx)
                elif i == 0:
                    dfx_dx = (f_x[i + 1, j] - f_x[i, j]) / self.dx
                else:  # i == H - 1
                    dfx_dx = (f_x[i, j] - f_x[i - 1, j]) / self.dx

                if 0 < j < self.W - 1:
                    dfy_dy = (f_y[i, j + 1] - f_y[i, j - 1]) / (2 * self.dy)
                elif j == 0:
                    dfy_dy = (f_y[i, j + 1] - f_y[i, j]) / self.dy
                else:  # j == W - 1
                    dfy_dy = (f_y[i, j] - f_y[i, j - 1]) / self.dy

                div_f[i, j] = dfx_dx + dfy_dy

        # Construct b vector as per equation (115) from the PDF
        # b_ij = m_ij^{n-1} / dt - Div(f)_ij
        b = m_prev / self.dt - div_f.ravel()
        return b

    def solve(self, initial_samples, max_iter=5):
        """Basic solve following paper's approach."""
        m_0 = self.initial_density(initial_samples)

        log_density = np.zeros((self.N_time + 1, self.N_space))
        log_density[0] = m_0

        x = np.linspace(0, 1, self.H)
        y = np.linspace(0, 1, self.W)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        print(f"Solving basic FP equation...")

        for n in tqdm(range(1, self.N_time + 1), desc="Time steps"):
            t = n * self.dt
            f_vals = self.f_func(grid_points, t)

            # m_prev is m^{n-1}, the solution from the previous time step
            m_prev = log_density[n - 1].copy()
            # m_current is the iterative guess for m^n, which is tilde{m} in the PDF
            m_current = m_prev.copy()

            # The RHS vector b depends only on the previous time step and is constant during the iteration
            b = self.construct_rhs(m_prev, f_vals)

            # Basic fixed-point iteration (paper's approach)
            for k in range(max_iter):
                # The matrix A depends on the current guess m_current (m_tilde)
                A = self.construct_system_matrix(n, m_current, f_vals)

                try:
                    m_new = spsolve(A, b)
                except Exception as e:
                    print(f"Warning: Solver failed at time step {n} with error: {e}. Using previous solution.")
                    m_new = m_current
                    break

                error = np.linalg.norm(m_new - m_current) / (np.linalg.norm(m_current) + 1e-8)

                # Basic update (no damping)
                m_current = m_new

                if error < self.tol:
                    break

            log_density[n] = m_current

        score = self.compute_score(log_density)
        return log_density.reshape(self.N_time + 1, self.H, self.W), score

    def compute_score(self, log_density):
        """Basic score computation."""
        score = np.zeros((self.N_time + 1, self.N_space, 2))

        for n in range(self.N_time + 1):
            m = log_density[n].reshape(self.H, self.W)

            grad_x = np.zeros_like(m)
            grad_y = np.zeros_like(m)

            for i in range(self.H):
                for j in range(self.W):
                    # Basic gradient computation
                    if 0 < i < self.H - 1:
                        grad_x[i, j] = (m[i + 1, j] - m[i - 1, j]) / (2 * self.dx)
                    elif i == 0:
                        grad_x[i, j] = (m[i + 1, j] - m[i, j]) / self.dx
                    else:
                        grad_x[i, j] = (m[i, j] - m[i - 1, j]) / self.dx

                    if 0 < j < self.W - 1:
                        grad_y[i, j] = (m[i, j + 1] - m[i, j - 1]) / (2 * self.dy)
                    elif j == 0:
                        grad_y[i, j] = (m[i, j + 1] - m[i, j]) / self.dy
                    else:
                        grad_y[i, j] = (m[i, j] - m[i, j - 1]) / self.dy

            score[n, :, 0] = grad_x.ravel()
            score[n, :, 1] = grad_y.ravel()

        return score


# ============= BASIC U-NET ARCHITECTURE =============

class BasicAttentionBlock(nn.Module):
    """Basic attention block."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=2)

        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)

        return x + self.proj_out(out)


class BasicUNet(nn.Module):
    """Enhanced U-Net for 128x128 images."""

    def __init__(self, n_channels=3, time_dim=256):
        super().__init__()

        # Basic time embedding (paper uses linear)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Additional layer for 128x128

        self.attn = BasicAttentionBlock(1024)

        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)  # Additional layer for 128x128

        self.outc = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self._get_timestep_embedding(t, 256)
        t_emb = self.time_mlp(t_emb)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.attn(x5)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.view(-1, 1) * emb.view(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============= MAIN CLASS WITH BASIC PAPER METHODOLOGY =============

class BasicScoreEmbeddingCustomImage:
    """Basic implementation following the paper's methodology."""

    def __init__(self, image_path):
        self.device = device
        self.image_path = image_path

        # Paper's basic parameters
        self.T = 1.0
        self.N_time = 100
        self.g_func = lambda t: 0.5  # Paper uses constant g
        self.f_func = lambda x, t: np.zeros_like(x)  # Paper uses zero drift
        self.lambda_func = lambda t: 0.1

        self.img_size = 32
        self.channels = 3

        self.model = BasicUNet(n_channels=self.channels).to(self.device)
        print(f"Basic model initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.fp_solver = BasicFokkerPlanckSolver(
            H=self.img_size, W=self.img_size,
            T=self.T, N_time=self.N_time,
            g_func=self.g_func, f_func=self.f_func
        )

        self.output_dir = 'Results'
        os.makedirs(self.output_dir, exist_ok=True)

        print("\nBasic Paper-Aligned Parameters:")
        print(f"  g(t) = {self.g_func(0.5)} (constant)")
        print(f"  f(x,t) = 0 (zero drift)")
        print(f"  λ(t) = {self.lambda_func(0.5)} (basic perturbation)")
        print(f"  Image size: {self.img_size}x{self.img_size}")

    def load_custom_image(self):
        """Load and preprocess custom image from path."""
        try:
            img = Image.open(self.image_path)
            print(f"Loaded image from: {self.image_path}")
            print(f"Original image size: {img.size}")
            print(f"Original image mode: {img.mode}")

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            image_tensor = transform(img)
            print(f"Processed image shape: {image_tensor.shape}")
            print(f"Image value range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

            return image_tensor

        except Exception as e:
            print(f"Error loading image from {self.image_path}: {e}")
            return None

    def precompute_score_for_image(self, img):
        """Basic score computation following paper."""
        print("Pre-computing score...")

        if torch.is_tensor(img):
            img_np = img.cpu().numpy().transpose(1, 2, 0)
        else:
            img_np = img

        img_np = np.clip(img_np, 0, 1)

        positions = []
        weights = []

        for i in range(self.img_size):
            for j in range(self.img_size):
                x_pos = i / (self.img_size - 1)
                y_pos = j / (self.img_size - 1)
                positions.append([x_pos, y_pos])

                intensity = np.mean(img_np[i, j, :])
                weights.append(intensity + 0.1)

        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / weights.sum()

        n_samples = 2000
        indices = np.random.choice(len(positions), size=n_samples, p=weights, replace=True)
        samples = positions[indices]

        noise_level = 0.01
        samples += noise_level * np.random.randn(n_samples, 2)
        samples = np.clip(samples, 0, 1)

        _, score = self.fp_solver.solve(samples, max_iter=5)

        return score

    def embed_score_in_image(self, original_image, score):
        """Basic score embedding following paper's transport equation."""
        print("Embedding score into image...")

        embedded_images = []
        x = original_image.clone()
        embedded_images.append(x.cpu())

        timesteps = np.linspace(1, self.N_time, 10, dtype=int)

        for n in timesteps:
            if n < len(score):
                t = n / self.N_time * self.T
                dt = self.T / self.N_time

                score_n = score[n]
                score_x = score_n[:, 0].reshape(self.img_size, self.img_size)
                score_y = score_n[:, 1].reshape(self.img_size, self.img_size)

                score_x_tensor = torch.tensor(score_x, device=self.device, dtype=torch.float32)
                score_y_tensor = torch.tensor(score_y, device=self.device, dtype=torch.float32)

                g_t = self.g_func(t)

                for c in range(self.channels):
                    x[c] += -0.5 * g_t ** 2 * score_x_tensor * dt

                if n < self.N_time:
                    noise = torch.randn_like(x) * g_t * np.sqrt(dt) * 0.1
                    x = x + noise

                x = torch.clamp(x, 0, 1)
                embedded_images.append(x.cpu())

        return embedded_images

    def train_with_score_embedding(self, original_image, embedded_images, epochs=50, lr=1e-3):
        """Basic training following paper's approach."""
        print("Training with basic score embedding...")

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        mse_values = []

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_mse = 0
            n_batches = 0

            for timestep_idx, x_embedded in enumerate(embedded_images):
                if timestep_idx == 0:
                    continue

                x_embedded = x_embedded.to(self.device)

                for _ in range(2):
                    t = timestep_idx / len(embedded_images)
                    t = np.clip(t, 0.01, 0.99)

                    z = torch.randn_like(x_embedded)
                    lambda_t = self.lambda_func(t)
                    x_perturbed = x_embedded + lambda_t * z

                    t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)
                    x_batch = x_perturbed.unsqueeze(0)
                    score_pred = self.model(x_batch, t_tensor)[0]

                    g_t = self.g_func(t * self.T)
                    loss = torch.mean((score_pred * lambda_t + z) ** 2) / (2 * g_t ** 2)

                    with torch.no_grad():
                        target_score = -z / lambda_t
                        mse = torch.mean((score_pred - target_score) ** 2)
                        epoch_mse += mse.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            avg_mse = epoch_mse / n_batches if n_batches > 0 else 0
            losses.append(avg_loss)
            mse_values.append(avg_mse)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}")

        return losses, mse_values

    def denoise_image(self, noisy_image, num_steps=50):
        """Basic denoising following paper's reverse SDE."""
        self.model.eval()

        with torch.no_grad():
            x = noisy_image.clone()
            denoised_steps = []

            denoised_steps.append(x.cpu().numpy().transpose(1, 2, 0))

            for i in range(num_steps):
                t = 1.0 - (i / num_steps)
                if t <= 0:
                    break

                dt = 1.0 / num_steps

                t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)
                x_batch = x.unsqueeze(0)
                score_pred = self.model(x_batch, t_tensor)[0]

                g_t = self.g_func(t * self.T)

                x = x + score_pred * g_t ** 2 * dt

                if i < num_steps - 1:
                    x = x + g_t * np.sqrt(dt) * torch.randn_like(x) * 0.1

                x = torch.clamp(x, 0, 1)

                if (i + 1) % 10 == 0:
                    denoised_steps.append(x.cpu().numpy().transpose(1, 2, 0))

            denoised_steps.append(x.cpu().numpy().transpose(1, 2, 0))

        return denoised_steps

    def compute_metrics(self, original, denoised):
        """Basic metrics computation."""

        def compute_ssim(y_true, y_pred):
            L = 1.0
            c1 = (0.01 * L) ** 2
            c2 = (0.03 * L) ** 2

            y_true = y_true.astype(np.float64)
            y_pred = y_pred.astype(np.float64)

            mu_y = np.mean(y_true)
            mu_y_tilde = np.mean(y_pred)

            var_y = np.var(y_true)
            var_y_tilde = np.var(y_pred)
            cov_y_y_tilde = np.cov(y_true.flatten(), y_pred.flatten())[0, 1]

            term1 = (2 * mu_y * mu_y_tilde + c1)
            term2 = (2 * cov_y_y_tilde + c2)
            term3 = (mu_y ** 2 + mu_y_tilde ** 2 + c1)
            term4 = (var_y + var_y_tilde + c2)

            return (term1 * term2) / (term3 * term4)

        original_clipped = np.clip(original, 0, 1)
        denoised_clipped = np.clip(denoised, 0, 1)

        mse = np.mean((original_clipped - denoised_clipped) ** 2)

        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(1.0 / mse)

        ssim = compute_ssim(original_clipped, denoised_clipped)

        return mse, psnr, ssim

    def add_noise_to_image(self, image, noise_level=0.5):
        """Basic noise addition."""
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def save_results(self, original_img, embedded_images, denoised_steps, losses, mse_values, final_mse, final_psnr,
                     final_ssim):
        """Basic result saving following paper's format."""
        print("Saving basic results...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(mse_values)
        ax2.set_title('Training MSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, img in enumerate(embedded_images[:10]):
            if torch.is_tensor(img):
                img_np = img.numpy().transpose(1, 2, 0)
            else:
                img_np = img
            img_np = np.clip(img_np, 0, 1)

            axes[i].imshow(img_np)
            axes[i].set_title(f'Step {i}')
            axes[i].axis('off')

        plt.suptitle('Score Embedding Progression (Paper Method)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_embedding_progression.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, len(denoised_steps), figsize=(3 * len(denoised_steps), 3))
        if len(denoised_steps) == 1:
            axes = [axes]

        for i, img in enumerate(denoised_steps):
            img_display = np.clip(img, 0, 1)
            axes[i].imshow(img_display)
            axes[i].set_title(f'Step {i}')
            axes[i].axis('off')

        plt.suptitle('Denoising Process (Paper Method)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'denoising_process.png'), dpi=150, bbox_inches='tight')
        plt.close()

        original_pil = Image.fromarray((np.clip(original_img, 0, 1) * 255).astype(np.uint8))
        original_pil.save(os.path.join(self.output_dir, 'original.png'))

        final_denoised = denoised_steps[-1]
        final_pil = Image.fromarray((np.clip(final_denoised, 0, 1) * 255).astype(np.uint8))
        final_pil.save(os.path.join(self.output_dir, 'final_denoised.png'))

        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write("=== BASIC PAPER-ALIGNED SCORE EMBEDDING RESULTS (128x128 Custom Image) ===\n\n")
            f.write(f"Input Image: {self.image_path}\n")
            f.write(f"Image Size: {self.img_size}x{self.img_size}\n\n")
            f.write(f"Final Metrics:\n")
            f.write(f"  MSE: {final_mse:.6f}\n")
            f.write(f"  PSNR: {final_psnr:.2f} dB\n")
            f.write(f"  SSIM: {final_ssim:.4f}\n\n")
            f.write(f"Training Details:\n")
            f.write(f"  Epochs: {len(losses)}\n")
            f.write(f"  Final loss: {losses[-1] if losses else 'N/A'}\n")
            f.write(f"  Final MSE: {mse_values[-1] if mse_values else 'N/A'}\n\n")
            f.write(f"Paper Method Features:\n")
            f.write(f"  ✓ Basic Fokker-Planck solver\n")
            f.write(f"  ✓ Simple score embedding via transport equation\n")
            f.write(f"  ✓ Enhanced U-Net architecture for 128x128\n")
            f.write(f"  ✓ Standard training loop\n")
            f.write(f"  ✓ Basic reverse SDE denoising\n")

        print(f"Basic results saved to: {self.output_dir}/")

    def _train_one_epoch(self, optimizer, embedded_images):
        """Helper function to run a single epoch of training."""
        self.model.train()
        epoch_loss = 0
        epoch_mse = 0
        n_batches = 0

        for timestep_idx, x_embedded in enumerate(embedded_images):
            if timestep_idx == 0:
                continue
            x_embedded = x_embedded.to(self.device)
            for _ in range(2):
                t = timestep_idx / len(embedded_images)
                t = np.clip(t, 0.01, 0.99)
                z = torch.randn_like(x_embedded)
                lambda_t = self.lambda_func(t)
                x_perturbed = x_embedded + lambda_t * z
                t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)
                x_batch = x_perturbed.unsqueeze(0)
                score_pred = self.model(x_batch, t_tensor)[0]
                g_t = self.g_func(t * self.T)
                loss = torch.mean((score_pred * lambda_t + z) ** 2) / (2 * g_t ** 2)
                with torch.no_grad():
                    target_score = -z / lambda_t
                    mse = torch.mean((score_pred - target_score) ** 2)
                    epoch_mse += mse.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        avg_mse = epoch_mse / n_batches if n_batches > 0 else 0
        return avg_loss, avg_mse

    def run_experiment_with_ssim_stopping(self, target_ssim=0.99, max_epochs=2000):
        """
        Runs the experiment with an early stopping condition based on SSIM.
        """
        print(f"\n=== Experiment with SSIM Early Stopping (Target: {target_ssim}) ===\n")

        original_image = self.load_custom_image()
        if original_image is None:
            print("Failed to load custom image. Experiment aborted.")
            return

        original_image = original_image.to(self.device)
        original_img_np = original_image.cpu().numpy().transpose(1, 2, 0)

        print("\n--- Pre-computation and Embedding ---")
        score = self.precompute_score_for_image(original_image)
        embedded_images = self.embed_score_in_image(original_image, score)

        print("\n--- Training with SSIM-based Early Stopping ---")
        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        losses, mse_values = [], []

        final_denoised, denoised_steps = None, None

        for epoch in range(max_epochs):
            avg_loss, avg_mse = self._train_one_epoch(optimizer, embedded_images)
            losses.append(avg_loss)
            mse_values.append(avg_mse)

            noisy_image = self.add_noise_to_image(original_image, noise_level=0.506)
            denoised_steps = self.denoise_image(noisy_image, num_steps=50)
            final_denoised = denoised_steps[-1]
            _, _, current_ssim = self.compute_metrics(original_img_np, final_denoised)

            print(f"Epoch {epoch + 1}/{max_epochs} | Loss: {avg_loss:.6f} | Current SSIM: {current_ssim:.4f}")

            if current_ssim >= target_ssim:
                print(f"\nTarget SSIM of {target_ssim} reached at epoch {epoch + 1}. Stopping training.")
                break
        else:
            print(f"\nMax epochs ({max_epochs}) reached without achieving target SSIM.")

        training_time = time.time() - start_time
        print(f"Total training time: {training_time:.2f} seconds")

        print("\n--- Finalizing and Saving Results ---")
        final_mse, final_psnr, final_ssim = self.compute_metrics(original_img_np, final_denoised)
        self.save_results(original_img_np, embedded_images, denoised_steps, losses, mse_values,
                          final_mse, final_psnr, final_ssim)

        print("\n" + "=" * 60)
        print(f"EARLY STOPPING RESULTS SUMMARY (Target SSIM: {target_ssim}):")
        print("=" * 60)
        print(f"Image source: {self.image_path}")
        print(f"Image size: {self.img_size}x{self.img_size}")
        print(f"Stopped at Epoch: {len(losses)}")
        print(f"Total training time: {training_time:.2f}s")
        print("\nFINAL METRICS:")
        print(f"Final MSE: {final_mse:.6f}")
        print(f"Final PSNR: {final_psnr:.2f} dB")
        print(f"Final SSIM: {final_ssim:.4f}")
        print(f"\nResults saved to: {self.output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    IMAGE_PATH = "miku2024.jpg"

    experiment = BasicScoreEmbeddingCustomImage(IMAGE_PATH)
    experiment.run_experiment_with_ssim_stopping(target_ssim=0.95, max_epochs=2000)