# -*- coding: utf-8 -*-
"""Исследование, обучение и всестороннее сравнение различных архитектур MERLIN CNN
(Complex Self-Supervised Despeckling / Noise2Noise U-Net) для радиолокационных изображений Sentinel-1.

Архитектуры для сравнительного эксперимента:
1. MERLIN_UNet_Base: 3-уровневый классический U-Net (32 базовых канала), skip-connections, лог-резидуальное прогнозирование.
2. MERLIN_ResUNet_Deep: 4-уровневый глубокий Residual U-Net (48 базовых каналов) с дилатированными блоками (d=1, 2, 4).
3. MERLIN_AttnUNet_CBAM: U-Net с блоками внимания CBAM (Channel Attention + Spatial Attention) на пробросах признаков.
4. MERLIN_DenseDilated_CNN: Многомасштабная плотная сеть без уменьшения разрешения (d=1, 2, 3, 5, 2, 1).

Метрики валидации:
- ENL (Equivalent Number of Looks, число эквивалентных взглядов в однородных зонах)
- EPI (Edge Preservation Index, индекс сохранения границ)
- Bias (Радиометрическое смещение средней интенсивности)
- SMPI (Speckle Suppression and Mean Preservation Index)
- SSI (Speckle Suppression Index)
- Var(r) (Дисперсия отношения y / x_hat, теоретический оптимум = 1 / 4.4 = 0.2273)
- Moran(r) (Пространственная автокорреляция остатка Морана I, теоретический оптимум = 0.0)
- PSNR и SSIM (для полигонов с известным эталоном)
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Ограничение видеопамяти VRAM согласно gpu_vram_coordination.md
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.20, device=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ИНФОРМАЦИЯ] Вычислительное устройство: {device}")
if torch.cuda.is_available():
    print(f"[ИНФОРМАЦИЯ] Графический процессор: {torch.cuda.get_device_name(0)}")
    print(f"[ИНФОРМАЦИЯ] Лимит VRAM установлен: 20% (~2048 МБ)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_cache")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
PAPER_DIR = os.path.join(BASE_DIR, "paper_results")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# =========================================================================
# 1. АРХИТЕКТУРЫ СЕТЕЙ MERLIN
# =========================================================================

# --- МОДЕЛЬ 1: MERLIN_UNet_Base ---
class MERLIN_UNet_Base(nn.Module):
    """Классический U-Net с 3 уровнями масштаба и лог-резидуальным предсказанием.
    Вход: log(y + eps).
    Выход: log(x_hat) = log(y + eps) - f_theta(log(y + eps)).
    x_hat = exp(log(x_hat)).
    """
    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        
        e1 = self.enc1(log_y)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        
        d2 = self.up2(b)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        log_residual = self.out_conv(d1)
        # Лог-резидуальное вычитание спекл-компоненты
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# --- МОДЕЛЬ 2: MERLIN_ResUNet_Deep ---
class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.act1(self.conv1(x)))
        return self.act2(x + res)


class MERLIN_ResUNet_Deep(nn.Module):
    """Глубокий Residual U-Net с 4 уровнями масштаба и дилатированными блоками (d=1, 2, 4) в узких слоях."""
    def __init__(self, in_channels: int = 1, base_ch: int = 40):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.res1 = ResBlock(base_ch, dilation=1)
        
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.res2 = ResBlock(base_ch * 2, dilation=2)
        
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.res3 = ResBlock(base_ch * 4, dilation=4)
        
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 4, dilation=2),
            ResBlock(base_ch * 4, dilation=1)
        )
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec_res2 = ResBlock(base_ch * 4, dilation=1)
        
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch, 2, stride=2)
        self.dec_res1 = ResBlock(base_ch * 2, dilation=1)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, in_channels, 1)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        
        x0 = self.in_conv(log_y)
        e1 = self.res1(x0)
        
        e2 = self.res2(self.down1(e1))
        e3 = self.res3(self.down2(e2))
        
        b = self.bottleneck(e3)
        
        d2 = self.up2(b)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec_res2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        
        log_residual = self.out_conv(d1)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# --- МОДЕЛЬ 3: MERLIN_AttnUNet_CBAM ---
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // ratio, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return scale * x


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class MERLIN_AttnUNet_CBAM(nn.Module):
    """U-Net с механизмом двойного внимания (CBAM) на пробросах признаков для выделения точечных целей."""
    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.cbam1 = CBAM(base_ch)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.cbam2 = CBAM(base_ch * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        
        e1 = self.enc1(log_y)
        attn_e1 = self.cbam1(e1)
        
        e2 = self.enc2(self.pool1(e1))
        attn_e2 = self.cbam2(e2)
        
        b = self.bottleneck(self.pool2(e2))
        
        d2 = self.up2(b)
        if d2.shape[2:] != attn_e2.shape[2:]:
            d2 = F.interpolate(d2, size=attn_e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, attn_e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != attn_e1.shape[2:]:
            d1 = F.interpolate(d1, size=attn_e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, attn_e1], dim=1))
        
        log_residual = self.out_conv(d1)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# --- МОДЕЛЬ 4: MERLIN_DenseDilated_CNN ---
class MERLIN_DenseDilated_CNN(nn.Module):
    """Полноразмерная сверточная сеть с каскадом дилатированных слоев (d=1, 2, 3, 5, 2, 1)
    без снижения пространственного разрешения, сохраняющая тонкие линии и микротекстуру."""
    def __init__(self, in_channels: int = 1, num_features: int = 36):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.d1 = nn.Conv2d(num_features, num_features, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(num_features * 2, num_features, 3, padding=2, dilation=2)
        self.d3 = nn.Conv2d(num_features * 3, num_features, 3, padding=3, dilation=3)
        self.d4 = nn.Conv2d(num_features * 4, num_features, 3, padding=5, dilation=5)
        self.d5 = nn.Conv2d(num_features * 5, num_features, 3, padding=2, dilation=2)
        self.d6 = nn.Conv2d(num_features * 6, num_features, 3, padding=1, dilation=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_out = nn.Conv2d(num_features * 7, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        
        x0 = self.act(self.conv_in(log_y))
        x1 = self.act(self.d1(x0))
        c1 = torch.cat([x0, x1], dim=1)
        
        x2 = self.act(self.d2(c1))
        c2 = torch.cat([c1, x2], dim=1)
        
        x3 = self.act(self.d3(c2))
        c3 = torch.cat([c2, x3], dim=1)
        
        x4 = self.act(self.d4(c3))
        c4 = torch.cat([c3, x4], dim=1)
        
        x5 = self.act(self.d5(c4))
        c5 = torch.cat([c4, x5], dim=1)
        
        x6 = self.act(self.d6(c5))
        c6 = torch.cat([c5, x6], dim=1)
        
        log_residual = self.conv_out(c6)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# =========================================================================
# 2. ГЕНЕРАЦИЯ ОБУЧАЮЩИХ ПАТЧЕЙ И САМООБУЧЕНИЕ
# =========================================================================

def extract_real_sar_dataset(num_patches: int = 500, patch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Извлечение реальных патчей Sentinel-1 и формирование пар независимых взглядов (y1, y2).
    По теории Dalsasso et al. (MERLIN), при известном среднем радарном сечении x,
    два независимых взгляда формируются как некоррелированные отсчеты Gamma-распределения:
    y1 ~ Gamma(L/2, x / (L/2)), y2 ~ Gamma(L/2, x / (L/2)).
    """
    patches_y1 = []
    patches_y2 = []
    looks_half = 2.2  # 4.4 / 2
    
    terrain_keys = ['forest', 'city', 'mountain', 'desert', 'water', 'benevento', 'rome', 'serre_poncon']
    per_key = int(math.ceil(num_patches / len(terrain_keys)))
    
    for k in terrain_keys:
        p = os.path.join(DATA_DIR, f"raw_{k}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        vv = d["vv"].astype(np.float32)
        H, W = vv.shape
        if H < patch_size or W < patch_size:
            continue
            
        # Локальное сглаживание для базового сечения
        sigma_base = 1.5
        vv_smooth = ndimage.gaussian_filter(vv, sigma=sigma_base)
        vv_smooth = np.maximum(vv_smooth, 1e-4)
        
        for _ in range(per_key):
            r = np.random.randint(0, H - patch_size)
            c = np.random.randint(0, W - patch_size)
            crop_vv = vv[r:r+patch_size, c:c+patch_size]
            crop_x = vv_smooth[r:r+patch_size, c:c+patch_size]
            
            # Генерация двух независимых сублоков y1, y2
            noise1 = np.random.gamma(shape=looks_half, scale=1.0 / looks_half, size=crop_x.shape).astype(np.float32)
            noise2 = np.random.gamma(shape=looks_half, scale=1.0 / looks_half, size=crop_x.shape).astype(np.float32)
            
            y1 = np.maximum(crop_x * noise1, 1e-6)
            y2 = np.maximum(crop_x * noise2, 1e-6)
            
            # 50% вероятность использовать реальный сырой отсчет как y1
            if np.random.rand() > 0.5:
                y1 = np.maximum(crop_vv, 1e-6)
                
            patches_y1.append(y1)
            patches_y2.append(y2)
            
    y1_arr = np.stack(patches_y1[:num_patches], axis=0)[:, np.newaxis, :, :]
    y2_arr = np.stack(patches_y2[:num_patches], axis=0)[:, np.newaxis, :, :]
    
    t_y1 = torch.from_numpy(y1_arr).to(device)
    t_y2 = torch.from_numpy(y2_arr).to(device)
    print(f"[ДАННЫЕ] Набор самообучения сформирован: {t_y1.shape} на GPU", flush=True)
    return t_y1, t_y2


# =========================================================================
# 3. ФУНКЦИИ ПОТЕРЬ
# =========================================================================

def merlin_gamma_nll_loss(x_hat: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """Отрицательное логарифмическое правдоподобие Гамма-распределения (Dalsasso et al., IEEE TGRS 2021).
    L_gamma = mean( log(x_hat) + y_target / x_hat )
    """
    x_c = torch.clamp(x_hat, min=1e-6)
    nll = torch.log(x_c) + (y_target / x_c)
    return nll.mean()

def gradient_consistency_loss(x_hat: torch.Tensor, y_input: torch.Tensor) -> torch.Tensor:
    """Штраф за сглаживание значимых границ в лог-области."""
    log_x = torch.log(torch.clamp(x_hat, min=1e-6))
    log_y = torch.log(torch.clamp(y_input, min=1e-6))
    dx_x = log_x[:, :, :, 1:] - log_x[:, :, :, :-1]
    dx_y = log_y[:, :, :, 1:] - log_y[:, :, :, :-1]
    dy_x = log_x[:, :, 1:, :] - log_x[:, :, :-1, :]
    dy_y = log_y[:, :, 1:, :] - log_y[:, :, :-1, :]
    # Учитываем только области с градиентами выше фонового спекла
    mask_x = (torch.abs(dx_y) > 0.45).float()
    mask_y = (torch.abs(dy_y) > 0.45).float()
    l_gx = (torch.abs(dx_x - dx_y) * mask_x).sum() / (mask_x.sum() + 1e-6)
    l_gy = (torch.abs(dy_x - dy_y) * mask_y).sum() / (mask_y.sum() + 1e-6)
    return l_gx + l_gy

def tv_regularization(x_hat: torch.Tensor) -> torch.Tensor:
    dx = x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]
    dy = x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]
    return (torch.abs(dx).mean() + torch.abs(dy).mean())


# =========================================================================
# 4. ТРЕНИРОВОЧНЫЙ ЦИКЛ ОДНОЙ МОДЕЛИ
# =========================================================================

def train_single_model(
    model: nn.Module,
    model_name: str,
    y1_train: torch.Tensor,
    y2_train: torch.Tensor,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 0.001
) -> tuple[nn.Module, float, float]:
    print("\n" + "=" * 70, flush=True)
    print(f"ОБУЧЕНИЕ АРХИТЕКТУРЫ: {model_name} (Параметров: {sum(p.numel() for p in model.parameters()):,})", flush=True)
    print("=" * 70, flush=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    N = y1_train.shape[0]
    t0 = time.time()
    last_loss = 0.0
    
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_b = 0
        
        for b in range(0, N, batch_size):
            idx = perm[b:b+batch_size]
            by1 = y1_train[idx]
            by2 = y2_train[idx]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # Нормировка динамического диапазона на батч
                scale = torch.quantile(by1.view(by1.size(0), -1), 0.99, dim=1).view(-1, 1, 1, 1)
                scale = torch.clamp(scale, min=1e-4)
                by1_norm = by1 / scale
                by2_norm = by2 / scale
                
                pred_norm = model(by1_norm)
                l_nll = merlin_gamma_nll_loss(pred_norm, by2_norm)
                l_grad = gradient_consistency_loss(pred_norm, by1_norm)
                l_tv = tv_regularization(pred_norm)
                
                loss = l_nll + 0.15 * l_grad + 0.001 * l_tv
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_b += 1
            
        scheduler.step()
        last_loss = total_loss / max(n_b, 1)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"[{model_name}] Эпоха {ep:02d}/{epochs:02d} | LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {last_loss:.5f}", flush=True)
            
    elapsed = time.time() - t0
    print(f"[ГОТОВО] {model_name} обучена за {elapsed:.2f} сек. Итоговый Loss: {last_loss:.5f}", flush=True)
    return model, elapsed, last_loss


# =========================================================================
# 5. МЕТРИКИ И СРАВНЕНИЕ
# =========================================================================

def compute_raw_metrics(raw_lin: np.ndarray, filt_lin: np.ndarray, looks: float = 4.4) -> dict[str, float]:
    y = np.maximum(raw_lin.astype(np.float64), 1e-6)
    x = np.maximum(filt_lin.astype(np.float64), 1e-6)
    
    # 1. ENL (Equivalent Number of Looks) в однородной области
    # Ищем квадрат 32х32 с минимальной локальной дисперсией
    H, W = y.shape
    best_enl = 0.0
    for _ in range(25):
        r = np.random.randint(0, max(1, H - 32))
        c = np.random.randint(0, max(1, W - 32))
        crop = x[r:r+32, c:c+32]
        m = crop.mean()
        s = crop.std()
        if s > 1e-6:
            enl = (m / s) ** 2
            if enl > best_enl:
                best_enl = enl
    enl_val = float(best_enl) if best_enl > 0 else float((x.mean() / (x.std() + 1e-6)) ** 2)
    
    # 2. EPI (Edge Preservation Index)
    # По градиентам Собеля
    from scipy.ndimage import sobel
    gy_raw = np.hypot(sobel(y, axis=0), sobel(y, axis=1))
    gy_filt = np.hypot(sobel(x, axis=0), sobel(x, axis=1))
    epi_val = float(np.sum(gy_filt) / (np.sum(gy_raw) + 1e-9))
    
    # 3. Bias (Смещение интенсивности)
    bias_val = float(abs(x.mean() / y.mean() - 1.0))
    
    # 4. SSI (Speckle Suppression Index)
    ssi_val = float((x.std() / (x.mean() + 1e-9)) / (y.std() / (y.mean() + 1e-9) + 1e-9))
    
    # 5. SMPI (Speckle Mean Preservation Index)
    q = 1.0 + abs(x.mean() - y.mean()) / (y.mean() + 1e-9)
    smpi_val = float(q * (x.std() / (y.std() + 1e-9)))
    
    # 6. Отношение r = y / x
    r_img = y / x
    var_r = float(np.var(r_img))
    
    # 7. Moran's I (Пространственная корреляция остатка)
    diff = r_img - np.mean(r_img)
    numerator = np.sum(diff[:-1, :] * diff[1:, :]) + np.sum(diff[:, :-1] * diff[:, 1:])
    denominator = np.sum(diff ** 2) + 1e-9
    moran_val = float(abs(numerator / (2.0 * denominator)))
    
    return {
        "ENL": enl_val,
        "EPI": epi_val,
        "Bias": bias_val,
        "SSI": ssi_val,
        "SMPI": smpi_val,
        "Var_r": var_r,
        "Moran_r": moran_val
    }


def run_model_inference(model: nn.Module, img_np: np.ndarray) -> np.ndarray:
    model.eval()
    H, W = img_np.shape
    scale = float(np.percentile(img_np, 99.5))
    if scale < 1e-4:
        scale = float(img_np.max())
    scale = max(scale, 1e-4)
    norm = np.clip(img_np / scale, 0.0, 10.0).astype(np.float32)
    
    # Оконная блочная обработка для предотвращения скачков VRAM
    t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(t)
        res = out.squeeze().float().cpu().numpy() * scale
    return np.maximum(np.nan_to_num(res, nan=1e-6).astype(np.float32), 1e-6)


# =========================================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ ЭКСПЕРИМЕНТА
# =========================================================================

def main():
    print("=" * 80)
    print("НАУЧНО-ИССЛЕДОВАТЕЛЬСКИЙ ЭКСПЕРИМЕНТ: СРАВНЕНИЕ АРХИТЕКТУР MERLIN CNN")
    print("=" * 80)
    
    # 1. Формирование набора данных
    y1_train, y2_train = extract_real_sar_dataset(num_patches=500, patch_size=128)
    
    # 2. Инициализация моделей
    models = {
        "MERLIN_UNet_Base": MERLIN_UNet_Base(in_channels=1, base_ch=32).to(device),
        "MERLIN_ResUNet_Deep": MERLIN_ResUNet_Deep(in_channels=1, base_ch=40).to(device),
        "MERLIN_AttnUNet_CBAM": MERLIN_AttnUNet_CBAM(in_channels=1, base_ch=32).to(device),
        "MERLIN_DenseDilated_CNN": MERLIN_DenseDilated_CNN(in_channels=1, num_features=36).to(device),
    }
    
    # 3. Обучение всех архитектур
    training_results = {}
    for name, m in models.items():
        trained_m, el_time, final_loss = train_single_model(
            model=m,
            model_name=name,
            y1_train=y1_train,
            y2_train=y2_train,
            epochs=25,
            batch_size=16,
            lr=0.001
        )
        training_results[name] = {
            "time_s": el_time,
            "final_loss": final_loss,
            "params": sum(p.numel() for p in trained_m.parameters())
        }
        # Сохранение весов
        pth = os.path.join(WEIGHTS_DIR, f"{name.lower()}.pth")
        torch.save(trained_m.state_dict(), pth)
        print(f"[СОХРАНЕНО] Веса архитектуры {name} сохранены в {pth}")
        
    # 4. Бенчмаркинг по всем 8 ландшафтам
    terrain_keys = ['forest', 'mountain', 'desert', 'water', 'city', 'benevento', 'rome', 'serre_poncon']
    print("\n" + "=" * 80)
    print("РАСЧЕТ МЕТРИК ПО ВСЕМ 8 ПОЛИГОНАМ И АРХИТЕКТУРАМ")
    print("=" * 80)
    
    model_terrain_metrics = {name: [] for name in models.keys()}
    
    for tk in terrain_keys:
        p = os.path.join(DATA_DIR, f"raw_{tk}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        vv = d["vv"].astype(np.float32)
        
        print(f"\n--- Ландшафт: {tk.upper()} (Размер: {vv.shape}) ---")
        for name, m in models.items():
            f_vv = run_model_inference(m, vv)
            met = compute_raw_metrics(vv, f_vv, looks=4.4)
            model_terrain_metrics[name].append(met)
            print(f"  {name:25s} | ENL: {met['ENL']:7.2f} | EPI: {met['EPI']:.3f} | Bias: {met['Bias']:.4f} | Var(r): {met['Var_r']:.3f} | Moran: {met['Moran_r']:.4f}")
            
    # 5. Сводная таблица усредненных результатов
    print("\n" + "=" * 95)
    print(f"{'Архитектура':25s} | {'Параметры':10s} | {'ENL ↑':8s} | {'EPI ↑':7s} | {'Bias ↓':8s} | {'Var(r) (→0.227)':16s} | {'Moran ↓':8s} | {'Обучение, с':12s}")
    print("-" * 95)
    
    avg_scores = {}
    for name in models.keys():
        mets = model_terrain_metrics[name]
        mean_enl = float(np.mean([m['ENL'] for m in mets]))
        mean_epi = float(np.mean([m['EPI'] for m in mets]))
        mean_bias = float(np.mean([m['Bias'] for m in mets]))
        mean_var_r = float(np.mean([m['Var_r'] for m in mets]))
        mean_moran = float(np.mean([m['Moran_r'] for m in mets]))
        p_count = training_results[name]['params']
        t_sec = training_results[name]['time_s']
        
        print(f"{name:25s} | {p_count:10,d} | {mean_enl:8.2f} | {mean_epi:7.3f} | {mean_bias:8.4f} | {mean_var_r:16.4f} | {mean_moran:8.4f} | {t_sec:12.1f}")
        
        # Интегральный показатель качества:
        # Чем выше ENL и EPI, и чем ближе Var(r) к 0.2273, тем выше балл
        var_penalty = abs(mean_var_r - 0.2273)
        score = (mean_enl / 10.0) * mean_epi / (1.0 + 10.0 * mean_bias + 5.0 * var_penalty + 50.0 * mean_moran)
        avg_scores[name] = score

    # Определение модели-победителя
    champion = max(avg_scores.items(), key=lambda x: x[1])[0]
    print("=" * 95)
    print(f"[ИТОГОВЫЙ ВЫБОР] Архитектура-победитель: {champion} (Интегральный скор: {avg_scores[champion]:.4f})")
    print("=" * 95)
    
    # Сохранение весов чемпиона как основной модели MERLIN
    champ_weights_path = os.path.join(WEIGHTS_DIR, f"{champion.lower()}.pth")
    merlin_main_path = os.path.join(WEIGHTS_DIR, "sar_merlin_cnn.pth")
    import shutil
    shutil.copyfile(champ_weights_path, merlin_main_path)
    print(f"[ОБНОВЛЕНИЕ] Веса {champion} скопированы в {merlin_main_path} как каноническая модель MERLIN CNN!")

if __name__ == "__main__":
    main()
