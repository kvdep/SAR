# -*- coding: utf-8 -*-
"""Комплексное экспериментальное исследование глубоких нейросетевых архитектур
для фильтрации спекла в данных Sentinel-1 SAR:
1. Архитектурное семейство YOLO26 (Nano, Small, Medium, Large) с блоками C3k2/C2f, DWConv и SiLU.
2. Семейство ResNet: масштаб по глубине (5, 17, 25, 33 слоя), ширине (32, 64, 128, 256 каналов),
   абляция нормализаций (NoNorm, BatchNorm, LayerNorm) и функций активации (ReLU, GELU, SiLU).
3. Семейство MERLIN: масштаб (Nano, Small, Medium, Wide, Extreme-Deep, Extreme-Wide),
   шаги дилатации (простые числа vs экспоненциальные), функции потерь (Gamma MLE, Log-Cosh, Charbonnier).
4. Перепроектированный SAR-Restormer V2 (Nano, Small, Medium, Large):
   устранение насыщения Softplus, лог-резидуальное вычитание спекла, канальный LayerNorm,
   калиброванное многомасштабное внимание MDTA и блоки GDFN.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ИНФОРМАЦИЯ] Устройство: {device}", flush=True)
if torch.cuda.is_available():
    print(f"[ИНФОРМАЦИЯ] GPU: {torch.cuda.get_device_name(0)}", flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "temp", "data_cache")
WEIGHTS_DIR = os.path.join(BASE_DIR, "HTML Визуализация", "веса")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# =========================================================================
# 1. СТАТИСТИЧЕСКИЕ МЕТРИКИ КАЧЕСТВА ДЛЯ РАДИОЛОКАЦИОННЫХ СНИМКОВ
# =========================================================================

def compute_moran_i(residual: np.ndarray) -> float:
    """Коэффициент пространственной автокорреляции Морана I для поля остаточного спекла r = y / x_hat."""
    r = residual.astype(np.float64)
    H, W = r.shape
    mu = np.mean(r)
    z = r - mu
    ss = np.sum(z ** 2)
    if ss < 1e-12:
        return 0.0
    # Пространственные веса: 4-связное соседство
    num = (np.sum(z[:-1, :] * z[1:, :]) +
           np.sum(z[:, :-1] * z[:, 1:]))
    den = ss
    W_sum = (H - 1) * W + H * (W - 1)
    N = H * W
    I = (N / (2.0 * W_sum)) * (num / den)
    return float(I)

def evaluate_sar_metrics(raw_lin: np.ndarray, filt_lin: np.ndarray, gt_lin: np.ndarray | None = None, looks: float = 4.4) -> dict[str, float]:
    """Строгие радиолокационные метрики: ENL, EPI, Moran I, Var(r), Bias (dB), PSNR, SSIM."""
    r_safe = np.maximum(filt_lin, 1e-6)
    y_safe = np.maximum(raw_lin, 1e-6)
    
    # 1. ENL в однородной области
    # Ищем окно 32x32 с наименьшим коэффициентом вариации
    H, W = filt_lin.shape
    win = 32
    min_cv = 1e9
    best_enl = looks
    for r in range(0, H - win, win // 2):
        for c in range(0, W - win, win // 2):
            patch = filt_lin[r:r+win, c:c+win]
            m = float(patch.mean())
            s = float(patch.std())
            if m > 1e-4 and s > 1e-6:
                cv = s / m
                if cv < min_cv:
                    min_cv = cv
                    best_enl = (m / s) ** 2
    
    # 2. EPI (Индекс сохранения контуров)
    # Высокочастотный лапласиан
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    d_raw = np.abs(ndimage.convolve(y_safe, lap))
    d_flt = np.abs(ndimage.convolve(r_safe, lap))
    den_epi = np.sum(d_raw)
    epi = float(np.sum(d_flt) / (den_epi + 1e-8)) if den_epi > 0 else 1.0
    
    # 3. Отношение y / x_hat
    ratio = y_safe / r_safe
    var_ratio = float(ratio.var())
    moran_val = compute_moran_i(ratio)
    
    # 4. Радиометрическое смещение в дБ
    mu_raw = float(y_safe.mean())
    mu_flt = float(r_safe.mean())
    bias_db = float(10.0 * np.log10(max(mu_flt, 1e-6) / max(mu_raw, 1e-6)))
    
    # 5. PSNR и SSIM при наличии эталона Ground Truth
    psnr, ssim = 0.0, 0.0
    if gt_lin is not None and gt_lin.shape == filt_lin.shape:
        gt_db = 10.0 * np.log10(np.maximum(gt_lin, 1e-6))
        fl_db = 10.0 * np.log10(r_safe)
        mse = float(np.mean((gt_db - fl_db) ** 2))
        dr = float(gt_db.max() - gt_db.min())
        psnr = 10.0 * np.log10((dr ** 2) / (mse + 1e-9)) if mse > 0 else 99.9
        
        mu_x = float(np.mean(gt_db))
        mu_y = float(np.mean(fl_db))
        sig_x = float(np.var(gt_db))
        sig_y = float(np.var(fl_db))
        sig_xy = float(np.mean((gt_db - mu_x) * (fl_db - mu_y)))
        c1 = (0.01 * dr) ** 2
        c2 = (0.03 * dr) ** 2
        ssim = float(((2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)) /
                     ((mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)))
                     
    return {
        "enl": round(best_enl, 2),
        "epi": round(epi, 3),
        "var_r": round(var_ratio, 4),
        "moran_i": round(moran_val, 4),
        "bias_db": round(bias_db, 3),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 3)
    }


# =========================================================================
# 2. АРХИТЕКТУРНЫЕ МОДУЛИ
# =========================================================================

# --- МОДУЛИ YOLO26: C3k2, C2f, RepConv, DWConv ---

class ConvBNSiLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, dilation=d, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    """Глубинно-разделяемая свертка (Depthwise Separable Convolution)"""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, stride=s, padding=d * (k // 2), dilation=d, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))

class C3k2Block(nn.Module):
    """Блок C3k2 / C2f из YOLO26: расщепление градиентных путей, каскад сверток и конкатенация."""
    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        c_hidden = int(c2 * expansion)
        self.cv1 = ConvBNSiLU(c1, c_hidden * 2, 1, 1, 0)
        self.cv2 = ConvBNSiLU((2 + n) * c_hidden, c2, 1, 1, 0)
        
        self.m = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(c_hidden, c_hidden, 3, 1, 1),
                DWConv(c_hidden, c_hidden, 3, 1, 1)
            ) for _ in range(n)
        ])
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1, y2 = self.cv1(x).chunk(2, dim=1)
        layers = [y1, y2]
        cur = y2
        for block in self.m:
            cur = cur + block(cur)
            layers.append(cur)
        out = self.cv2(torch.cat(layers, dim=1))
        return x + out if self.shortcut else out

class SAR_YOLO26(nn.Module):
    """Полноразмерная сеть фильтрации спекла на базе архитектурных примитивов YOLO26.
    Работает в логарифмическом пространстве для линейной аддитивности спекла:
    ln(y) = ln(x) + ln(u).
    Сеть предсказывает спекл-компоненту r = f_theta(ln y),
    восстановление: x_hat = exp(ln y - r).
    """
    def __init__(self, in_channels: int = 1, width: int = 32, num_blocks: int = 3, use_log_residual: bool = True):
        super().__init__()
        self.use_log_residual = use_log_residual
        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, width, 3, 1, 1),
            DWConv(width, width, 3, 1, 1)
        )
        
        self.blocks = nn.ModuleList([
            C3k2Block(width, width, n=2, shortcut=True, expansion=0.5)
            for _ in range(num_blocks)
        ])
        
        # Финальный проекционный слой без BatchNorm для сохранения радиометрии
        self.head = nn.Sequential(
            ConvBNSiLU(width, width, 3, 1, 1),
            nn.Conv2d(width, in_channels, 1)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.use_log_residual:
            log_y = torch.log(torch.clamp(y, min=1e-6))
            x = self.stem(log_y)
            for b in self.blocks:
                x = b(x)
            log_residual = self.head(x)
            log_x = log_y - log_residual
            x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
            return torch.clamp(x_hat, min=1e-6)
        else:
            x = self.stem(y)
            for b in self.blocks:
                x = b(x)
            residual = self.head(x)
            return torch.clamp(y - residual, min=1e-6)


# --- СЕМЕЙСТВО RESNET И АБЛЯЦИОННЫЕ ВАРИАНТЫ ---

class SAR_ResNet_Study(nn.Module):
    """Параметризуемый ResNet для абляционных исследований:
    - глубина num_layers (5, 17, 25, 33)
    - ширина num_features (32, 64, 128, 256)
    - тип нормализации norm_type ('none', 'batch', 'layer', 'instance')
    - тип активации act_type ('relu', 'leaky', 'gelu', 'silu')
    - режим остаточной связи residual_mode ('linear_sub', 'log_sub')
    """
    def __init__(self, in_channels: int = 1, num_features: int = 64, num_layers: int = 17,
                 norm_type: str = 'batch', act_type: str = 'relu', residual_mode: str = 'linear_sub'):
        super().__init__()
        self.residual_mode = residual_mode
        self.norm_type = norm_type
        
        def get_act():
            if act_type == 'leaky': return nn.LeakyReLU(0.1, inplace=True)
            elif act_type == 'gelu': return nn.GELU()
            elif act_type == 'silu': return nn.SiLU(inplace=True)
            return nn.ReLU(inplace=True)

        def get_norm(c):
            if norm_type == 'batch': return nn.BatchNorm2d(c)
            elif norm_type == 'instance': return nn.InstanceNorm2d(c, affine=True)
            elif norm_type == 'layer': return nn.GroupNorm(1, c)
            return nn.Identity()

        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            get_act()
        ]
        
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            if norm_type != 'none':
                layers.append(get_norm(num_features))
            layers.append(get_act())
            
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)
        
        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.residual_mode == 'log_sub':
            log_y = torch.log(torch.clamp(y, min=1e-6))
            log_r = self.net(log_y)
            x_hat = torch.exp(torch.clamp(log_y - log_r, min=-14.0, max=8.0))
            return torch.clamp(x_hat, min=1e-6)
        else:
            residual = self.net(y)
            return torch.clamp(y - residual, min=1e-6)


# --- СЕМЕЙСТВО MERLIN И АБЛЯЦИЯ МАСШТАБА / ДИЛАТАЦИИ ---

class MERLIN_Scale_Study(nn.Module):
    """Исследование масштаба и дилатации MERLIN CNN:
    - num_features (16, 24, 36, 48, 64, 96, 128)
    - dilation_type ('prime', 'exponential', 'dense_unit')
    - act_fn ('leaky', 'gelu', 'silu')
    """
    def __init__(self, in_channels: int = 1, num_features: int = 36,
                 dilation_type: str = 'prime', depth: int = 6):
        super().__init__()
        self.depth = depth
        if dilation_type == 'exponential':
            dilations = [2 ** i for i in range(depth // 2)] + [2 ** (depth // 2 - 1 - i) for i in range(depth // 2)]
        elif dilation_type == 'dense_unit':
            dilations = [1] * depth
        else: # 'prime'
            prime_seq = [1, 2, 3, 5, 7, 11, 7, 5, 3, 2, 1]
            dilations = prime_seq[:depth]
            
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.d_layers = nn.ModuleList()
        for i, d in enumerate(dilations):
            in_c = num_features * (i + 1)
            self.d_layers.append(nn.Conv2d(in_c, num_features, 3, padding=d, dilation=d))
            
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_out = nn.Conv2d(num_features * (depth + 1), in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        x0 = self.act(self.conv_in(log_y))
        accum = [x0]
        cur = x0
        for layer in self.d_layers:
            inp = torch.cat(accum, dim=1)
            x_next = self.act(layer(inp))
            accum.append(x_next)
            
        cat_all = torch.cat(accum, dim=1)
        log_residual = self.conv_out(cat_all)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# --- ПЕРЕПРОЕКТИРОВАННЫЙ RESTORMER V2 ---

class MDTA_V2(nn.Module):
    """Калиброванное многоголовое транспонированное внимание (MDTA) в лог-пространстве.
    Вычисляет кросс-ковариацию каналов (C x C), а не пространственных токенов,
    что дает вычислительную сложность O(H*W*C) вместо O((H*W)^2).
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # Инициализация температуры физически обоснованным значением 1 / sqrt(d)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * (1.0 / math.sqrt(head_dim)))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Внимание по каналам: (b, num_heads, head_dim, head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)

class GDFN_V2(nn.Module):
    """Gated-Dconv Feed-Forward Network с гелевым вентилем."""
    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.dwconv(self.project_in(x))
        x1, x2 = proj.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class RestormerBlock_V2(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        # Канальный LayerNorm вместо разрушающего радиометрию пространственного GroupNorm
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MDTA_V2(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = GDFN_V2(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm в формате (B, H, W, C)
        x_ln1 = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_ln1)
        x_ln2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_ln2)
        return x

class SAR_Restormer_V2(nn.Module):
    """Исправленный SAR-Restormer V2:
    1. Логарифмический домен: ln y = ln x + ln u.
    2. Глобальная остаточная связь: ln x_hat = ln y - r_theta(ln y).
    3. Канальная нормализация LayerNorm вместо GroupNorm.
    4. Отсутствие насыщающей функции Softplus, вызывавшей сбой исходной модели.
    """
    def __init__(self, in_channels: int = 1, base_dim: int = 32, num_heads: int = 4, depth: int = 4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            RestormerBlock_V2(base_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_dim, in_channels, kernel_size=1)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        feat = self.embed(log_y)
        for blk in self.blocks:
            feat = blk(feat)
        log_residual = self.proj_out(feat)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# =========================================================================
# 3. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
# =========================================================================

def build_training_dataset(num_patches: int = 600, patch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Сборка смешанного обучающего датасета:
    - Реальные полигоны Sentinel-1 (Benevento, Rome, Serre-Poncon с Ground Truth).
    - Текстурные регионы (лес, город, горы, пустыня, море).
    - Синтетические текстуры с контролируемым L=4.4.
    """
    print(f"\n[ДАННЫЕ] Сборка датасета из {num_patches} патчей ({patch_size}x{patch_size})...", flush=True)
    noisy_list = []
    clean_list = []
    
    # 1. Извлечение пар из полигонов с многопроходным эталоном Ground Truth
    gt_scenes = ['benevento', 'rome', 'serre_poncon']
    per_gt = num_patches // 3 // len(gt_scenes)
    for sc in gt_scenes:
        p = os.path.join(DATA_DIR, f"raw_{sc}.npz")
        if not os.path.exists(p): continue
        d = np.load(p)
        vv = d["vv"].astype(np.float32)
        gt = d["vv_gt"].astype(np.float32) if "vv_gt" in d and d["vv_gt"] is not None else vv
        H, W = vv.shape
        for _ in range(per_gt):
            r = np.random.randint(0, H - patch_size)
            c = np.random.randint(0, W - patch_size)
            noisy_list.append(np.maximum(vv[r:r+patch_size, c:c+patch_size], 1e-6))
            clean_list.append(np.maximum(gt[r:r+patch_size, c:c+patch_size], 1e-6))
            
    # 2. Извлечение из текстурных полигонов (сглаженная база как приближение x)
    tex_scenes = ['forest', 'city', 'mountain', 'desert', 'water']
    per_tex = num_patches // 3 // len(tex_scenes)
    for sc in tex_scenes:
        p = os.path.join(DATA_DIR, f"raw_{sc}.npz")
        if not os.path.exists(p): continue
        d = np.load(p)
        vv = d["vv"].astype(np.float32)
        H, W = vv.shape
        x_base = ndimage.gaussian_filter(vv, sigma=1.2)
        for _ in range(per_tex):
            r = np.random.randint(0, H - patch_size)
            c = np.random.randint(0, W - patch_size)
            noisy_list.append(np.maximum(vv[r:r+patch_size, c:c+patch_size], 1e-6))
            clean_list.append(np.maximum(x_base[r:r+patch_size, c:c+patch_size], 1e-6))
            
    # 3. Синтетические градиенты и текстуры с истинным Gamma-спеклом
    remaining = num_patches - len(noisy_list)
    looks = 4.4
    for _ in range(remaining):
        y, x = np.mgrid[0:patch_size, 0:patch_size]
        freq = np.random.uniform(2.0, 10.0)
        theta = np.random.uniform(0, math.pi)
        rot_coord = (x * np.cos(theta) + y * np.sin(theta)) / patch_size
        clean = np.clip(0.04 + 0.35 * (0.5 + 0.5 * np.sin(rot_coord * freq * math.pi)), 0.01, 0.8)
        speckle = np.random.gamma(shape=looks, scale=1.0 / looks, size=clean.shape).astype(np.float32)
        noisy = clean * speckle
        noisy_list.append(noisy)
        clean_list.append(clean)
        
    arr_noisy = np.stack(noisy_list[:num_patches], axis=0)[:, np.newaxis, :, :].astype(np.float32)
    arr_clean = np.stack(clean_list[:num_patches], axis=0)[:, np.newaxis, :, :].astype(np.float32)
    
    # Храним полный датасет в системной RAM (CPU), передавая на GPU только батчи
    t_noisy = torch.from_numpy(arr_noisy)
    t_clean = torch.from_numpy(arr_clean)
    print(f"[ДАННЫЕ] Готово в RAM: Noisy {t_noisy.shape}, Clean {t_clean.shape}", flush=True)
    return t_noisy, t_clean


# =========================================================================
# 4. ТРЕНИРОВОЧНЫЙ ЦИКЛ С КОНТРОЛЕМ VRAM
# =========================================================================

def train_model_experiment(model: nn.Module, model_name: str,
                           t_noisy: torch.Tensor, t_clean: torch.Tensor,
                           epochs: int = 20, batch_size: int = 8,
                           lr: float = 0.001, loss_type: str = 'log_l1') -> tuple[nn.Module, float]:
    """Быстрый детерминированный цикл обучения в AMP FP16 с компактными батчами (batch_size=8)."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler('cuda')
    
    N = t_noisy.shape[0]
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        batches = 0
        
        for b in range(0, N, batch_size):
            idx = perm[b:b+batch_size]
            bx = t_noisy[idx].to(device, non_blocking=True)
            by = t_clean[idx].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(bx)
                
                if loss_type == 'gamma_mle':
                    pred_c = torch.clamp(pred, min=1e-6)
                    loss = (torch.log(pred_c) + (bx / pred_c)).mean()
                elif loss_type == 'log_cosh':
                    diff = torch.log(torch.clamp(pred, min=1e-6)) - torch.log(torch.clamp(by, min=1e-6))
                    loss = torch.log(torch.cosh(diff) + 1e-12).mean()
                elif loss_type == 'charbonnier':
                    diff = pred - by
                    loss = torch.sqrt(diff ** 2 + 1e-6).mean()
                else: # 'log_l1'
                    l1_lin = F.l1_loss(pred, by)
                    l1_log = F.l1_loss(torch.log(torch.clamp(pred, min=1e-6)),
                                       torch.log(torch.clamp(by, min=1e-6)))
                    loss = 0.5 * l1_lin + 0.5 * l1_log
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batches += 1
            del bx, by, pred, loss
            
        scheduler.step()
        
    elapsed = time.time() - t0
    avg_loss = epoch_loss / max(batches, 1)
    print(f"[{model_name}] Обучение завершено: {epochs} эпох за {elapsed:.1f} сек. Финальный Loss: {avg_loss:.5f}", flush=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, elapsed


# =========================================================================
# 5. ВАЛИДАЦИЯ И СРАВНЕНИЕ НА ВСЕХ 8 ПОЛИГОНАХ
# =========================================================================

def run_dl_eval(im: np.ndarray, model: nn.Module) -> np.ndarray:
    """Безопасный инференс по тайлам без импорта sar_server (исключает загрузку дублирующих моделей в VRAM)."""
    tile_size = 256
    stride = 224
    H, W = im.shape
    
    def get_starts(length: int, t_sz: int = 256, st: int = 224) -> list[int]:
        if length <= t_sz: return [0]
        starts = list(range(0, length - t_sz + 1, st))
        if starts[-1] != length - t_sz: starts.append(length - t_sz)
        return starts

    y_starts = get_starts(H, tile_size, stride)
    x_starts = get_starts(W, tile_size, stride)
    
    w_y = np.hanning(tile_size).astype(np.float32)
    w_x = np.hanning(tile_size).astype(np.float32)
    window_2d = np.outer(w_y, w_x) + 1e-4
    
    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for r in y_starts:
            for c in x_starts:
                tile = im[r:r+tile_size, c:c+tile_size].astype(np.float32)
                t_in = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
                with torch.amp.autocast('cuda'):
                    t_out = model(t_in)
                tile_out = t_out.squeeze().cpu().numpy().astype(np.float32)
                accum[r:r+tile_size, c:c+tile_size] += tile_out * window_2d
                weight[r:r+tile_size, c:c+tile_size] += window_2d
                del t_in, t_out
                
    return accum / np.maximum(weight, 1e-6)

def validate_model_on_real_terrains(model: nn.Module) -> dict[str, Any]:
    """Комплексная оценка модели на реальных сценах: Benevento (с GT), Rome (с GT), Serre-Poncon, Forest, City, Mountain, Desert, Water."""
    model.eval()
    terrains = ['benevento', 'rome', 'serre_poncon', 'forest', 'city', 'mountain', 'desert', 'water']
    metrics_per_terrain = {}
    
    all_enl = []
    all_epi = []
    all_var_r = []
    all_moran = []
    all_bias = []
    psnr_list = []
    ssim_list = []
    
    for k in terrains:
        p = os.path.join(DATA_DIR, f"raw_{k}.npz")
        if not os.path.exists(p): continue
        d = np.load(p)
        vv = d["vv"].astype(np.float32)
        gt = d["vv_gt"].astype(np.float32) if "vv_gt" in d and d["vv_gt"] is not None else None
        
        # Инференс
        t0 = time.perf_counter()
        with torch.no_grad():
            flt = run_dl_eval(vv, model)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        m = evaluate_sar_metrics(vv, flt, gt_lin=gt, looks=4.4)
        m["latency_ms"] = round(latency_ms, 1)
        metrics_per_terrain[k] = m
        
        all_enl.append(m["enl"])
        all_epi.append(m["epi"])
        all_var_r.append(m["var_r"])
        all_moran.append(m["moran_i"])
        all_bias.append(abs(m["bias_db"]))
        if m["psnr"] > 0:
            psnr_list.append(m["psnr"])
            ssim_list.append(m["ssim"])
            
    # Параметры модели
    n_params = sum(p.numel() for p in model.parameters())
    
    summary = {
        "params": n_params,
        "mean_enl": round(float(np.mean(all_enl)), 2),
        "mean_epi": round(float(np.mean(all_epi)), 3),
        "mean_var_r": round(float(np.mean(all_var_r)), 4),
        "mean_moran_i": round(float(np.mean(all_moran)), 4),
        "mean_abs_bias": round(float(np.mean(all_bias)), 3),
        "mean_psnr_gt": round(float(np.mean(psnr_list)), 2) if psnr_list else 0.0,
        "mean_ssim_gt": round(float(np.mean(ssim_list)), 3) if ssim_list else 0.0,
        "terrains": metrics_per_terrain
    }
    return summary


# =========================================================================
# 6. ГЛАВНЫЙ СЦЕНАРИЙ МАСШТАБНОГО ИССЛЕДОВАНИЯ
# =========================================================================

def main():
    print("=" * 80, flush=True)
    print("МАСШТАБНЫЙ ЭКСПЕРИМЕНТ: YOLO26, RESNET EXTREME, MERLIN SCALE, RESTORMER V2", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Датасет
    t_noisy, t_clean = build_training_dataset(num_patches=720, patch_size=128)
    
    # 2. Список экспериментов
    experiments = [
        # --- ГРУППА 1: YOLO26 ДЕСПЕКЛЕРЫ (Nano, Small, Medium, Large, LinearResidual) ---
        ("YOLO26-Nano", SAR_YOLO26(width=16, num_blocks=2, use_log_residual=True), {"lr": 0.002, "loss": "log_l1"}),
        ("YOLO26-Small", SAR_YOLO26(width=32, num_blocks=3, use_log_residual=True), {"lr": 0.0015, "loss": "log_l1"}),
        ("YOLO26-Medium", SAR_YOLO26(width=64, num_blocks=4, use_log_residual=True), {"lr": 0.001, "loss": "log_l1"}),
        ("YOLO26-Large", SAR_YOLO26(width=96, num_blocks=6, use_log_residual=True), {"lr": 0.0008, "loss": "log_l1", "batch_size": 2}),
        ("YOLO26-LinearResidual", SAR_YOLO26(width=32, num_blocks=3, use_log_residual=False), {"lr": 0.0015, "loss": "charbonnier"}),
        
        # --- ГРУППА 2: RESNET EXTREMES & ABLATIONS (Глубина, ширина, нормализация, активация) ---
        ("ResNet-Nano-5L-32C", SAR_ResNet_Study(num_features=32, num_layers=5, norm_type='batch', act_type='relu'), {"lr": 0.002, "loss": "charbonnier"}),
        ("ResNet-Vitale-17L-64C", SAR_ResNet_Study(num_features=64, num_layers=17, norm_type='batch', act_type='relu'), {"lr": 0.0015, "loss": "charbonnier"}),
        ("ResNet-Deep-25L-64C", SAR_ResNet_Study(num_features=64, num_layers=25, norm_type='batch', act_type='relu'), {"lr": 0.001, "loss": "charbonnier"}),
        ("ResNet-ExtremeDeep-33L-64C", SAR_ResNet_Study(num_features=64, num_layers=33, norm_type='batch', act_type='relu'), {"lr": 0.0008, "loss": "charbonnier"}),
        ("ResNet-Wide-17L-128C", SAR_ResNet_Study(num_features=128, num_layers=17, norm_type='batch', act_type='relu'), {"lr": 0.001, "loss": "charbonnier"}),
        ("ResNet-NoNorm-17L", SAR_ResNet_Study(num_features=64, num_layers=17, norm_type='none', act_type='relu'), {"lr": 0.001, "loss": "charbonnier"}),
        ("ResNet-SiLU-17L", SAR_ResNet_Study(num_features=64, num_layers=17, norm_type='batch', act_type='silu'), {"lr": 0.0015, "loss": "charbonnier"}),
        ("ResNet-LogResidual-17L", SAR_ResNet_Study(num_features=64, num_layers=17, norm_type='batch', act_type='leaky', residual_mode='log_sub'), {"lr": 0.0015, "loss": "log_l1"}),
        
        # --- ГРУППА 3: MERLIN SCALE & DILATION & LOSS ABLATIONS ---
        ("MERLIN-Nano-16C-4L", MERLIN_Scale_Study(num_features=16, depth=4, dilation_type='prime'), {"lr": 0.002, "loss": "log_l1"}),
        ("MERLIN-Small-24C-6L", MERLIN_Scale_Study(num_features=24, depth=6, dilation_type='prime'), {"lr": 0.0015, "loss": "log_l1"}),
        ("MERLIN-Medium-36C-6L", MERLIN_Scale_Study(num_features=36, depth=6, dilation_type='prime'), {"lr": 0.001, "loss": "log_l1"}),
        ("MERLIN-Wide-64C-6L", MERLIN_Scale_Study(num_features=64, depth=6, dilation_type='prime'), {"lr": 0.001, "loss": "log_l1"}),
        ("MERLIN-ExtremeDeep-48C-10L", MERLIN_Scale_Study(num_features=48, depth=10, dilation_type='prime'), {"lr": 0.0008, "loss": "log_l1"}),
        ("MERLIN-ExpDilation-36C", MERLIN_Scale_Study(num_features=36, depth=6, dilation_type='exponential'), {"lr": 0.001, "loss": "log_l1"}),
        ("MERLIN-DenseUnit-36C", MERLIN_Scale_Study(num_features=36, depth=6, dilation_type='dense_unit'), {"lr": 0.001, "loss": "log_l1"}),
        ("MERLIN-Loss-GammaMLE", MERLIN_Scale_Study(num_features=36, depth=6, dilation_type='prime'), {"lr": 0.001, "loss": "gamma_mle"}),
        ("MERLIN-Loss-LogCosh", MERLIN_Scale_Study(num_features=36, depth=6, dilation_type='prime'), {"lr": 0.001, "loss": "log_cosh"}),
        
        # --- ГРУППА 4: RESTORMER V2 OVERHAUL & SCALE (Устранение сбоя, канальный LayerNorm) ---
        ("RestormerV2-Nano", SAR_Restormer_V2(base_dim=16, num_heads=2, depth=2), {"lr": 0.001, "loss": "log_l1"}),
        ("RestormerV2-Small", SAR_Restormer_V2(base_dim=32, num_heads=4, depth=4), {"lr": 0.0008, "loss": "log_l1"}),
        ("RestormerV2-Medium", SAR_Restormer_V2(base_dim=48, num_heads=6, depth=6), {"lr": 0.0005, "loss": "log_l1"}),
        ("RestormerV2-Large", SAR_Restormer_V2(base_dim=64, num_heads=8, depth=8), {"lr": 0.0004, "loss": "log_l1", "batch_size": 2}),
    ]
    
    results_table = []
    
    for name, m, cfg in experiments:
        m = m.to(device)
        print(f"\n>>> Запуск эксперимента: {name} (Параметров: {sum(p.numel() for p in m.parameters()):,})", flush=True)
        bs = cfg.get("batch_size", 4)
        m, train_time = train_model_experiment(
            m, name, t_noisy, t_clean,
            epochs=15, batch_size=bs,
            lr=cfg["lr"], loss_type=cfg["loss"]
        )
        
        # Сохранение весов
        weight_file = os.path.join(WEIGHTS_DIR, f"exp_{name.lower().replace('-', '_')}.pth")
        torch.save(m.state_dict(), weight_file)
        
        # Валидация на реальных полигонах
        val_res = validate_model_on_real_terrains(m)
        val_res["name"] = name
        val_res["train_time"] = round(train_time, 1)
        val_res["weight_file"] = weight_file
        results_table.append(val_res)
        
        print(f"    [ВАЛИДАЦИЯ] ENL: {val_res['mean_enl']} | EPI: {val_res['mean_epi']} | Moran I: {val_res['mean_moran_i']} | Var(r): {val_res['mean_var_r']} | PSNR: {val_res['mean_psnr_gt']} дБ | SSIM: {val_res['mean_ssim_gt']}", flush=True)
        
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Сохранение итогового отчета в JSON и Markdown
    import json
    report_json_path = os.path.join(BASE_DIR, "massive_experiments_results.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2, ensure_ascii=False)
        
    report_md_path = os.path.join(BASE_DIR, "RESEARCH_MASSIVE_DL_EXPERIMENTS.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Исследование архитектур глубокого обучения для радиолокационных данных Sentinel-1 SAR\n\n")
        f.write("Сравнение 4 семейств моделей: YOLO26 (C3k2/DWConv/SiLU), ResNet Extreme, MERLIN Scale, Restormer V2 Overhaul.\n\n")
        f.write("## 1. Сводная таблица метрик качества\n\n")
        f.write("| Модель | Параметры | Время обуч. (с) | ENL (↑) | EPI (контуры) | Moran I (↓) | Var(r) (теор: 0.227) | Bias (дБ) | PSNR GT (дБ) | SSIM GT |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results_table:
            f.write(f"| **{r['name']}** | {r['params']:,} | {r['train_time']:.1f} | {r['mean_enl']:.2f} | {r['mean_epi']:.3f} | {r['mean_moran_i']:.4f} | {r['mean_var_r']:.4f} | {r['mean_abs_bias']:.3f} | {r['mean_psnr_gt']:.2f} | {r['mean_ssim_gt']:.3f} |\n")
        f.write("\n## 2. Детальные метрики по полигонам\n\n")
        for r in results_table:
            f.write(f"### {r['name']}\n\n")
            f.write("| Полигон | ENL | EPI | Moran I | Var(r) | Bias (дБ) | Latency (мс) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for t_name, m in r["terrains"].items():
                f.write(f"| {t_name} | {m['enl']:.2f} | {m['epi']:.3f} | {m['moran_i']:.4f} | {m['var_r']:.4f} | {m['bias_db']:.3f} | {m['latency_ms']:.1f} |\n")
            f.write("\n")
    print(f"[УСПЕХ] Отчет Markdown сохранен: {report_md_path}", flush=True)
        
    print("\n" + "=" * 80, flush=True)
    print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА ЭКСПЕРИМЕНТОВ (26 МОДЕЛЕЙ)", flush=True)
    print("=" * 80, flush=True)
    header = f"{'Модель':<28} | {'Параметры':<10} | {'ENL (выш.луч)':<12} | {'EPI':<6} | {'Moran I':<8} | {'Var(r)':<7} | {'PSNR GT':<8} | {'SSIM GT':<7}"
    print(header)
    print("-" * len(header))
    for r in results_table:
        print(f"{r['name']:<28} | {r['params']:<10,d} | {r['mean_enl']:<12.2f} | {r['mean_epi']:<6.3f} | {r['mean_moran_i']:<8.4f} | {r['mean_var_r']:<7.4f} | {r['mean_psnr_gt']:<8.2f} | {r['mean_ssim_gt']:<7.3f}")
        
    print(f"\n[УСПЕХ] Результаты сохранены в: {report_json_path}", flush=True)

if __name__ == "__main__":
    main()

