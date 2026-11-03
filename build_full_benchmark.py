# -*- coding: utf-8 -*-
"""Скрипт сборки полного бенчмарка:
8 геопривязанных полигонов (включая реальные сцены из статей: Vitale/Беневенто, Рим, MERLIN/Сер-Понсон, а также российские ландшафты)
х 16 конвейеров фильтрации
х Полный набор фильтров в эталонах статей (14-17 фильтров на каждый эталон)
х Полный набор метрик (ENL, EPI, Bias, SSI, SMPI, Var(r), Moran(r), PSNR, SSIM).
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import tempfile
import time
from typing import Any

import matplotlib.cm as cm
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
from PIL import Image
import xarray as xr

from train_sar_models import (
    SAR_ResNet,
    SAR_Restormer,
    SAR_MERLIN_CNN,
    MERLIN_AttnUNet_CBAM,
    MERLIN_ResUNet_Deep,
    MERLIN_UNet_Base,
    SAR_MERLIN_BlindSpot_Legacy,
)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.20, device=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 1. СЦЕНЫ
# =========================================================================

SCENES = {
    "benevento": {
        "title": "🇮🇹 1. Беневенто, Италия (Vitale et al. 2021, S1-lat 41.13, lon 14.75)",
        "desc": "Реальный полигон статьи Vitale et al. (IEEE JSTARS 2021) и GitHub-репозитория CNNSpeckleFilter. Реальные данные Sentinel-1 IW GRD из Google Earth Engine (8 временных проходов). Холмистый рельеф региона Кампания, оливковые рощи и историческая застройка.",
        "bbox": [14.70, 41.10, 14.80, 41.20],
        "center": [41.15, 14.75],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_benevento.npz",
    },
    "rome": {
        "title": "🇮🇹 2. Рим, Италия (Vitale et al. 2021, Рис. 1: Тибр и Ватикан)",
        "desc": "Городской полигон из статьи Vitale et al. (Рис. 1): излучина реки Тибр, Ватикан, мосты и городские кварталы Рима. 15 временных проходов Sentinel-1 IW GRD из Google Earth Engine.",
        "bbox": [12.45, 41.85, 12.55, 41.95],
        "center": [41.90, 12.50],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_rome.npz",
    },
    "serre_poncon": {
        "title": "🇫🇷 3. Озеро Сер-Понсон, Альпы (MERLIN, Dalsasso 2021, Рис. 9)",
        "desc": "Полигон статьи MERLIN (IEEE TGRS 2021, Рис. 9): высокогорное водохранилище и плотина Сер-Понсон во Французских Альпах. Контраст глади горного озера и крутых освещенных склонов.",
        "bbox": [6.30, 44.50, 6.40, 44.60],
        "center": [44.55, 6.35],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_serre_poncon.npz",
    },
    "forest": {
        "title": "🌲 4. Лес (Мещёрский национальный парк, Россия)",
        "desc": "Преобладает объемное деполяризационное рассеяние в кронах деревьев. Высокий уровень VH (от -16 до -11 дБ), умеренный VV, низкий ratio.",
        "bbox": [40.10, 55.45, 40.20, 55.55],
        "center": [55.50, 40.15],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_forest.npz",
    },
    "city": {
        "title": "🏙️ 5. Город (Москва, центр, река)",
        "desc": "Двойной отскок («стена-земля») от ортогональных граней зданий и мостов (VV > 0 дБ). Русло Москвы-реки с зеркальным отражением служит строгим маркером геометрического совмещения.",
        "bbox": [37.55, 55.70, 37.65, 55.80],
        "center": [55.75, 37.60],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_city.npz",
    },
    "water": {
        "title": "🌊 6. Водоем (Чёрное море, шельф Сочи)",
        "desc": "Зеркальное отражение радиоволн от водной поверхности в направлении от сенсора Sentinel-1. Экстремально низкие значения VV (до -28 дБ) и четкая береговая линия.",
        "bbox": [39.50, 43.40, 39.60, 43.50],
        "center": [43.45, 39.55],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_water.npz",
    },
    "desert": {
        "title": "🏜️ 7. Пустыня (Сахара, дюны Эрг-Шебби)",
        "desc": "Поверхностное рассеяние песчаных дюн и проникновение волн C-диапазона в сухой песок. Низкая шероховатость, слабый уровень кросс-поляризации VH.",
        "bbox": [-4.05, 31.10, -3.95, 31.20],
        "center": [31.15, -4.00],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_desert.npz",
    },
    "mountain": {
        "title": "🏔️ 8. Гора (Эльбрус, Кавказ)",
        "desc": "Экстремальный макрорельеф: радарные эффекты укорочения склонов (foreshortening), наложения (layover) на освещенных склонах и радиотени на противоположных.",
        "bbox": [42.40, 43.30, 42.50, 43.40],
        "center": [43.35, 42.45],
        "zoom": 13,
        "npz_file": r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_mountain.npz",
    },
}

# =========================================================================
# 2. ФИЛЬТРЫ (РАЗМЕР ОКНА w=5)
# =========================================================================

def filter_boxcar_mean(im: np.ndarray, w: int = 5) -> np.ndarray:
    return ndimage.uniform_filter(im, size=w)

def filter_median(im: np.ndarray, w: int = 5) -> np.ndarray:
    return ndimage.median_filter(im, size=w)

def filter_lee(im: np.ndarray, w: int = 5, looks: float = 4.4) -> np.ndarray:
    im_c = np.maximum(im, 1e-10)
    mu = ndimage.uniform_filter(im_c, size=w)
    sq = ndimage.uniform_filter(im_c**2, size=w)
    var = np.maximum(sq - mu**2, 1e-10)
    ci_sq = var / (mu**2 + 1e-10)
    cu_sq = 1.0 / looks
    weight = np.clip(1.0 - (cu_sq / (ci_sq + 1e-10)), 0.0, 1.0)
    return mu + weight * (im_c - mu)

def filter_enhanced_lee(im: np.ndarray, w: int = 5, looks: float = 4.4, k: float = 1.0) -> np.ndarray:
    im_c = np.maximum(im, 1e-10)
    mu = ndimage.uniform_filter(im_c, size=w)
    sq = ndimage.uniform_filter(im_c**2, size=w)
    var = np.maximum(sq - mu**2, 1e-10)
    ci = np.sqrt(var) / (mu + 1e-10)
    cu = 1.0 / np.sqrt(looks)
    cmax = np.sqrt(1.0 + 2.0 / looks)
    out = np.copy(im_c)
    out[ci <= cu] = mu[ci <= cu]
    mask_hetero = (ci > cu) & (ci < cmax)
    weight = np.exp(-k * (ci[mask_hetero] - cu) / (cmax - ci[mask_hetero] + 1e-10))
    out[mask_hetero] = mu[mask_hetero] + weight * (im_c[mask_hetero] - mu[mask_hetero])
    return out

def filter_kuan(im: np.ndarray, w: int = 5, looks: float = 4.4) -> np.ndarray:
    im_c = np.maximum(im, 1e-10)
    mu = ndimage.uniform_filter(im_c, size=w)
    sq = ndimage.uniform_filter(im_c**2, size=w)
    var = np.maximum(sq - mu**2, 1e-10)
    ci_sq = var / (mu**2 + 1e-10)
    cu_sq = 1.0 / looks
    weight = np.clip((1.0 - (cu_sq / (ci_sq + 1e-10))) / (1.0 + cu_sq), 0.0, 1.0)
    return mu + weight * (im_c - mu)

def filter_frost(im: np.ndarray, w: int = 5, damping: float = 2.0) -> np.ndarray:
    rad = w // 2
    im_c = np.maximum(im, 1e-10)
    mu = ndimage.uniform_filter(im_c, size=w)
    sq = ndimage.uniform_filter(im_c**2, size=w)
    var = np.maximum(sq - mu**2, 1e-10)
    ci = np.sqrt(var) / (mu + 1e-10)
    out = np.zeros_like(im_c)
    weight_sum = np.zeros_like(im_c)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            d = np.sqrt(dx**2 + dy**2)
            shift_im = np.roll(np.roll(im_c, dy, axis=0), dx, axis=1)
            w_kernel = np.exp(-damping * ci * d)
            out += shift_im * w_kernel
            weight_sum += w_kernel
    return out / (weight_sum + 1e-10)

def filter_enhanced_frost(im: np.ndarray, w: int = 5, looks: float = 4.4, damping: float = 2.0) -> np.ndarray:
    im_c = np.maximum(im, 1e-10)
    mu = ndimage.uniform_filter(im_c, size=w)
    sq = ndimage.uniform_filter(im_c**2, size=w)
    var = np.maximum(sq - mu**2, 1e-10)
    ci = np.sqrt(var) / (mu + 1e-10)
    cu = 1.0 / np.sqrt(looks)
    cmax = np.sqrt(1.0 + 2.0 / looks)
    frost_out = filter_frost(im_c, w=w, damping=damping)
    out = np.copy(im_c)
    out[ci <= cu] = mu[ci <= cu]
    mask_hetero = (ci > cu) & (ci < cmax)
    out[mask_hetero] = frost_out[mask_hetero]
    return out

def filter_bilateral(im: np.ndarray, sigma_color: float = 0.15, sigma_space: float = 2.5, w: int = 5, d: int | None = None, sigma_s: float | None = None, sigma_r: float | None = None) -> np.ndarray:
    import cv2
    effective_d = d if d is not None else w
    if sigma_s is not None: sigma_space = float(sigma_s)
    if sigma_r is not None: sigma_color = float(sigma_r)
    scale = float(np.percentile(im, 99.0))
    if scale < 1e-6: scale = 1.0
    im_norm = (np.clip(im / scale, 0.0, 1.0) * 255.0).astype(np.float32)
    filtered = cv2.bilateralFilter(im_norm, d=effective_d, sigmaColor=sigma_color * 255.0, sigmaSpace=sigma_space)
    return (filtered / 255.0) * scale

def filter_gamma_map(im: np.ndarray, w: int = 5, looks: float = 4.4) -> np.ndarray:
    im_c = np.maximum(im, 1e-10)
    m = ndimage.uniform_filter(im_c, size=w)
    v = np.maximum(ndimage.uniform_filter(im_c**2, size=w) - m**2, 1e-10)
    ci = np.sqrt(v) / (m + 1e-9)
    cu = 1.0 / math.sqrt(looks)
    cmax = math.sqrt(2.0) * cu
    out = np.copy(im_c)
    hetero = (ci > cu) & (ci < cmax)
    if np.any(hetero):
        ci_h = ci[hetero]
        im_h = im_c[hetero]
        m_h = m[hetero]
        a = (1.0 + cu**2) / (ci_h**2 - cu**2 + 1e-9)
        t = (a - looks - 1.0) * m_h
        d = t**2 + 4.0 * a * looks * im_h * m_h
        out[hetero] = (t + np.sqrt(np.maximum(d, 0.0))) / (2.0 * a)
    out[ci <= cu] = m[ci <= cu]
    return out

# =========================================================================
# 3. ВАРИАЦИОННЫЙ МЕТОД JOINT FTV-SARP (α=1.25) [ИСПРАВЛЕННЫЙ ШАМБОЛЛЬ-ПОК]
# =========================================================================

class JointFTVSARP:
    def __init__(self, alpha: float = 1.25, K: int = 11, device: torch.device = device):
        self.alpha = alpha
        self.K = K
        self.device = device
        w = np.zeros(K, dtype=np.float32)
        w[0] = 1.0
        for k in range(1, K):
            w[k] = (1.0 - (alpha + 1.0) / k) * w[k - 1]
        w[0] = -np.sum(w[1:])
        self.w_x = torch.from_numpy(w).flip(0).view(1, 1, 1, K).to(device)
        self.w_y = torch.from_numpy(w).flip(0).view(1, 1, K, 1).to(device)
        self.w_x_adj = torch.from_numpy(w).view(1, 1, 1, K).to(device)
        self.w_y_adj = torch.from_numpy(w).view(1, 1, K, 1).to(device)
        norm_l_sq = 2.0 ** (2.0 * alpha + 1.0)
        self.tau = float(np.sqrt(0.99) / np.sqrt(norm_l_sq))
        self.sigma = float(np.sqrt(0.99) / np.sqrt(norm_l_sq))

    def grad_x(self, u: torch.Tensor) -> torch.Tensor:
        C = u.shape[1]
        pad = nn.functional.pad(u, (self.K - 1, 0, 0, 0), mode="replicate")
        return nn.functional.conv2d(pad, self.w_x.expand(C, -1, -1, -1), groups=C)

    def grad_y(self, u: torch.Tensor) -> torch.Tensor:
        C = u.shape[1]
        pad = nn.functional.pad(u, (0, 0, self.K - 1, 0), mode="replicate")
        return nn.functional.conv2d(pad, self.w_y.expand(C, -1, -1, -1), groups=C)

    def div_adj(self, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        C = px.shape[1]
        pad_x = nn.functional.pad(px, (0, self.K - 1, 0, 0), mode="constant", value=0.0)
        pad_y = nn.functional.pad(py, (0, 0, 0, self.K - 1), mode="constant", value=0.0)
        return nn.functional.conv2d(pad_x, self.w_x_adj.expand(C, -1, -1, -1), groups=C) + nn.functional.conv2d(pad_y, self.w_y_adj.expand(C, -1, -1, -1), groups=C)

    def solve(self, I_lin: torch.Tensor, n_iter: int = 25) -> torch.Tensor:
        I_clamped = torch.clamp(I_lin, min=1e-6)
        C = I_clamped.shape[1]
        mu_15 = nn.functional.avg_pool2d(I_clamped, 15, stride=1, padding=7)
        sq_15 = nn.functional.avg_pool2d(I_clamped**2, 15, stride=1, padding=7)
        sig_15 = torch.sqrt(torch.clamp(sq_15 - mu_15**2, min=1e-8))
        cfar_mask = (I_clamped[:, 0:1] > mu_15[:, 0:1] + 5.5 * sig_15[:, 0:1]) & (I_clamped[:, 0:1] > 0.5)
        cfar_mask = cfar_mask.expand(-1, C, -1, -1)

        mu_7 = nn.functional.avg_pool2d(I_clamped, 7, stride=1, padding=3)
        sq_7 = nn.functional.avg_pool2d(I_clamped**2, 7, stride=1, padding=3)
        sig_7 = torch.sqrt(torch.clamp(sq_7 - mu_7**2, min=1e-8))
        ci_map = (sig_7 / (mu_7 + 1e-8)).mean(dim=1, keepdim=True)
        Cu = 1.0 / math.sqrt(4.4)
        Cmax = math.sqrt(2.0 / 4.4)
        delta = torch.clamp((ci_map - Cu) / (Cmax - Cu + 1e-8), 0.0, 1.0)
        lam_map = (3.50 + (11.50 - 3.50) * delta).expand_as(I_clamped)

        v = torch.log(I_clamped)
        v_bar = v.clone()
        p_x = torch.zeros_like(v)
        p_y = torch.zeros_like(v)

        for _ in range(n_iter):
            gx = self.grad_x(v_bar)
            gy = self.grad_y(v_bar)
            p_x = p_x + self.sigma * gx
            p_y = p_y + self.sigma * gy
            norm = torch.sqrt(torch.clamp(p_x**2 + p_y**2, min=1e-12))
            re_weight = torch.clamp(norm, min=1.0)
            p_x = p_x / re_weight
            p_y = p_y / re_weight
            div = self.div_adj(p_x, p_y)
            z = v - self.tau * div
            step_prox = self.tau * lam_map
            v_new = z.clone()
            for _ in range(3):
                f = (v_new - z) + step_prox * (1.0 - I_clamped * torch.exp(-v_new))
                f_prime = 1.0 + step_prox * I_clamped * torch.exp(-v_new)
                v_new = v_new - f / torch.clamp(f_prime, min=1e-6)
            v_bar = 2.0 * v_new - v
            v = v_new

        R_est = torch.exp(v)
        R_est = torch.where(cfar_mask, I_clamped, R_est)
        return torch.clamp(R_est, min=1e-6)

# =========================================================================
# 4. МЕТРИКИ КАЧЕСТВА
# =========================================================================

def compute_metrics(raw_lin: np.ndarray, filt_lin: np.ndarray, gt_lin: np.ndarray | None = None, looks: float = 4.4) -> dict[str, Any]:
    from generate_advanced_comparison import compute_all_metrics as base_metrics
    m = base_metrics(raw_lin, filt_lin, looks=looks)
    if gt_lin is not None and gt_lin.shape == filt_lin.shape:
        gt_db = 10.0 * np.log10(np.maximum(gt_lin, 1e-6))
        fl_db = 10.0 * np.log10(np.maximum(filt_lin, 1e-6))
        mse = float(np.mean((gt_db - fl_db)**2))
        dr = float(gt_db.max() - gt_db.min())
        psnr = 10.0 * np.log10((dr**2) / (mse + 1e-9)) if mse > 0 else 99.9
        mu_x = float(np.mean(gt_db))
        mu_y = float(np.mean(fl_db))
        sig_x = float(np.var(gt_db))
        sig_y = float(np.var(fl_db))
        sig_xy = float(np.mean((gt_db - mu_x) * (fl_db - mu_y)))
        c1 = (0.01 * dr)**2
        c2 = (0.03 * dr)**2
        ssim = float(((2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2)))
        m["psnr"] = round(psnr, 2)
        m["ssim"] = round(ssim, 3)
    else:
        m["psnr"] = None
        m["ssim"] = None
    return m

# =========================================================================
# 5. КОДИРОВАНИЕ РАСТРОВ В BASE64
# =========================================================================

def array_to_png_base64(arr: np.ndarray, vmin: float, vmax: float, cmap_name: str | None = None) -> str:
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    if cmap_name is None:
        gray = (norm * 255).astype(np.uint8)
        img = Image.fromarray(gray, mode="L").convert("RGBA")
    else:
        cmap = cm.get_cmap(cmap_name)
        rgba = (cmap(norm) * 255).astype(np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def rgb_composite_base64(vv_db: np.ndarray, vh_db: np.ndarray, ratio_db: np.ndarray) -> str:
    r = np.clip((vv_db - (-18.0)) / (-2.0 - (-18.0)), 0.0, 1.0)
    g = np.clip((vh_db - (-24.0)) / (-8.0 - (-24.0)), 0.0, 1.0)
    b = np.clip((ratio_db - 2.0) / (14.0 - 2.0), 0.0, 1.0)
    rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=-1)
    rgb_8u = (rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb_8u, mode="RGB").convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# =========================================================================
# 6. ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ
# =========================================================================

def main():
    print("="*70)
    print("СБОРКА КОМПЛЕКСНОГО АНАЛИЗАТОРА: 8 СЦЕН x 16 ФИЛЬТРОВ + ЭТАЛОНЫ arXiv")
    print("="*70)

    weights_dir = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\weights"
    model_resnet = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device)
    model_resnet.load_state_dict(torch.load(os.path.join(weights_dir, "sar_resnet_vitale.pth"), map_location=device))
    model_resnet.eval()

    # 5 моделей семейства MERLIN
    model_mer_dense = SAR_MERLIN_CNN(in_channels=1, num_features=36).to(device)
    p_dense = os.path.join(weights_dir, "merlin_densedilated_cnn.pth")
    if not os.path.exists(p_dense):
        p_dense = os.path.join(weights_dir, "sar_merlin_cnn.pth")
    model_mer_dense.load_state_dict(torch.load(p_dense, map_location=device))
    model_mer_dense.eval()

    model_mer_cbam = MERLIN_AttnUNet_CBAM(in_channels=1, base_ch=32).to(device)
    model_mer_cbam.load_state_dict(torch.load(os.path.join(weights_dir, "merlin_attnunet_cbam.pth"), map_location=device))
    model_mer_cbam.eval()

    model_mer_res = MERLIN_ResUNet_Deep(in_channels=1, base_ch=40).to(device)
    model_mer_res.load_state_dict(torch.load(os.path.join(weights_dir, "merlin_resunet_deep.pth"), map_location=device))
    model_mer_res.eval()

    model_mer_base = MERLIN_UNet_Base(in_channels=1, base_ch=32).to(device)
    model_mer_base.load_state_dict(torch.load(os.path.join(weights_dir, "merlin_unet_base.pth"), map_location=device))
    model_mer_base.eval()

    model_mer_blind = SAR_MERLIN_BlindSpot_Legacy(in_channels=1, num_features=48).to(device)
    model_mer_blind.load_state_dict(torch.load(os.path.join(weights_dir, "sar_merlin_blindspot.pth"), map_location=device))
    model_mer_blind.eval()

    model_restormer = SAR_Restormer(in_channels=1, base_dim=32).to(device)
    model_restormer.load_state_dict(torch.load(os.path.join(weights_dir, "sar_restormer.pth"), map_location=device))
    model_restormer.eval()

    solver_ftv = JointFTVSARP(alpha=1.25, K=11, device=device)

    from sar_server import run_dl as run_dl_inference

    app_data = {}

    from sar_server import run_rpca_ialm

    # ---------------------------------------------------------------------
    # ОБРАБОТКА 8 ГЕОПРИВЯЗАННЫХ СЦЕН
    # ---------------------------------------------------------------------
    for key, sc in SCENES.items():
        print(f"\n[INFO] Обработка полигона: {sc['title']}...")

        if "npz_file" in sc and os.path.exists(sc["npz_file"]):
            # Прямая загрузка истинных однопроходных данных со спеклом
            npz_data = np.load(sc["npz_file"])
            vv_raw = npz_data["vv"]
            vh_raw = npz_data["vh"]
            bbox = list(npz_data["bbox"])
            vv_gt = npz_data["vv_gt"] if ("vv_gt" in npz_data and npz_data["vv_gt"] is not None and getattr(npz_data["vv_gt"], "ndim", 0) > 0) else None
            vh_gt = npz_data["vh_gt"] if ("vh_gt" in npz_data and npz_data["vh_gt"] is not None and getattr(npz_data["vh_gt"], "ndim", 0) > 0) else None
            has_gt = vv_gt is not None
        else:
            nc_path = sc["nc_file"]
            if not os.path.exists(nc_path):
                print(f"  [ПРЕДУПРЕЖДЕНИЕ] Файл {nc_path} не найден, пропуск.")
                continue

            tmp_nc = os.path.join(tempfile.gettempdir(), f"load_{key}_{os.urandom(4).hex()}.nc")
            shutil.copy2(nc_path, tmp_nc)
            with xr.open_dataset(tmp_nc) as ds:
                vv_all = ds["VV"].values
                vh_all = ds["VH"].values
                if np.nanmean(vv_all) < 0.0:
                    vv_lin_all = 10.0 ** (vv_all / 10.0)
                    vh_lin_all = 10.0 ** (vh_all / 10.0)
                else:
                    vv_lin_all = vv_all
                    vh_lin_all = vh_all

                vv_raw = vv_lin_all[0]
                vh_raw = vh_lin_all[0]

                if vv_lin_all.shape[0] > 1:
                    vv_gt = np.nanmean(vv_lin_all, axis=0)
                    vh_gt = np.nanmean(vh_lin_all, axis=0)
                    has_gt = True
                else:
                    vv_gt = None
                    vh_gt = None
                    has_gt = False

            if os.path.exists(tmp_nc):
                os.remove(tmp_nc)
            bbox = sc["bbox"]

        # Выполнение симметричной фильтрации обоих каналов VV и VH
        print("  -> Запуск фильтров (w=5, симметрично VV и VH)...")
        f_boxcar = filter_boxcar_mean(vv_raw, w=5)
        vh_boxcar = filter_boxcar_mean(vh_raw, w=5)

        f_median = filter_median(vv_raw, w=5)
        vh_median = filter_median(vh_raw, w=5)

        f_lee = filter_lee(vv_raw, w=5, looks=4.4)
        vh_lee = filter_lee(vh_raw, w=5, looks=4.4)

        f_enh_lee = filter_enhanced_lee(vv_raw, w=5, looks=4.4)
        vh_enh_lee = filter_enhanced_lee(vh_raw, w=5, looks=4.4)

        f_kuan = filter_kuan(vv_raw, w=5, looks=4.4)
        vh_kuan = filter_kuan(vh_raw, w=5, looks=4.4)

        f_frost = filter_frost(vv_raw, w=5)
        vh_frost = filter_frost(vh_raw, w=5)

        f_enh_frost = filter_enhanced_frost(vv_raw, w=5, looks=4.4)
        vh_enh_frost = filter_enhanced_frost(vh_raw, w=5, looks=4.4)

        f_bilateral = filter_bilateral(vv_raw)
        vh_bilateral = filter_bilateral(vh_raw)

        f_gamma = filter_gamma_map(vv_raw, w=5, looks=4.4)
        vh_gamma = filter_gamma_map(vh_raw, w=5, looks=4.4)

        # Joint FTV-SARP
        I_torch = torch.from_numpy(np.stack([vv_raw, vh_raw], axis=0)).unsqueeze(0).to(device)
        with torch.no_grad():
            R_ftv = solver_ftv.solve(I_torch, n_iter=25)
        f_ftv = R_ftv[0, 0].cpu().numpy()
        vh_ftv = R_ftv[0, 1].cpu().numpy()

        # DL inference (симметрично для обоих каналов)
        f_resnet = run_dl_inference(vv_raw, model_resnet)
        vh_resnet = run_dl_inference(vh_raw, model_resnet)

        f_mer_dense = run_dl_inference(vv_raw, model_mer_dense)
        vh_mer_dense = run_dl_inference(vh_raw, model_mer_dense)

        f_mer_cbam = run_dl_inference(vv_raw, model_mer_cbam)
        vh_mer_cbam = run_dl_inference(vh_raw, model_mer_cbam)

        f_mer_res = run_dl_inference(vv_raw, model_mer_res)
        vh_mer_res = run_dl_inference(vh_raw, model_mer_res)

        f_mer_base = run_dl_inference(vv_raw, model_mer_base)
        vh_mer_base = run_dl_inference(vh_raw, model_mer_base)

        f_mer_blind = run_dl_inference(vv_raw, model_mer_blind)
        vh_mer_blind = run_dl_inference(vh_raw, model_mer_blind)

        f_restormer = run_dl_inference(vv_raw, model_restormer)
        vh_restormer = run_dl_inference(vh_raw, model_restormer)

        # Каскадные конвейеры из сетки 1884 тестов
        f_ftv_resnet = run_dl_inference(f_ftv, model_resnet)
        vh_ftv_resnet = run_dl_inference(vh_ftv, model_resnet)

        f_ftv_bilateral = filter_bilateral(f_ftv)
        vh_ftv_bilateral = filter_bilateral(vh_ftv)

        t_bil_vv = torch.from_numpy(f_bilateral.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        t_bil_vh = torch.from_numpy(vh_bilateral.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        t_bil_joint = torch.cat([t_bil_vv, t_bil_vh], dim=1)
        with torch.no_grad():
            R_bil_ftv = solver_ftv.solve(t_bil_joint, n_iter=15)
        f_bil_ftv = R_bil_ftv[0, 0].cpu().numpy()
        vh_bil_ftv = R_bil_ftv[0, 1].cpu().numpy()
        f_bil_ftv_bil = filter_bilateral(f_bil_ftv)
        vh_bil_ftv_bil = filter_bilateral(vh_bil_ftv)

        # Методы НИР: Robust PCA
        L_vv, S_vv = run_rpca_ialm(vv_raw, max_iter=12)
        L_vh, S_vh = run_rpca_ialm(vh_raw, max_iter=12)
        S_vv_abs = np.maximum(np.abs(S_vv), 1e-6)
        S_vh_abs = np.maximum(np.abs(S_vh), 1e-6)

        filter_outputs = {
            "raw": ("0. Входной сигнал со спеклом (Raw RTC)", vv_raw, vh_raw),
            "boxcar": ("1. Mean / Boxcar 5x5", f_boxcar, vh_boxcar),
            "median": ("2. Median 5x5", f_median, vh_median),
            "lee": ("3. Классический Lee 5x5", f_lee, vh_lee),
            "enh_lee": ("4. Enhanced Lee 5x5", f_enh_lee, vh_enh_lee),
            "kuan": ("5. Kuan 5x5", f_kuan, vh_kuan),
            "frost": ("6. Frost 5x5", f_frost, vh_frost),
            "enh_frost": ("7. Enhanced Frost 5x5", f_enh_frost, vh_enh_frost),
            "bilateral": ("8. Bilateral Filter", f_bilateral, vh_bilateral),
            "gamma": ("9. Gamma MAP 5x5", f_gamma, vh_gamma),
            "ftv": ("10. Новый: Joint FTV-SARP (α=1.25)", f_ftv, vh_ftv),
            "resnet": ("11. ML: SAR-ResNet (Vitale)", f_resnet, vh_resnet),
            "merlin": ("12. ML: MERLIN Dense-Dilated (Чемпион, Dalsasso)", f_mer_dense, vh_mer_dense),
            "merlin_dense": ("12a. ML: MERLIN Dense-Dilated (Без потери разрешения)", f_mer_dense, vh_mer_dense),
            "merlin_cbam": ("12b. ML: MERLIN Dual-Attention (CBAM)", f_mer_cbam, vh_mer_cbam),
            "merlin_res": ("12c. ML: MERLIN Deep Res-U-Net", f_mer_res, vh_mer_res),
            "merlin_base": ("12d. ML: MERLIN U-Net Base (Dalsasso)", f_mer_base, vh_mer_base),
            "merlin_blind": ("12e. ML: MERLIN Blind-Spot (Legacy Softplus)", f_mer_blind, vh_mer_blind),
            "restormer": ("13. ML: SAR-Restormer (Perera)", f_restormer, vh_restormer),
            "ftv_resnet": ("14. Каскад (2 эт.): FTV-SARP → SAR-ResNet (Парето)", f_ftv_resnet, vh_ftv_resnet),
            "ftv_bilateral": ("15. Каскад (2 эт.): FTV-SARP → Bilateral", f_ftv_bilateral, vh_ftv_bilateral),
            "bil_ftv_bil": ("16. Каскад (3 эт.): Bilateral → FTV → Bilateral", f_bil_ftv_bil, vh_bil_ftv_bil),
            "rpca": ("18. НИР: Robust PCA (Low-Rank L, IALM)", L_vv, L_vh),
            "rpca_s": ("19. НИР: Robust PCA (Sparse S, цели и корабли)", S_vv_abs, S_vh_abs),
        }

        if has_gt:
            filter_outputs["multitemp_gt"] = ("17. GEE Multi-Temporal Ground Truth (Эталон)", vv_gt, vh_gt)

        pipe_dict = {}
        for p_id, (p_name, out_vv, out_vh) in filter_outputs.items():
            vv_db = 10.0 * np.log10(np.maximum(out_vv, 1e-6))
            vh_db = 10.0 * np.log10(np.maximum(out_vh, 1e-6))
            ratio_db = vv_db - vh_db
            res_ratio = vv_raw / np.maximum(out_vv, 1e-8) if p_id != "raw" else np.ones_like(vv_raw)

            m = compute_metrics(vv_raw, out_vv, gt_lin=vv_gt, looks=4.4)

            pipe_dict[p_id] = {
                "name": p_name,
                "metrics": m,
                "layers": {
                    "vv": array_to_png_base64(vv_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_db, vh_db, ratio_db),
                    "res_ratio": array_to_png_base64(res_ratio, 0.0, 2.5, cmap_name="coolwarm"),
                }
            }

        leaflet_bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
        app_data[key] = {
            "title": sc["title"],
            "desc": sc["desc"],
            "center": sc["center"],
            "zoom": sc["zoom"],
            "bounds": leaflet_bounds,
            "shape": list(vv_raw.shape),
            "pipelines": pipe_dict,
        }
        print(f"  [ГОТОВО] Полигон {key}: {len(pipe_dict)} фильтров сгенерировано.")

    # ---------------------------------------------------------------------
    # ОБРАБОТКА ЭТАЛОНОВ ИЗ СТАТЕЙ (ПОЛНЫЙ ПЕРЕБОР ВСЕХ 14-17 МЕТОДОВ)
    # ---------------------------------------------------------------------
    print("\n[INFO] Обработка эталонов статей arXiv (запуск полного набора фильтров)...")
    paper_results_dir = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\paper_results"

    def load_b64(fname: str) -> str:
        p = os.path.join(paper_results_dir, fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                ext = "jpeg" if fname.endswith(".jpeg") or fname.endswith(".jpg") else "png"
                return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode("utf-8")
        return ""

    def process_bench_case(bench_id: str, paper_name: str, title: str, desc: str, strip_fname: str,
                           input_fname: str, paper_outputs: dict[str, tuple[str, str]],
                           gt_fname: str | None = None) -> dict[str, Any]:
        p_in = os.path.join(paper_results_dir, input_fname)
        im_in_u8 = Image.open(p_in).convert("L")
        im_in = np.array(im_in_u8).astype(np.float32) / 255.0

        if gt_fname and os.path.exists(os.path.join(paper_results_dir, gt_fname)):
            im_gt_u8 = Image.open(os.path.join(paper_results_dir, gt_fname)).convert("L")
            im_gt = np.array(im_gt_u8).astype(np.float32) / 255.0
            if im_gt.shape != im_in.shape:
                im_gt = None
        else:
            im_gt = None

        def arr_to_b64(arr_2d: np.ndarray) -> str:
            norm = np.clip(arr_2d, 0.0, 1.0)
            u8 = (norm * 255.0).astype(np.uint8)
            img = Image.fromarray(u8, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        # 1. Классические фильтры
        f_boxcar = filter_boxcar_mean(im_in, w=5)
        f_median = filter_median(im_in, w=5)
        f_lee = filter_lee(im_in, w=5, looks=4.4)
        f_enh_lee = filter_enhanced_lee(im_in, w=5, looks=4.4)
        f_kuan = filter_kuan(im_in, w=5, looks=4.4)
        f_frost = filter_frost(im_in, w=5, damping=2.0)
        f_enh_frost = filter_enhanced_frost(im_in, w=5, looks=4.4, damping=2.0)
        f_bilateral = filter_bilateral(im_in)
        f_gamma = filter_gamma_map(im_in, w=5, looks=4.4)

        # 2. Joint FTV-SARP
        t_in = torch.from_numpy(im_in).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            f_ftv = solver_ftv.solve(t_in)[0, 0].cpu().numpy()

        # 3. DL модели
        f_resnet = run_dl_inference(im_in, model_resnet)
        f_mer_dense = run_dl_inference(im_in, model_mer_dense)
        f_mer_cbam = run_dl_inference(im_in, model_mer_cbam)
        f_mer_res = run_dl_inference(im_in, model_mer_res)
        f_mer_base = run_dl_inference(im_in, model_mer_base)
        f_mer_blind = run_dl_inference(im_in, model_mer_blind)
        f_restormer = run_dl_inference(im_in, model_restormer)

        pairs_dict = {
            "noisy": {
                "name": "0. Входной снимок со спеклом",
                "img": arr_to_b64(im_in),
                "metrics": compute_metrics(im_in, im_in, gt_lin=im_gt, looks=4.4)
            },
            "boxcar": {
                "name": "1. Mean / Boxcar 5x5",
                "img": arr_to_b64(f_boxcar),
                "metrics": compute_metrics(im_in, f_boxcar, gt_lin=im_gt, looks=4.4)
            },
            "median": {
                "name": "2. Median 5x5",
                "img": arr_to_b64(f_median),
                "metrics": compute_metrics(im_in, f_median, gt_lin=im_gt, looks=4.4)
            },
            "lee": {
                "name": "3. Классический Lee 5x5",
                "img": arr_to_b64(f_lee),
                "metrics": compute_metrics(im_in, f_lee, gt_lin=im_gt, looks=4.4)
            },
            "enh_lee": {
                "name": "4. Enhanced Lee 5x5",
                "img": arr_to_b64(f_enh_lee),
                "metrics": compute_metrics(im_in, f_enh_lee, gt_lin=im_gt, looks=4.4)
            },
            "kuan": {
                "name": "5. Kuan 5x5",
                "img": arr_to_b64(f_kuan),
                "metrics": compute_metrics(im_in, f_kuan, gt_lin=im_gt, looks=4.4)
            },
            "frost": {
                "name": "6. Frost 5x5",
                "img": arr_to_b64(f_frost),
                "metrics": compute_metrics(im_in, f_frost, gt_lin=im_gt, looks=4.4)
            },
            "enh_frost": {
                "name": "7. Enhanced Frost 5x5",
                "img": arr_to_b64(f_enh_frost),
                "metrics": compute_metrics(im_in, f_enh_frost, gt_lin=im_gt, looks=4.4)
            },
            "bilateral": {
                "name": "8. Bilateral Filter",
                "img": arr_to_b64(f_bilateral),
                "metrics": compute_metrics(im_in, f_bilateral, gt_lin=im_gt, looks=4.4)
            },
            "gamma": {
                "name": "9. Gamma MAP 5x5",
                "img": arr_to_b64(f_gamma),
                "metrics": compute_metrics(im_in, f_gamma, gt_lin=im_gt, looks=4.4)
            },
            "ftv": {
                "name": "10. Новый: Joint FTV-SARP (α=1.25)",
                "img": arr_to_b64(f_ftv),
                "metrics": compute_metrics(im_in, f_ftv, gt_lin=im_gt, looks=4.4)
            },
            "resnet": {
                "name": "11. ML: SAR-ResNet (Vitale)",
                "img": arr_to_b64(f_resnet),
                "metrics": compute_metrics(im_in, f_resnet, gt_lin=im_gt, looks=4.4)
            },
            "merlin": {
                "name": "12. ML: MERLIN Dense-Dilated (Чемпион, Dalsasso)",
                "img": arr_to_b64(f_mer_dense),
                "metrics": compute_metrics(im_in, f_mer_dense, gt_lin=im_gt, looks=4.4)
            },
            "merlin_dense": {
                "name": "12a. ML: MERLIN Dense-Dilated (Без потери разрешения)",
                "img": arr_to_b64(f_mer_dense),
                "metrics": compute_metrics(im_in, f_mer_dense, gt_lin=im_gt, looks=4.4)
            },
            "merlin_cbam": {
                "name": "12b. ML: MERLIN Dual-Attention (CBAM)",
                "img": arr_to_b64(f_mer_cbam),
                "metrics": compute_metrics(im_in, f_mer_cbam, gt_lin=im_gt, looks=4.4)
            },
            "merlin_res": {
                "name": "12c. ML: MERLIN Deep Res-U-Net",
                "img": arr_to_b64(f_mer_res),
                "metrics": compute_metrics(im_in, f_mer_res, gt_lin=im_gt, looks=4.4)
            },
            "merlin_base": {
                "name": "12d. ML: MERLIN U-Net Base (Dalsasso)",
                "img": arr_to_b64(f_mer_base),
                "metrics": compute_metrics(im_in, f_mer_base, gt_lin=im_gt, looks=4.4)
            },
            "merlin_blind": {
                "name": "12e. ML: MERLIN Blind-Spot (Legacy Softplus)",
                "img": arr_to_b64(f_mer_blind),
                "metrics": compute_metrics(im_in, f_mer_blind, gt_lin=im_gt, looks=4.4)
            },
            "restormer": {
                "name": "13. ML: SAR-Restormer (Perera)",
                "img": arr_to_b64(f_restormer),
                "metrics": compute_metrics(im_in, f_restormer, gt_lin=im_gt, looks=4.4)
            },
        }

        # Каскадные конвейеры для эталонов
        f_ftv_resnet = run_dl_inference(f_ftv, model_resnet)
        f_ftv_bilateral = filter_bilateral(f_ftv)
        t_bench_bil = torch.from_numpy(f_bilateral.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            f_bench_bil_ftv = solver_ftv.solve(t_bench_bil, n_iter=15)[0, 0].cpu().numpy()
        f_bench_bil_ftv_bil = filter_bilateral(f_bench_bil_ftv)

        pairs_dict["ftv_resnet"] = {
            "name": "14. Каскад (2 эт.): FTV → ResNet (Парето)",
            "img": arr_to_b64(f_ftv_resnet),
            "metrics": compute_metrics(im_in, f_ftv_resnet, gt_lin=im_gt, looks=4.4)
        }
        pairs_dict["ftv_bilateral"] = {
            "name": "15. Каскад (2 эт.): FTV → Bilateral",
            "img": arr_to_b64(f_ftv_bilateral),
            "metrics": compute_metrics(im_in, f_ftv_bilateral, gt_lin=im_gt, looks=4.4)
        }
        pairs_dict["bil_ftv_bil"] = {
            "name": "16. Каскад (3 эт.): Bilateral → FTV → Bil",
            "img": arr_to_b64(f_bench_bil_ftv_bil),
            "metrics": compute_metrics(im_in, f_bench_bil_ftv_bil, gt_lin=im_gt, looks=4.4)
        }

        # Методы НИР в эталонах
        L_bench, S_bench = run_rpca_ialm(im_in, max_iter=12)
        pairs_dict["rpca"] = {
            "name": "18. НИР: Robust PCA (Low-Rank L, IALM)",
            "img": arr_to_b64(L_bench),
            "metrics": compute_metrics(im_in, L_bench, gt_lin=im_gt, looks=4.4)
        }
        pairs_dict["rpca_s"] = {
            "name": "19. НИР: Robust PCA (Sparse S, цели)",
            "img": arr_to_b64(np.maximum(np.abs(S_bench), 1e-6)),
            "metrics": compute_metrics(im_in, np.maximum(np.abs(S_bench), 1e-6), gt_lin=im_gt, looks=4.4)
        }

        # Добавляем результаты из самой статьи
        for p_key, (p_label, p_file) in paper_outputs.items():
            full_p = os.path.join(paper_results_dir, p_file)
            if os.path.exists(full_p):
                p_im_u8 = Image.open(full_p).convert("L")
                p_arr = np.array(p_im_u8).astype(np.float32) / 255.0
                p_gt_arg = im_gt if (im_gt is not None and im_gt.shape == p_arr.shape) else None
                m_paper = compute_metrics(im_in, p_arr, gt_lin=p_gt_arg, looks=4.4)
                pairs_dict[p_key] = {
                    "name": p_label,
                    "img": load_b64(p_file),
                    "metrics": m_paper
                }

        # Если GT есть и еще не добавлен
        if gt_fname and "gt" not in pairs_dict:
            pairs_dict["gt"] = {
                "name": "Эталон: Ground Truth",
                "img": load_b64(gt_fname),
                "metrics": compute_metrics(im_in, im_gt, gt_lin=im_gt, looks=4.4) if im_gt is not None else {}
            }

        return {
            "id": bench_id,
            "paper": paper_name,
            "title": title,
            "desc": desc,
            "strip_img": load_b64(strip_fname),
            "pairs": pairs_dict,
        }

    app_data["_benchmarks"] = [
        process_bench_case(
            bench_id="vitale",
            paper_name="Vitale et al. (arXiv:2104.09350v2, IEEE JSTARS 2021)",
            title="SAR-ResNet: Тестирование на синтетических текстурах и Sentinel-1",
            desc="В статье Vitale et al. для проверки сохранения резких границ и подавления спекла при L=4 использовались классические синтетические текстуры (включая текстуру шкуры зебры из базы BSDS500 для оценки контрастных краев), а также многопроходное усреднение Sentinel-1 (50 проходов в GEE). Сравнение проводится в естественном пиксельном пространстве (1:1).",
            strip_fname="strip_vitale_resnet.png",
            input_fname="Vitale_SAR_ResNet_p8_img1_256x256.png",
            paper_outputs={
                "paper_resnet": ("Авторы: SAR-ResNet (Vitale et al.)", "Vitale_SAR_ResNet_p8_img2_256x256.png"),
                "gt": ("Эталон: Ground Truth (50 проходов GEE)", "Vitale_SAR_ResNet_p8_img0_256x256.png")
            },
            gt_fname="Vitale_SAR_ResNet_p8_img0_256x256.png"
        ),
        process_bench_case(
            bench_id="merlin",
            paper_name="Dalsasso et al. (arXiv:2110.13148v2, IEEE TGRS 2021)",
            title="MERLIN: Реальный снимок TerraSAR-X плотины Сер-Понсон во Французских Альпах",
            desc="Реальный однопроходный снимок TerraSAR-X (X-диапазон). Авторы сравнивают предложенную слепую сверточную сеть MERLIN с вейвлетным методом SAR-BM3D и моделью Speckle2Void.",
            strip_fname="strip_merlin.png",
            input_fname="MERLIN_Dalsasso_p9_img0_512x512.png",
            paper_outputs={
                "paper_merlin": ("Авторы: MERLIN Blind-Spot CNN", "MERLIN_Dalsasso_p9_img3_512x512.png"),
                "paper_bm3d": ("Базовый: SAR-BM3D (Parrilli et al.)", "MERLIN_Dalsasso_p9_img1_512x512.png"),
                "paper_s2v": ("Базовый: Speckle2Void", "MERLIN_Dalsasso_p9_img2_512x512.png")
            },
            gt_fname=None
        ),
        process_bench_case(
            bench_id="restormer",
            paper_name="Perera et al. (arXiv:2201.09355v1, IEEE 2022)",
            title="SAR-Restormer: Сравнение PPB, BM3D, CNN и трансформера на реальном SAR",
            desc="Реальный радиолокационный снимок. Авторы сравнивают классический нелокальный фильтр PPB, вейвлетный SAR-BM3D, сверточный SAR-CNN, дилатированный ID-CNN и предложенный трансформер SAR-Restormer.",
            strip_fname="strip_restormer.png",
            input_fname="Restormer_SAR_p4_img0_512x512.jpeg",
            paper_outputs={
                "paper_restormer": ("Авторы: SAR-Restormer (Perera et al.)", "Restormer_SAR_p4_img5_512x512.jpeg"),
                "paper_bm3d": ("Базовый: SAR-BM3D", "Restormer_SAR_p4_img2_512x512.jpeg"),
                "paper_cnn": ("Базовый: SAR-CNN (Chierchia et al.)", "Restormer_SAR_p4_img3_512x512.jpeg"),
                "paper_idcnn": ("Базовый: ID-CNN (Wang et al.)", "Restormer_SAR_p4_img4_512x512.jpeg"),
                "paper_ppb": ("Базовый: PPB Filter (Deledalle et al.)", "Restormer_SAR_p4_img1_512x512.jpeg")
            },
            gt_fname=None
        ),
        process_bench_case(
            bench_id="sim2real",
            paper_name="Al-Husseini et al. (arXiv:2601.01541v1, 2026)",
            title="Sim2Real: Оптический эталон vs Capella Space SAR vs SEDRUNet",
            desc="Прямое попиксельное сопоставление реального радиолокационного снимка Capella Space сверхвысокого разрешения, оптического эталона Ground Truth и результата нейросети SEDRUNet.",
            strip_fname="strip_sim2real.png",
            input_fname="Sim2Real_2026_p6_img1_308x308.png",
            paper_outputs={
                "paper_sedrunet": ("Авторы: SEDRUNet (Al-Husseini et al. 2026)", "Sim2Real_2026_p6_img2_308x308.png"),
                "gt": ("Эталон: Оптический снимок высокого разрешения", "Sim2Real_2026_p6_img0_308x308.png")
            },
            gt_fname="Sim2Real_2026_p6_img0_308x308.png"
        )
    ]

    print("\n[INFO] Сериализация данных JSON...")
    data_json = json.dumps(app_data)

    from html_template import HTML_TEMPLATE
    html_content = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    out_paths = [
        r"D:\ПРОФ\ВУЗ\ВТСК\SAR\sar_advanced_comparison.html",
        r"D:\ПРОФ\ВУЗ\ВТСК\SAR\sar_visualization.html",
        r"D:\ПРОФ\ВУЗ\ВТСК\SAR\index.html",
    ]

    for p in out_paths:
        with open(p, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[УСПЕХ] Обновлен HTML: {p} (размер: {os.path.getsize(p):,} байт)")

if __name__ == "__main__":
    main()
