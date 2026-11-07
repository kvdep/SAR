# -*- coding: utf-8 -*-
"""Полный набор алгоритмов фильтрации радиолокационных изображений,
зарегистрированных в едином реестре FilterRegistry:
- Классические пространственные адаптивные фильтры (Boxcar, Median, Lee, Enhanced Lee, Kuan, Frost, Enhanced Frost, Bilateral, Gamma MAP)
- Вариационный метод двухканальной вариации полной вариации (Joint FTV-SARP)
- Глубокие нейросетевые модели (SAR-ResNet, семейство MERLIN 5 модификаций, SAR-Restormer V2, архитектуры YOLO26)
- Матричный метод декомпозиции Robust PCA (IALM: Low-Rank L и Sparse S)
- Многостадийные гибридные каскады (FTV + ResNet, Bilateral + FTV + Bilateral)
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import WEIGHTS_DIR, VRAM_FRACTION, ROOT_DIR
    from .filter_registry import register_filter, FilterRegistry
except ImportError:
    from config import WEIGHTS_DIR, ROOT_DIR
    from filter_registry import register_filter, FilterRegistry

# Подключение модулей обучения нейросетей
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------------------------------------------------------
# 1. ИНИЦИАЛИЗАЦИЯ ВЫЧИСЛИТЕЛЬНОГО УСТРОЙСТВА
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Глобальный словарь синглтонов загруженных моделей
LOADED_MODELS: dict[str, nn.Module] = {}


def get_model(model_name: str) -> Optional[nn.Module]:
    """Ленивая загрузка весов обученных нейросетевых моделей из папки весов."""
    global LOADED_MODELS
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    from train_sar_models import (
        SAR_ResNet, SAR_MERLIN_CNN, SAR_Restormer,
        MERLIN_AttnUNet_CBAM, MERLIN_ResUNet_Deep, MERLIN_UNet_Base,
        SAR_MERLIN_BlindSpot_Legacy, SAR_YOLO26, MERLIN_Scale_Study,
        SAR_Restormer_V2
    )

    model = None
    if model_name == "resnet":
        p = os.path.join(WEIGHTS_DIR, "sar_resnet_vitale.pth")
        if os.path.exists(p):
            m = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "mer_dense":
        p = os.path.join(WEIGHTS_DIR, "sar_merlin_cnn.pth")
        if not os.path.exists(p):
            p = os.path.join(WEIGHTS_DIR, "merlin_densedilated_cnn.pth")
        if os.path.exists(p):
            m = SAR_MERLIN_CNN(in_channels=1, num_features=36).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "mer_cbam":
        p = os.path.join(WEIGHTS_DIR, "merlin_attnunet_cbam.pth")
        if os.path.exists(p):
            m = MERLIN_AttnUNet_CBAM(in_channels=1, base_ch=32).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "mer_res":
        p = os.path.join(WEIGHTS_DIR, "merlin_resunet_deep.pth")
        if os.path.exists(p):
            m = MERLIN_ResUNet_Deep(in_channels=1, base_ch=40).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "mer_base":
        p = os.path.join(WEIGHTS_DIR, "merlin_unet_base.pth")
        if os.path.exists(p):
            m = MERLIN_UNet_Base(in_channels=1, base_ch=32).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "mer_blind":
        p = os.path.join(WEIGHTS_DIR, "sar_merlin_blindspot.pth")
        if os.path.exists(p):
            m = SAR_MERLIN_BlindSpot_Legacy(in_channels=1, num_features=48).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "mer_gammamle":
        p = os.path.join(WEIGHTS_DIR, "sar_merlin_gammamle.pth")
        if os.path.exists(p):
            m = MERLIN_Scale_Study(in_channels=1, num_features=36, depth=6).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "mer_extremedeep":
        p = os.path.join(WEIGHTS_DIR, "sar_merlin_extremedeep.pth")
        if os.path.exists(p):
            m = MERLIN_Scale_Study(in_channels=1, num_features=48, depth=10).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "restormer":
        p = os.path.join(WEIGHTS_DIR, "sar_restormer_v2.pth")
        if not os.path.exists(p):
            p = os.path.join(WEIGHTS_DIR, "sar_restormer.pth")
        if os.path.exists(p):
            m = SAR_Restormer_V2(in_channels=1, base_dim=32).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "yolo26_s":
        p = os.path.join(WEIGHTS_DIR, "sar_yolo26_small.pth")
        if os.path.exists(p):
            m = SAR_YOLO26(width=32, num_blocks=3).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    elif model_name == "yolo26_n":
        p = os.path.join(WEIGHTS_DIR, "sar_yolo26_nano.pth")
        if os.path.exists(p):
            m = SAR_YOLO26(width=16, num_blocks=2).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            model = m.eval()

    if model is not None:
        LOADED_MODELS[model_name] = model
    return model


# -----------------------------------------------------------------------------
# 2. РЕГИСТРАЦИЯ 01. НЕОБРАБОТАННОГО РАДАРА
# -----------------------------------------------------------------------------

@register_filter(
    filter_id="raw",
    name="01. Необработанный радар (Raw Sentinel-1 GRD)",
    category="raw",
    default_kwargs={},
    description="Исходный калиброванный растр линейной интенсивности Sentinel-1 IW GRD."
)
def filter_raw(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return vv.copy(), vh.copy()


# -----------------------------------------------------------------------------
# 3. КЛАССИЧЕСКИЕ ПРОСТРАНСТВЕННЫЕ АДАПТИВНЫЕ ФИЛЬТРЫ
# -----------------------------------------------------------------------------

def _boxcar_1ch(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    return ndimage.uniform_filter(img.astype(np.float32), size=window_size)

@register_filter(
    filter_id="boxcar",
    name="02. Boxcar Mean 5x5",
    category="spatial",
    default_kwargs={"window_size": 5},
    description="Пространственное скользящее усреднение прямоугольным окном."
)
def filter_boxcar(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _boxcar_1ch(vv, window_size), _boxcar_1ch(vh, window_size)


def _median_1ch(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    return ndimage.median_filter(img.astype(np.float32), size=window_size)

@register_filter(
    filter_id="median",
    name="03. Median 5x5",
    category="spatial",
    default_kwargs={"window_size": 5},
    description="Ранговый медианный фильтр, устойчивый к импульсным выбросам."
)
def filter_median(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _median_1ch(vv, window_size), _median_1ch(vh, window_size)


def _lee_1ch(img: np.ndarray, window_size: int = 5, looks: float = 4.4) -> np.ndarray:
    img = img.astype(np.float32)
    cu = 1.0 / math.sqrt(max(looks, 1e-4))
    cu2 = cu * cu
    mean = ndimage.uniform_filter(img, size=window_size)
    mean_sq = ndimage.uniform_filter(img * img, size=window_size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    ci2 = var / (mean * mean + 1e-8)
    w = np.maximum(0.0, 1.0 - cu2 / np.maximum(ci2, 1e-8))
    w = np.clip(w, 0.0, 1.0)
    return mean + w * (img - mean)

@register_filter(
    filter_id="lee",
    name="04. Lee Filter 5x5",
    category="spatial",
    default_kwargs={"window_size": 5, "looks": 4.4},
    description="Классический адаптивный фильтр Ли на основе локальной линейной ММСЕ-оценки."
)
def filter_lee(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, looks: float = 4.4, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _lee_1ch(vv, window_size, looks), _lee_1ch(vh, window_size, looks)


def _enhanced_lee_1ch(img: np.ndarray, window_size: int = 5, looks: float = 4.4, damping: float = 1.0) -> np.ndarray:
    img = img.astype(np.float32)
    cu = 1.0 / math.sqrt(max(looks, 1e-4))
    cmax = math.sqrt(1.0 + 2.0 / max(looks, 1e-4))
    mean = ndimage.uniform_filter(img, size=window_size)
    mean_sq = ndimage.uniform_filter(img * img, size=window_size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    ci = np.sqrt(var) / (mean + 1e-8)
    w = np.exp(-damping * (ci - cu) / np.maximum(cmax - ci, 1e-6))
    w = np.clip(w, 0.0, 1.0)
    out = np.where(ci <= cu, mean, np.where(ci >= cmax, img, mean * w + img * (1.0 - w)))
    return out.astype(np.float32)

@register_filter(
    filter_id="enhanced_lee",
    name="05. Enhanced Lee 5x5",
    category="spatial",
    default_kwargs={"window_size": 5, "looks": 4.4, "damping": 1.0},
    description="Модифицированный фильтр Ли с тремя зонами вариации (однородная, текстурная, точечная)."
)
def filter_enhanced_lee(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, looks: float = 4.4, damping: float = 1.0, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _enhanced_lee_1ch(vv, window_size, looks, damping), _enhanced_lee_1ch(vh, window_size, looks, damping)


def _kuan_1ch(img: np.ndarray, window_size: int = 5, looks: float = 4.4) -> np.ndarray:
    img = img.astype(np.float32)
    cu2 = 1.0 / max(looks, 1e-4)
    mean = ndimage.uniform_filter(img, size=window_size)
    mean_sq = ndimage.uniform_filter(img * img, size=window_size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    ci2 = var / (mean * mean + 1e-8)
    w = np.maximum(0.0, (1.0 - cu2 / np.maximum(ci2, 1e-8)) / (1.0 + cu2))
    w = np.clip(w, 0.0, 1.0)
    return mean + w * (img - mean)

@register_filter(
    filter_id="kuan",
    name="06. Kuan Filter 5x5",
    category="spatial",
    default_kwargs={"window_size": 5, "looks": 4.4},
    description="Фильтр Куана с точным мультипликативным уравнением ММСЕ без линеаризации Тейлора."
)
def filter_kuan(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, looks: float = 4.4, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _kuan_1ch(vv, window_size, looks), _kuan_1ch(vh, window_size, looks)


def _frost_1ch(img: np.ndarray, window_size: int = 5, damping: float = 2.0) -> np.ndarray:
    img = img.astype(np.float32)
    rad = window_size // 2
    y, x = np.mgrid[-rad:rad+1, -rad:rad+1]
    dist = np.sqrt(x*x + y*y).astype(np.float32)
    mean = ndimage.uniform_filter(img, size=window_size)
    mean_sq = ndimage.uniform_filter(img * img, size=window_size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    ci = np.sqrt(var) / (mean + 1e-8)

    pad = np.pad(img, rad, mode='reflect')
    h, w = img.shape
    out = np.empty_like(img)
    for i in range(h):
        for j in range(w):
            patch = pad[i:i+window_size, j:j+window_size]
            k = damping * ci[i, j]
            weights = np.exp(-k * dist)
            out[i, j] = np.sum(patch * weights) / (np.sum(weights) + 1e-8)
    return out

@register_filter(
    filter_id="frost",
    name="07. Frost Filter 5x5",
    category="spatial",
    default_kwargs={"window_size": 5, "damping": 2.0},
    description="Экспоненциально-взвешенный адаптивный фильтр Фроста."
)
def filter_frost(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, damping: float = 2.0, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _frost_1ch(vv, window_size, damping), _frost_1ch(vh, window_size, damping)


def _enhanced_frost_1ch(img: np.ndarray, window_size: int = 5, damping: float = 2.0, looks: float = 4.4) -> np.ndarray:
    img = img.astype(np.float32)
    cu = 1.0 / math.sqrt(max(looks, 1e-4))
    cmax = math.sqrt(1.0 + 2.0 / max(looks, 1e-4))
    mean = ndimage.uniform_filter(img, size=window_size)
    mean_sq = ndimage.uniform_filter(img * img, size=window_size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    ci = np.sqrt(var) / (mean + 1e-8)

    rad = window_size // 2
    y, x = np.mgrid[-rad:rad+1, -rad:rad+1]
    dist = np.sqrt(x*x + y*y).astype(np.float32)
    pad = np.pad(img, rad, mode='reflect')
    h, w = img.shape
    out = np.empty_like(img)
    for i in range(h):
        for j in range(w):
            c = ci[i, j]
            if c <= cu:
                out[i, j] = mean[i, j]
            elif c >= cmax:
                out[i, j] = img[i, j]
            else:
                patch = pad[i:i+window_size, j:j+window_size]
                k = damping * (c - cu) / max(cmax - c, 1e-6)
                weights = np.exp(-k * dist)
                out[i, j] = np.sum(patch * weights) / (np.sum(weights) + 1e-8)
    return out

@register_filter(
    filter_id="enhanced_frost",
    name="08. Enhanced Frost 5x5",
    category="spatial",
    default_kwargs={"window_size": 5, "damping": 2.0, "looks": 4.4},
    description="Модифицированный фильтр Фроста с адаптивным зональным весом."
)
def filter_enhanced_frost(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, damping: float = 2.0, looks: float = 4.4, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _enhanced_frost_1ch(vv, window_size, damping, looks), _enhanced_frost_1ch(vh, window_size, damping, looks)


def _bilateral_1ch(img: np.ndarray, d: int = 5, sigma_color: float = 0.15, sigma_space: float = 2.5) -> np.ndarray:
    import cv2
    v_min, v_max = float(np.min(img)), float(np.percentile(img, 99.5))
    norm = np.clip((img - v_min) / max(v_max - v_min, 1e-6), 0.0, 1.0)
    filtered = cv2.bilateralFilter(norm.astype(np.float32), d, sigma_color, sigma_space)
    return filtered * (v_max - v_min) + v_min

@register_filter(
    filter_id="bilateral",
    name="09. Bilateral Filter",
    category="spatial",
    default_kwargs={"d": 5, "sigma_color": 0.15, "sigma_space": 2.5},
    description="Двусторонний пространственно-диапазонный фильтр сохранения границ."
)
def filter_bilateral(vv: np.ndarray, vh: np.ndarray, d: int = 5, sigma_color: float = 0.15, sigma_space: float = 2.5, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _bilateral_1ch(vv, d, sigma_color, sigma_space), _bilateral_1ch(vh, d, sigma_color, sigma_space)


def _gamma_map_1ch(img: np.ndarray, window_size: int = 5, looks: float = 4.4) -> np.ndarray:
    img = img.astype(np.float32)
    L = float(looks)
    cu2 = 1.0 / max(L, 1e-4)
    cmax2 = 2.0 * cu2
    mean = ndimage.uniform_filter(img, size=window_size)
    mean_sq = ndimage.uniform_filter(img * img, size=window_size)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    ci2 = var / (mean * mean + 1e-8)

    alpha = (1.0 + cu2) / np.maximum(ci2 - cu2, 1e-8)
    b = alpha - L - 1.0
    D = (mean * b) ** 2 + 4.0 * alpha * L * mean * img
    D = np.maximum(D, 0.0)
    x_hat = (mean * b + np.sqrt(D)) / (2.0 * alpha)

    out = np.where(ci2 <= cu2, mean, np.where(ci2 >= cmax2, img, x_hat))
    return np.maximum(out, 1e-8).astype(np.float32)

@register_filter(
    filter_id="gamma_map",
    name="10. Gamma MAP Filter",
    category="spatial",
    default_kwargs={"window_size": 5, "looks": 4.4},
    description="Максимальная апостериорная (MAP) байесовская оценка при гамма-распределении яркости."
)
def filter_gamma_map(vv: np.ndarray, vh: np.ndarray, window_size: int = 5, looks: float = 4.4, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return _gamma_map_1ch(vv, window_size, looks), _gamma_map_1ch(vh, window_size, looks)


# -----------------------------------------------------------------------------
# 4. ВАРИАЦИОННЫЙ МЕТОД JOINT FTV-SARP
# -----------------------------------------------------------------------------

class JointFTVSARP:
    def __init__(self, alpha: float = 1.25, K: int = 25, device=device):
        self.alpha = alpha
        self.K = K
        self.device = device

    def solve(self, u_obs: torch.Tensor, v_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = u_obs.shape[-2:]
        fy = 2 * torch.sin(torch.pi * torch.arange(H, device=self.device) / H).unsqueeze(1)
        fx = 2 * torch.sin(torch.pi * torch.arange(W, device=self.device) / W).unsqueeze(0)
        denom_base = fy**2 + fx**2

        tau = 0.125
        u = u_obs.clone()
        v = v_obs.clone()
        p1 = torch.zeros(1, 1, H, W, device=self.device)
        p2 = torch.zeros(1, 1, H, W, device=self.device)

        for _ in range(self.K):
            ux = torch.roll(u, -1, -1) - u
            uy = torch.roll(u, -1, -2) - u
            vx = torch.roll(v, -1, -1) - v
            vy = torch.roll(v, -1, -2) - v

            joint_norm = torch.sqrt(ux**2 + uy**2 + vx**2 + vy**2 + 1e-8)
            p1 = (p1 + tau * (ux + vx)) / (1.0 + tau * joint_norm)
            p2 = (p2 + tau * (uy + vy)) / (1.0 + tau * joint_norm)

            div_p = (p1 - torch.roll(p1, 1, -1)) + (p2 - torch.roll(p2, 1, -2))
            R_u = u_obs + self.alpha * div_p
            R_v = v_obs + self.alpha * div_p

            u = torch.real(torch.fft.ifft2(torch.fft.fft2(R_u) / (1.0 + self.alpha * denom_base + 1e-8)))
            v = torch.real(torch.fft.ifft2(torch.fft.fft2(R_v) / (1.0 + self.alpha * denom_base + 1e-8)))

        return torch.clamp(u, min=0.0), torch.clamp(v, min=0.0)

@register_filter(
    filter_id="joint_ftv",
    name="11. Joint FTV-SARP (alpha=1.25)",
    category="variational",
    default_kwargs={"alpha": 1.25, "K": 25},
    is_gpu=True,
    description="Вариационная совместная минимизация полной вариации двух каналов VV+VH."
)
def filter_joint_ftv(vv: np.ndarray, vh: np.ndarray, alpha: float = 1.25, K: int = 25, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    solver = JointFTVSARP(alpha=alpha, K=K, device=device)
    with torch.no_grad():
        t_vv = torch.from_numpy(vv).float().unsqueeze(0).unsqueeze(0).to(device)
        t_vh = torch.from_numpy(vh).float().unsqueeze(0).unsqueeze(0).to(device)
        o_vv, o_vh = solver.solve(t_vv, t_vh)
        out_vv = o_vv.squeeze().cpu().numpy()
        out_vh = o_vh.squeeze().cpu().numpy()
    return out_vv, out_vh


# -----------------------------------------------------------------------------
# 5. ГЛУБОКИЕ НЕЙРОСЕТЕВЫЕ МОДЕЛИ (DL)
# -----------------------------------------------------------------------------

def _run_dl_tiled(model: nn.Module, img: np.ndarray, patch_size: int = 256, overlap: int = 32) -> np.ndarray:
    """Выполняет инференс нейросети методом перекрывающихся тайлов с весовым окном Ханна."""
    if model is None:
        return img.copy()

    H, W = img.shape
    v_min, v_max = float(np.min(img)), float(np.percentile(img, 99.5))
    norm = np.clip((img - v_min) / max(v_max - v_min, 1e-6), 0.0, 1.0)

    if H <= patch_size and W <= patch_size:
        pad_h = (patch_size - H)
        pad_w = (patch_size - W)
        inp = np.pad(norm, ((0, pad_h), (0, pad_w)), mode='reflect')
        t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t).squeeze().cpu().numpy()
        res = out[:H, :W]
    else:
        step = patch_size - overlap
        y_points = list(range(0, H - patch_size + 1, step))
        if y_points[-1] != H - patch_size:
            y_points.append(H - patch_size)
        x_points = list(range(0, W - patch_size + 1, step))
        if x_points[-1] != W - patch_size:
            x_points.append(W - patch_size)

        accum = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        hann_1d = np.hanning(patch_size)
        hann_2d = np.outer(hann_1d, hann_1d) + 1e-4

        for y in y_points:
            for x in x_points:
                crop = norm[y:y+patch_size, x:x+patch_size]
                t = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(t).squeeze().cpu().numpy()
                accum[y:y+patch_size, x:x+patch_size] += out * hann_2d
                weight[y:y+patch_size, x:x+patch_size] += hann_2d

        res = accum / np.maximum(weight, 1e-8)

    denorm = res * (v_max - v_min) + v_min
    return np.maximum(denorm, 1e-8).astype(np.float32)


@register_filter(
    filter_id="resnet",
    name="12. SAR-ResNet (17L Deep Residual)",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Глубокая остаточная сверточная сеть (Vitale et al., IEEE JSTARS 2021)."
)
def filter_resnet(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("resnet")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="mer_dense",
    name="13. MERLIN Dense-Dilated (Champion)",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Флагманская сеть архитектуры 'слепого пятна' с плотными расширенными свертками."
)
def filter_merlin_dense(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("mer_dense")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="mer_cbam",
    name="14. MERLIN CBAM Attention U-Net",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Сеть MERLIN с блоками пространственного и канального внимания CBAM."
)
def filter_merlin_cbam(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("mer_cbam")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="mer_res",
    name="15. MERLIN Deep Res-U-Net",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Глубокая остаточная U-Net архитектура слепого пятна MERLIN."
)
def filter_merlin_res(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("mer_res")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="mer_base",
    name="16. MERLIN U-Net Base",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Базовая 4-уровневая U-Net архитектура MERLIN."
)
def filter_merlin_base(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("mer_base")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="mer_blind",
    name="17. MERLIN Blind-Spot Legacy",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Классическая каноническая модель MERLIN (Dalsasso et al., IEEE TGRS 2021)."
)
def filter_merlin_blind(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("mer_blind")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="mer_gammamle",
    name="18. MERLIN Gamma MLE Loss",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Сеть MERLIN, обученная по функции правдоподобия гамма-распределения спекла."
)
def filter_merlin_gammamle(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("mer_gammamle")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="mer_extremedeep",
    name="19. MERLIN Extreme-Deep 10L",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Сверхглубокая 10-слойная модель MERLIN (4.57M параметров)."
)
def filter_merlin_extremedeep(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("mer_extremedeep")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="restormer",
    name="20. SAR-Restormer V2 (Transformer)",
    category="dl",
    default_kwargs={"patch_size": 48, "overlap": 12},
    is_gpu=True,
    description="Восстанавливающий трансформер SAR с вниманием через каналы (Perera et al.)."
)
def filter_restormer(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("restormer")
    return _run_dl_tiled(m, vv, patch_size=48, overlap=12), _run_dl_tiled(m, vh, patch_size=48, overlap=12)


@register_filter(
    filter_id="yolo26_s",
    name="21. YOLO26-Small (CSPDarknet)",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Сверточная архитектура со сквозными связями CSPDarknet."
)
def filter_yolo26_s(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("yolo26_s")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


@register_filter(
    filter_id="yolo26_n",
    name="22. YOLO26-Nano (Compact)",
    category="dl",
    default_kwargs={"patch_size": 256, "overlap": 32},
    is_gpu=True,
    description="Компактная сверхбыстрая архитектура со сквозными связями."
)
def filter_yolo26_n(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    m = get_model("yolo26_n")
    return _run_dl_tiled(m, vv), _run_dl_tiled(m, vh)


# -----------------------------------------------------------------------------
# 6. МАТРИЧНЫЙ МЕТОД ROBUST PCA (IALM)
# -----------------------------------------------------------------------------

def run_rpca_ialm(D: np.ndarray, max_iter: int = 12, tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Разложение матрицы радиолокационного изображения на низкоранговую и разреженную составляющие:
    D = L (низкоранговый стационарный сигнал) + S (разреженная нестационарная компонента).
    """
    m, n = D.shape
    lam = 1.0 / math.sqrt(max(m, n))

    norm_two = np.linalg.norm(D, 2)
    norm_inf = np.max(np.abs(D)) / lam
    dual_norm = max(norm_two, norm_inf)
    Y = D / max(dual_norm, 1e-8)

    mu = 1.25 / max(norm_two, 1e-8)
    rho = 1.5

    L = np.zeros_like(D)
    S = np.zeros_like(D)
    d_norm = np.linalg.norm(D, 'fro')

    for _ in range(max_iter):
        temp_T = D - L + (1.0 / mu) * Y
        S = np.sign(temp_T) * np.maximum(np.abs(temp_T) - lam / mu, 0.0)

        temp_L = D - S + (1.0 / mu) * Y
        U, s, Vt = np.linalg.svd(temp_L, full_matrices=False)
        s_th = np.maximum(s - 1.0 / mu, 0.0)
        L = np.dot(U * s_th, Vt)

        Z = D - L - S
        Y = Y + mu * Z
        mu = mu * rho

        if (np.linalg.norm(Z, 'fro') / max(d_norm, 1e-8)) < tol:
            break

    return L.astype(np.float32), S.astype(np.float32)


@register_filter(
    filter_id="rpca_ialm",
    name="23. Robust PCA IALM (Low-Rank L)",
    category="matrix",
    default_kwargs={"max_iter": 12},
    description="Низкоранговая стационарная матричная компонента L алгоритма IALM."
)
def filter_rpca_lowrank(vv: np.ndarray, vh: np.ndarray, max_iter: int = 12, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    l_vv, _ = run_rpca_ialm(vv, max_iter=max_iter)
    l_vh, _ = run_rpca_ialm(vh, max_iter=max_iter)
    return np.maximum(np.abs(l_vv), 1e-8), np.maximum(np.abs(l_vh), 1e-8)


@register_filter(
    filter_id="rpca_sparse",
    name="24. Robust PCA IALM (Sparse S)",
    category="matrix",
    default_kwargs={"max_iter": 12},
    description="Разреженная точечная матричная компонента S алгоритма IALM."
)
def filter_rpca_sparse(vv: np.ndarray, vh: np.ndarray, max_iter: int = 12, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    _, s_vv = run_rpca_ialm(vv, max_iter=max_iter)
    _, s_vh = run_rpca_ialm(vh, max_iter=max_iter)
    return np.maximum(np.abs(s_vv), 1e-8), np.maximum(np.abs(s_vh), 1e-8)


# -----------------------------------------------------------------------------
# 7. МНОГОСТАДИЙНЫЕ ГИБРИДНЫЕ КАСКАДЫ
# -----------------------------------------------------------------------------

@register_filter(
    filter_id="cascade_ftv_resnet",
    name="25. Каскад: FTV + SAR-ResNet",
    category="cascade",
    default_kwargs={},
    is_gpu=True,
    description="Двухстадийная обработка: предварительная вариация FTV с последующим уточнением нейросетью ResNet."
)
def filter_cascade_ftv_resnet(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    ftv_vv, ftv_vh = filter_joint_ftv(vv, vh)
    m = get_model("resnet")
    return _run_dl_tiled(m, ftv_vv), _run_dl_tiled(m, ftv_vh)


@register_filter(
    filter_id="cascade_ftv_bilateral",
    name="26. Каскад: FTV + Bilateral",
    category="cascade",
    default_kwargs={},
    is_gpu=True,
    description="Двухстадийная обработка: вариационный FTV-фильтр с финишным двусторонним сглаживанием."
)
def filter_cascade_ftv_bilateral(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    ftv_vv, ftv_vh = filter_joint_ftv(vv, vh)
    return _bilateral_1ch(ftv_vv), _bilateral_1ch(ftv_vh)


@register_filter(
    filter_id="cascade_bilateral_ftv_bilateral",
    name="27. Каскад: Bilateral + FTV + Bilateral",
    category="cascade",
    default_kwargs={},
    is_gpu=True,
    description="Трехстадийный гибридный каскад: Bilateral -> FTV -> Bilateral."
)
def filter_cascade_triplet(vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    b1_vv, b1_vh = _bilateral_1ch(vv), _bilateral_1ch(vh)
    ftv_vv, ftv_vh = filter_joint_ftv(b1_vv, b1_vh)
    return _bilateral_1ch(ftv_vv), _bilateral_1ch(ftv_vh)
