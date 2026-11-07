# -*- coding: utf-8 -*-
"""Единый модульный реестр алгоритмов фильтрации радиолокационных изображений (SAR).
Реализует паттерн Registry / Plugin Architecture для простого добавления любых новых
пространственных, вариационных, нейросетевых и матричных методов фильтрации без хардкода.

Добавление нового фильтра в проект:
    from filter_registry import register_filter

    @register_filter(
        filter_id="my_custom_filter",
        name="30. Пользовательский адаптивный фильтр",
        category="spatial",
        default_kwargs={"window_size": 7, "param_k": 1.2}
    )
    def my_custom_filter(vv, vh, window_size=7, param_k=1.2, **kwargs):
        # Обработка линейных интенсивностей двух каналов (float32 >= 0)
        out_vv = ...
        out_vh = ...
        return out_vv, out_vh
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
from PIL import Image

# Импорт конфигурации путей
try:
    from .config import WEIGHTS_DIR, WEBP_QUALITY
except ImportError:
    from config import WEIGHTS_DIR, WEBP_QUALITY


# -----------------------------------------------------------------------------
# 1. СПЕЦИФИКАЦИЯ И РЕЕСТР ФИЛЬТРОВ
# -----------------------------------------------------------------------------

class FilterSpec:
    """Метаданные и исполняемый интерфейс зарегистрированного фильтра."""
    def __init__(
        self,
        filter_id: str,
        name: str,
        category: str,
        func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        default_kwargs: Optional[Dict[str, Any]] = None,
        is_gpu: bool = False,
        description: str = "",
    ):
        self.filter_id = filter_id
        self.name = name
        self.category = category  # "raw", "spatial", "variational", "dl", "matrix", "cascade"
        self.func = func
        self.default_kwargs = default_kwargs or {}
        self.is_gpu = is_gpu
        self.description = description

    def execute(self, vv: np.ndarray, vh: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        merged_kwargs = {**self.default_kwargs, **kwargs}
        return self.func(vv, vh, **merged_kwargs)


class FilterRegistry:
    """Глобальный синглтон-реестр алгоритмов фильтрации."""
    _registry: Dict[str, FilterSpec] = {}

    @classmethod
    def register(
        cls,
        filter_id: str,
        name: str,
        category: str = "spatial",
        default_kwargs: Optional[Dict[str, Any]] = None,
        is_gpu: bool = False,
        description: str = "",
    ):
        """Декоратор регистрации алгоритма фильтрации в реестре."""
        def decorator(func: Callable[..., Tuple[np.ndarray, np.ndarray]]):
            spec = FilterSpec(
                filter_id=filter_id,
                name=name,
                category=category,
                func=func,
                default_kwargs=default_kwargs,
                is_gpu=is_gpu,
                description=description,
            )
            cls._registry[filter_id] = spec
            return func
        return decorator

    @classmethod
    def get(cls, filter_id: str) -> FilterSpec:
        if filter_id not in cls._registry:
            raise KeyError(f"Неизвестный идентификатор фильтра: '{filter_id}'. Зарегистрированы: {list(cls._registry.keys())}")
        return cls._registry[filter_id]

    @classmethod
    def list_all(cls) -> Dict[str, FilterSpec]:
        return cls._registry

    @classmethod
    def get_metadata(cls) -> List[Dict[str, Any]]:
        """Возвращает структурированный список фильтров для передачи в UI / JSON."""
        res = []
        for fid, spec in cls._registry.items():
            res.append({
                "id": fid,
                "name": spec.name,
                "category": spec.category,
                "default_kwargs": spec.default_kwargs,
                "is_gpu": spec.is_gpu,
                "description": spec.description,
            })
        return res


register_filter = FilterRegistry.register


# -----------------------------------------------------------------------------
# 2. МАТЕМАТИЧЕСКИЙ РАСЧЕТ МЕТРИК КАЧЕСТВА ФИЛЬТРАЦИИ SAR
# -----------------------------------------------------------------------------

def compute_metrics(
    raw_lin: np.ndarray,
    filt_lin: np.ndarray,
    gt_lin: Optional[np.ndarray] = None,
    looks: float = 4.4
) -> Dict[str, float]:
    """Вычисляет полный вектор показателей фильтрации радиолокационных изображений.
    
    Показатели:
    1. ENL (Equivalent Number of Looks) = mu^2 / var на однородных участках
    2. EPI (Edge Preservation Index) = sum(|grad(filt)|) / sum(|grad(raw)|)
    3. Bias (дБ) = 10 * log10(mu(filt) / mu(raw))
    4. SSI (Speckle Suppression Index) = sqrt(var(filt))/mu(filt) / (sqrt(var(raw))/mu(raw))
    5. SMPI (Speckle Suppression and Mean Preservation Index) = Q * var(filt) / var(raw)
    6. Var(r) = дисперсия отношения r = raw / filt (теоретический ориентир: 1/looks ~ 0.227)
    7. PSNR, SSIM (при наличии эталона Ground Truth)
    """
    raw_c = np.maximum(raw_lin.astype(np.float64), 1e-8)
    filt_c = np.maximum(filt_lin.astype(np.float64), 1e-8)

    # 1. ENL по однородному центральному патчу
    h, w = raw_c.shape
    r0, r1 = int(h * 0.35), int(h * 0.65)
    c0, c1 = int(w * 0.35), int(w * 0.65)
    patch = filt_c[r0:r1, c0:c1]
    m_p = np.mean(patch)
    v_p = np.var(patch)
    enl = float((m_p ** 2) / max(v_p, 1e-10))

    # 2. EPI (сохранение перепадов градиента)
    sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    g_raw = np.abs(ndimage.convolve(raw_c, sobel_v)) + np.abs(ndimage.convolve(raw_c, sobel_h))
    g_filt = np.abs(ndimage.convolve(filt_c, sobel_v)) + np.abs(ndimage.convolve(filt_c, sobel_h))
    epi = float(np.sum(g_filt) / max(np.sum(g_raw), 1e-10))

    # 3. Bias (радиометрическое смещение среднего)
    mean_raw = np.mean(raw_c)
    mean_filt = np.mean(filt_c)
    bias_db = float(10.0 * np.log10(max(mean_filt, 1e-10) / max(mean_raw, 1e-10)))

    # 4. SSI (критерий подавления спекла)
    std_raw = np.std(raw_c)
    std_filt = np.std(filt_c)
    ssi = float((std_filt / max(mean_filt, 1e-10)) / max(std_raw / max(mean_raw, 1e-10), 1e-10))

    # 5. SMPI
    q = 1.0 + np.abs(mean_raw - mean_filt)
    smpi = float(q * np.var(filt_c) / max(np.var(raw_c), 1e-10))

    # 6. Дисперсия растра отношения r = I / I_hat
    ratio = raw_c / filt_c
    ratio_var = float(np.var(ratio))

    res: Dict[str, float] = {
        "enl": round(enl, 2),
        "epi": round(epi, 3),
        "bias_db": round(bias_db, 3),
        "ssi": round(ssi, 3),
        "smpi": round(smpi, 3),
        "ratio_var": round(ratio_var, 3),
        "time_ms": 0.0,
    }

    # 7. PSNR и SSIM при наличии Ground Truth
    if gt_lin is not None and gt_lin.shape == filt_lin.shape:
        gt_c = np.maximum(gt_lin.astype(np.float64), 1e-8)
        max_val = np.percentile(gt_c, 99.5)
        mse = np.mean((gt_c - filt_c) ** 2)
        psnr = float(10.0 * np.log10((max_val ** 2) / max(mse, 1e-12)))

        mu_x = np.mean(gt_c)
        mu_y = np.mean(filt_c)
        var_x = np.var(gt_c)
        var_y = np.var(filt_c)
        cov_xy = np.mean((gt_c - mu_x) * (filt_c - mu_y))

        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        ssim = float(((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) /
                     ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))

        res["psnr"] = round(psnr, 2)
        res["ssim"] = round(ssim, 3)

    return res


# -----------------------------------------------------------------------------
# 3. ФУНКЦИИ КОДИРОВАНИЯ ИЗОБРАЖЕНИЙ В BASE64 (WebP и PNG)
# -----------------------------------------------------------------------------

def array_to_webp_base64(
    arr: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str = "gray",
    quality: int = WEBP_QUALITY
) -> str:
    """Кодирует двумерный массив в base64 строку формата WebP с заданным качеством."""
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    if cmap_name == "gray":
        u8 = (norm * 255.0).astype(np.uint8)
        img = Image.fromarray(u8, mode="L")
    else:
        cmap = cm.get_cmap(cmap_name)
        rgba = cmap(norm)
        u8 = (rgba * 255.0).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rgb_composite_webp_base64(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    ratio_db: np.ndarray,
    quality: int = WEBP_QUALITY
) -> str:
    """Формирует калиброванный псевдоцветной трехканальный композит SAR: R=VV, G=VH, B=|VV-VH|."""
    r = np.clip((vv_db - (-18.0)) / (-2.0 - (-18.0)), 0.0, 1.0)
    g = np.clip((vh_db - (-24.0)) / (-8.0 - (-24.0)), 0.0, 1.0)
    b = np.clip((ratio_db - 2.0) / (14.0 - 2.0), 0.0, 1.0)

    rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=-1)
    u8 = (rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def array_to_png_base64(
    arr: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str = "gray"
) -> str:
    """Кодирует массив в PNG base64 (используется для передачи в интерактивном режиме)."""
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    if cmap_name == "gray":
        u8 = (norm * 255.0).astype(np.uint8)
        img = Image.fromarray(u8, mode="L")
    else:
        cmap = cm.get_cmap(cmap_name)
        rgba = cmap(norm)
        u8 = (rgba * 255.0).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rgb_composite_base64(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    ratio_db: np.ndarray
) -> str:
    """Формирует калиброванный RGB композит в формате PNG."""
    r = np.clip((vv_db - (-18.0)) / (-2.0 - (-18.0)), 0.0, 1.0)
    g = np.clip((vh_db - (-24.0)) / (-8.0 - (-24.0)), 0.0, 1.0)
    b = np.clip((ratio_db - 2.0) / (14.0 - 2.0), 0.0, 1.0)

    rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=-1)
    u8 = (rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# -----------------------------------------------------------------------------
# 4. АППАРАТНЫЙ ЗАМЕР ВРЕМЕНИ ВЫПОЛНЕНИЯ ФИЛЬТРА
# -----------------------------------------------------------------------------

def benchmark_filter(
    spec: FilterSpec,
    vv: np.ndarray,
    vh: np.ndarray,
    n_runs: int = 3,
    **kwargs
) -> float:
    """Выполняет замер времени выполнения фильтра (warmup + усреднение n_runs)."""
    # Прогревочный запуск
    try:
        spec.execute(vv, vh, **kwargs)
    except Exception:
        pass

    if torch.cuda.is_available() and spec.is_gpu:
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    for _ in range(n_runs):
        spec.execute(vv, vh, **kwargs)
        if torch.cuda.is_available() and spec.is_gpu:
            torch.cuda.synchronize()
    t_end = time.perf_counter()

    avg_ms = ((t_end - t_start) / max(n_runs, 1)) * 1000.0
    return float(round(avg_ms, 2))
