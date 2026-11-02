# -*- coding: utf-8 -*-
"""Скрипт генерации сравнительного анализа Sentinel-1 SAR:
Текущий конвейер (Gamma MAP) vs Новые вариационные (Joint FTV-SARP) и
нейросетевые (RFF-DIP с вариограммным лоссом) методы.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from typing import Any

import matplotlib.cm as cm
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUADRANTS = {
    "forest": {
        "title": "Лес (Мещёрский национальный парк)",
        "desc": "Преобладает объемное рассеяние кронами деревьев. Высокий уровень VH (от -16 до -11 дБ), умеренный VV, низкий ratio.",
        "bbox": [40.10, 55.45, 40.20, 55.55],
        "center": [55.50, 40.15],
        "zoom": 13,
    },
    "water": {
        "title": "Водоем (Чёрное море)",
        "desc": "Зеркальное отражение от гладкой поверхности воды в направлении от сенсора. Экстремально низкие значения VV (до -28 дБ) и VH (до -35 дБ), высокий ratio.",
        "bbox": [39.50, 43.40, 39.60, 43.50],
        "center": [43.45, 39.55],
        "zoom": 13,
    },
    "city": {
        "title": "Город (Москва, центр)",
        "desc": "Двойной отскок от углов зданий и металлоконструкций. Максимальные значения VV (до +25 дБ) и высокая плотность стабильных точечных отражателей.",
        "bbox": [37.55, 55.70, 37.65, 55.80],
        "center": [55.75, 37.60],
        "zoom": 13,
    },
    "desert": {
        "title": "Пустыня (Сахара, дюны)",
        "desc": "Поверхностное рассеяние песчаных дюн и проникновение C-диапазона в сухой песок. Низкая шероховатость, слабый уровень кросс-поляризации VH.",
        "bbox": [-4.05, 31.10, -3.95, 31.20],
        "center": [31.15, -4.00],
        "zoom": 13,
    },
    "mountain": {
        "title": "Гора (Эльбрус, Кавказ)",
        "desc": "Экстремальный макрорельеф: радарные эффекты укорочения склонов (foreshortening), наложения (layover) на освещенных склонах и радиотени на противоположных.",
        "bbox": [42.40, 43.30, 42.50, 43.40],
        "center": [43.35, 42.45],
        "zoom": 13,
    },
}

CACHE_DIR = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache"


# =========================================================================
# 1. ТЕКУЩИЙ ПАЙПЛАЙН: GAMMA MAP
# =========================================================================

def run_gamma_map(im: np.ndarray, w: int = 3, looks: float = 5.0) -> np.ndarray:
    """Пространственный байесовский фильтр Gamma MAP (из текущего run.py)."""
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
# 2. НОВЫЙ ПАЙПЛАЙН 1: JOINT FTV-SARP GAMMA (ВАРИАЦИОННЫЙ)
# =========================================================================

class JointFTVSARP:
    """Прямо-двойственный алгоритм Шамболля-Пока (PDHG) для совместной
    дробной вариации по каналам VV и VH с отрицательным логарифмом
    правдоподобия Гамма-распределения и SARP-модуляцией.
    """
    def __init__(self, alpha: float = 1.25, K: int = 11, device: torch.device = device):
        self.alpha = alpha
        self.K = K
        self.device = device

        w = np.zeros(K, dtype=np.float32)
        w[0] = 1.0
        for k in range(1, K):
            w[k] = (1.0 - (alpha + 1.0) / k) * w[k - 1]
        w[0] = -np.sum(w[1:])  # Zero-response balance constraint: sum(w) == 0

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
        dx = nn.functional.conv2d(pad_x, self.w_x_adj.expand(C, -1, -1, -1), groups=C)
        dy = nn.functional.conv2d(pad_y, self.w_y_adj.expand(C, -1, -1, -1), groups=C)
        return dx + dy

    def solve(self, I_lin: torch.Tensor, n_iter: int = 25) -> torch.Tensor:
        I_clamped = torch.clamp(I_lin, min=1e-6)

        # 1. Строгий физический CFAR-детектор точечных целей
        mu_15 = nn.functional.avg_pool2d(I_clamped, 15, stride=1, padding=7)
        sq_15 = nn.functional.avg_pool2d(I_clamped**2, 15, stride=1, padding=7)
        sig_15 = torch.sqrt(torch.clamp(sq_15 - mu_15**2, min=1e-8))
        cfar_mask = (I_clamped[:, 0:1] > mu_15[:, 0:1] + 5.5 * sig_15[:, 0:1]) & (I_clamped[:, 0:1] > 0.5)
        cfar_mask = cfar_mask.expand(-1, 2, -1, -1)

        # 2. Адаптивная карта SARP по коэффициенту вариации CI
        mu_7 = nn.functional.avg_pool2d(I_clamped, 7, stride=1, padding=3)
        sq_7 = nn.functional.avg_pool2d(I_clamped**2, 7, stride=1, padding=3)
        sig_7 = torch.sqrt(torch.clamp(sq_7 - mu_7**2, min=1e-8))
        ci_map = (sig_7 / (mu_7 + 1e-8)).mean(dim=1, keepdim=True)
        Cu = 1.0 / math.sqrt(4.4)
        Cmax = math.sqrt(2.0 / 4.4)
        delta = torch.clamp((ci_map - Cu) / (Cmax - Cu + 1e-8), 0.0, 1.0)
        lam_map = 3.50 + 8.00 * delta

        # 3. Инициализация переменных PDHG
        v = torch.log(I_clamped)
        v_bar = v.clone()
        px = torch.zeros_like(I_clamped)
        py = torch.zeros_like(I_clamped)

        for _ in range(n_iter):
            gx = self.grad_x(v_bar)
            gy = self.grad_y(v_bar)
            t_px = px + self.sigma * gx
            t_py = py + self.sigma * gy

            joint_norm = torch.sqrt((t_px**2 + t_py**2).sum(dim=1, keepdim=True) + 1e-12)
            denom = torch.clamp(joint_norm, min=1.0)
            px = t_px / denom
            py = t_py / denom

            div = self.div_adj(px, py)
            v_tilde = v - self.tau * div
            v_old = v.clone()

            for _ in range(3):
                exp_neg_v = torch.exp(-v)
                f_val = v - v_tilde + self.tau * lam_map * (1.0 - I_clamped * exp_neg_v)
                f_prime = 1.0 + self.tau * lam_map * I_clamped * exp_neg_v
                v = v - f_val / f_prime

            v_bar = 2.0 * v - v_old

        R_pred = torch.exp(v)
        scale = I_clamped.mean(dim=(-2, -1), keepdim=True) / R_pred.mean(dim=(-2, -1), keepdim=True)
        R_pred = R_pred * scale
        R_pred[cfar_mask] = I_clamped[cfar_mask]
        return R_pred


# =========================================================================
# 3. НОВЫЙ ПАЙПЛАЙН 2: RFF-DIP GAMMA + VARIOGRAM MATCHING
# =========================================================================

class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features: int = 2, num_features: int = 16, sigma: float = 8.0):
        super().__init__()
        b_mat = torch.randn(num_features, in_features) * sigma
        self.B = nn.Parameter(b_mat, requires_grad=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * torch.einsum("nc,bchw->bnhw", self.B, coords)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)


class SARUNet(nn.Module):
    def __init__(self, in_ch: int = 32, out_ch: int = 2):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2))
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2))
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2))
        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = nn.functional.interpolate(e3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = nn.functional.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def run_rff_dip_sar(I_lin: torch.Tensor, n_iter: int = 80) -> torch.Tensor:
    _, _, H, W = I_lin.shape
    y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing="ij")
    coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)

    rff = RandomFourierFeatures(in_features=2, num_features=16, sigma=8.0).to(device)
    net = SARUNet(in_ch=32, out_ch=2).to(device)
    z = rff(coords)

    I_clamped = torch.clamp(I_lin, min=1e-6)
    optimizer = optim.AdamW(net.parameters(), lr=1.5e-3)
    psi_target = 0.255036

    for _ in range(n_iter):
        optimizer.zero_grad()
        pred_log = net(z)
        pred_R = torch.exp(pred_log)

        loss_nll = (pred_log + I_clamped / pred_R).mean()

        res_log = torch.log(I_clamped) - pred_log
        loss_vario = 0.0
        for h in [1, 2, 3]:
            diff_x = res_log[:, :, :, h:] - res_log[:, :, :, :-h]
            diff_y = res_log[:, :, h:, :] - res_log[:, :, :-h, :]
            gamma_h = 0.5 * (diff_x.pow(2).mean() + diff_y.pow(2).mean())
            loss_vario = loss_vario + (gamma_h - psi_target).pow(2)

        loss = loss_nll + 2.5 * loss_vario
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_R = torch.exp(net(z))
        scale = I_clamped.mean(dim=(-2, -1), keepdim=True) / pred_R.mean(dim=(-2, -1), keepdim=True)
        pred_R = pred_R * scale
    return pred_R


# =========================================================================
# 4. МЕТРИКИ КАЧЕСТВА ФИЛЬТРАЦИИ
# =========================================================================

def compute_all_metrics(I_raw: np.ndarray, I_filt: np.ndarray, looks: float = 4.4) -> dict[str, float]:
    Cu = 1.0 / math.sqrt(looks)

    mu_7 = ndimage.uniform_filter(I_raw, size=7)
    sq_7 = ndimage.uniform_filter(I_raw**2, size=7)
    sig_7 = np.sqrt(np.maximum(0, sq_7 - mu_7**2))
    ci = sig_7 / (mu_7 + 1e-8)
    homo_mask = (ci <= 1.05 * Cu)
    if np.sum(homo_mask) < 100:
        homo_mask = np.ones_like(I_raw, dtype=bool)

    raw_homo = I_raw[homo_mask]
    filt_homo = I_filt[homo_mask]

    enl_val = float((filt_homo.mean()**2) / (filt_homo.var() + 1e-9))

    l_raw = ndimage.laplace(I_raw).flatten()
    l_filt = ndimage.laplace(I_filt).flatten()
    if l_raw.std() > 1e-9 and l_filt.std() > 1e-9:
        epi_val = float(np.corrcoef(l_raw, l_filt)[0, 1])
    else:
        epi_val = 1.0

    bias_db = float(10.0 * np.log10(max(1e-10, I_filt.mean()) / max(1e-10, I_raw.mean())))

    m_o, v_o = float(I_raw.mean()), float(I_raw.var())
    m_f, v_f = float(I_filt.mean()), float(I_filt.var())
    ssi_val = float((np.sqrt(v_f) / (m_f + 1e-9)) * (m_o / (np.sqrt(v_o) + 1e-9)))

    q_fact = 1.0 + abs(m_o - m_f)
    smpi_val = float(q_fact * (v_f / (v_o + 1e-9)))

    ratio_img = np.clip(I_raw / np.maximum(I_filt, 1e-4), 0.0, 50.0)
    ratio_mean = float(ratio_img.mean())
    ratio_var = float(ratio_img.var())

    sub = ratio_img[::4, ::4]
    sub_c = sub - sub.mean()
    var_sub = float(np.mean(sub_c**2))
    if var_sub > 1e-9:
        cov_y = float((sub_c[:-1, :] * sub_c[1:, :]).mean())
        cov_x = float((sub_c[:, :-1] * sub_c[:, 1:]).mean())
        moran_val = float(0.5 * (cov_x + cov_y) / var_sub)
    else:
        moran_val = 0.0

    return {
        "enl": round(enl_val, 2),
        "epi": round(epi_val, 4),
        "bias_db": round(bias_db, 4),
        "ssi": round(ssi_val, 4),
        "smpi": round(smpi_val, 4),
        "ratio_mean": round(ratio_mean, 4),
        "ratio_var": round(ratio_var, 4),
        "moran_i": round(moran_val, 4),
    }


# =========================================================================
# 5. КОДИРОВАНИЕ СЛОЕВ В BASE64 PNG
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

    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)

    rgb = np.stack([r, g, b], axis=-1)
    rgb_8u = (rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb_8u, mode="RGB").convert("RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# =========================================================================
# 6. ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ ВСЕХ 5 КВАДРАНТОВ
# =========================================================================

def process_all_quadrants() -> dict[str, Any]:
    solver_ftv = JointFTVSARP(alpha=1.25, K=11, device=device)
    app_data = {}

    from train_sar_models import SAR_ResNet, SAR_MERLIN_BlindSpot, SAR_Restormer
    weights_dir = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\weights"
    
    model_resnet = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device)
    model_resnet.load_state_dict(torch.load(os.path.join(weights_dir, "sar_resnet_vitale.pth"), map_location=device))
    model_resnet.eval()
    
    model_merlin = SAR_MERLIN_BlindSpot(in_channels=1, num_features=48).to(device)
    model_merlin.load_state_dict(torch.load(os.path.join(weights_dir, "sar_merlin_blindspot.pth"), map_location=device))
    model_merlin.eval()
    
    model_restormer = SAR_Restormer(in_channels=1, base_dim=32).to(device)
    model_restormer.load_state_dict(torch.load(os.path.join(weights_dir, "sar_restormer.pth"), map_location=device))
    model_restormer.eval()

    def run_dl_inference(im_np: np.ndarray, model) -> np.ndarray:
        with torch.no_grad():
            scale = float(np.percentile(im_np, 99.5))
            if scale < 1e-4:
                scale = float(im_np.max())
            scale = max(scale, 1e-4)
            t = torch.from_numpy((im_np / scale).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                out = model(t)
            res = out.squeeze().cpu().numpy() * scale
        return np.maximum(res, 1e-6)

    for key, q in QUADRANTS.items():
        print(f"\n[INFO] Обработка квадранта: {q['title']}...")
        npz_file = os.path.join(CACHE_DIR, f"raw_{key}.npz")
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"Кэш-файл не найден: {npz_file}")

        data = np.load(npz_file)
        vv_raw = data["vv"]
        vh_raw = data["vh"]
        bbox = list(data["bbox"])

        t_start = time.time()

        vv_gamma = run_gamma_map(vv_raw, w=3, looks=5.0)
        vh_gamma = run_gamma_map(vh_raw, w=3, looks=5.0)

        I_torch = torch.from_numpy(np.stack([vv_raw, vh_raw], axis=0)).unsqueeze(0).to(device)
        with torch.no_grad():
            R_ftv = solver_ftv.solve(I_torch, n_iter=25)
        vv_ftv = R_ftv[0, 0].cpu().numpy()
        vh_ftv = R_ftv[0, 1].cpu().numpy()

        R_dip = run_rff_dip_sar(I_torch, n_iter=150)
        vv_dip = R_dip[0, 0].cpu().numpy()
        vh_dip = R_dip[0, 1].cpu().numpy()

        vv_resnet = run_dl_inference(vv_raw, model_resnet)
        vh_resnet = run_dl_inference(vh_raw, model_resnet)

        vv_merlin = run_dl_inference(vv_raw, model_merlin)
        vh_merlin = run_dl_inference(vh_raw, model_merlin)

        vv_restormer = run_dl_inference(vv_raw, model_restormer)
        vh_restormer = run_dl_inference(vh_raw, model_restormer)

        t_elapsed = time.time() - t_start
        print(f"[SUCCESS] {key} обработан за {t_elapsed:.2f} сек.")

        vv_raw_db = 10.0 * np.log10(np.maximum(vv_raw, 1e-6))
        vh_raw_db = 10.0 * np.log10(np.maximum(vh_raw, 1e-6))
        ratio_raw_db = vv_raw_db - vh_raw_db

        vv_gamma_db = 10.0 * np.log10(np.maximum(vv_gamma, 1e-6))
        vh_gamma_db = 10.0 * np.log10(np.maximum(vh_gamma, 1e-6))
        ratio_gamma_db = vv_gamma_db - vh_gamma_db

        vv_ftv_db = 10.0 * np.log10(np.maximum(vv_ftv, 1e-6))
        vh_ftv_db = 10.0 * np.log10(np.maximum(vh_ftv, 1e-6))
        ratio_ftv_db = vv_ftv_db - vh_ftv_db

        vv_dip_db = 10.0 * np.log10(np.maximum(vv_dip, 1e-6))
        vh_dip_db = 10.0 * np.log10(np.maximum(vh_dip, 1e-6))
        ratio_dip_db = vv_dip_db - vh_dip_db

        vv_resnet_db = 10.0 * np.log10(np.maximum(vv_resnet, 1e-6))
        vh_resnet_db = 10.0 * np.log10(np.maximum(vh_resnet, 1e-6))
        ratio_resnet_db = vv_resnet_db - vh_resnet_db

        vv_merlin_db = 10.0 * np.log10(np.maximum(vv_merlin, 1e-6))
        vh_merlin_db = 10.0 * np.log10(np.maximum(vh_merlin, 1e-6))
        ratio_merlin_db = vv_merlin_db - vh_merlin_db

        vv_restormer_db = 10.0 * np.log10(np.maximum(vv_restormer, 1e-6))
        vh_restormer_db = 10.0 * np.log10(np.maximum(vh_restormer, 1e-6))
        ratio_restormer_db = vv_restormer_db - vh_restormer_db

        m_raw = compute_all_metrics(vv_raw, vv_raw, looks=4.4)
        m_gamma = compute_all_metrics(vv_raw, vv_gamma, looks=4.4)
        m_ftv = compute_all_metrics(vv_raw, vv_ftv, looks=4.4)
        m_dip = compute_all_metrics(vv_raw, vv_dip, looks=4.4)
        m_resnet = compute_all_metrics(vv_raw, vv_resnet, looks=4.4)
        m_merlin = compute_all_metrics(vv_raw, vv_merlin, looks=4.4)
        m_restormer = compute_all_metrics(vv_raw, vv_restormer, looks=4.4)

        pipelines_dict = {
            "raw": {
                "name": "0. Исходный сигнал (Raw RTC, со спеклом)",
                "metrics": m_raw,
                "layers": {
                    "vv": array_to_png_base64(vv_raw_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_raw_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_raw_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_raw_db, vh_raw_db, ratio_raw_db),
                    "res_ratio": array_to_png_base64(np.ones_like(vv_raw), 0.0, 2.5, cmap_name="coolwarm"),
                },
            },
            "gamma": {
                "name": "1. Текущий пайплайн (Gamma MAP 3x3)",
                "metrics": m_gamma,
                "layers": {
                    "vv": array_to_png_base64(vv_gamma_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_gamma_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_gamma_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_gamma_db, vh_gamma_db, ratio_gamma_db),
                    "res_ratio": array_to_png_base64(vv_raw / np.maximum(vv_gamma, 1e-8), 0.0, 2.5, cmap_name="coolwarm"),
                },
            },
            "ftv": {
                "name": "2. Новый: Joint FTV-SARP Gamma (α=1.25)",
                "metrics": m_ftv,
                "layers": {
                    "vv": array_to_png_base64(vv_ftv_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_ftv_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_ftv_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_ftv_db, vh_ftv_db, ratio_ftv_db),
                    "res_ratio": array_to_png_base64(vv_raw / np.maximum(vv_ftv, 1e-8), 0.0, 2.5, cmap_name="coolwarm"),
                },
            },
            "dip": {
                "name": "3. Неконтролируемый: RFF-DIP Gamma + Variogram",
                "metrics": m_dip,
                "layers": {
                    "vv": array_to_png_base64(vv_dip_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_dip_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_dip_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_dip_db, vh_dip_db, ratio_dip_db),
                    "res_ratio": array_to_png_base64(vv_raw / np.maximum(vv_dip, 1e-8), 0.0, 2.5, cmap_name="coolwarm"),
                },
            },
            "resnet": {
                "name": "4. ML: SAR-ResNet (Vitale et al. 2021)",
                "metrics": m_resnet,
                "layers": {
                    "vv": array_to_png_base64(vv_resnet_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_resnet_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_resnet_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_resnet_db, vh_resnet_db, ratio_resnet_db),
                    "res_ratio": array_to_png_base64(vv_raw / np.maximum(vv_resnet, 1e-8), 0.0, 2.5, cmap_name="coolwarm"),
                },
            },
            "merlin": {
                "name": "5. ML: MERLIN Blind-Spot CNN (Dalsasso 2021)",
                "metrics": m_merlin,
                "layers": {
                    "vv": array_to_png_base64(vv_merlin_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_merlin_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_merlin_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_merlin_db, vh_merlin_db, ratio_merlin_db),
                    "res_ratio": array_to_png_base64(vv_raw / np.maximum(vv_merlin, 1e-8), 0.0, 2.5, cmap_name="coolwarm"),
                },
            },
            "restormer": {
                "name": "6. ML: SAR-Restormer Attention (Perera 2022)",
                "metrics": m_restormer,
                "layers": {
                    "vv": array_to_png_base64(vv_restormer_db, -25.0, 0.0),
                    "vh": array_to_png_base64(vh_restormer_db, -32.0, -8.0),
                    "ratio": array_to_png_base64(ratio_restormer_db, 0.0, 20.0, cmap_name="viridis"),
                    "rgb": rgb_composite_base64(vv_restormer_db, vh_restormer_db, ratio_restormer_db),
                    "res_ratio": array_to_png_base64(vv_raw / np.maximum(vv_restormer, 1e-8), 0.0, 2.5, cmap_name="coolwarm"),
                },
            },
        }

        leaflet_bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
        app_data[key] = {
            "title": q["title"],
            "desc": q["desc"],
            "center": q["center"],
            "zoom": q["zoom"],
            "bounds": leaflet_bounds,
            "shape": list(vv_raw.shape),
            "pipelines": pipelines_dict,
        }

    paper_results_dir = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\paper_results"
    def load_b64(fname):
        p = os.path.join(paper_results_dir, fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
        return ""

    app_data["_benchmarks"] = [
        {
            "id": "vitale",
            "paper": "Vitale et al. (arXiv:2104.09350v2, IEEE JSTARS 2021)",
            "title": "SAR-ResNet: Тестирование на синтетических текстурах и Sentinel-1",
            "desc": "В статье Vitale et al. для проверки сохранения резких границ и подавления спекла при L=4 использовались классические синтетические текстуры (включая текстуру шкуры зебры из базы BSDS500 для оценки контрастных краев), а также многопроходное усреднение Sentinel-1 (50 проходов в GEE). Сравнение проводится в естественном пиксельном пространстве (1:1).",
            "strip_img": load_b64("strip_vitale_resnet.png"),
            "pairs": {
                "noisy": { "name": "0. Вход со спеклом (L=4)", "img": load_b64("Vitale_SAR_ResNet_p8_img1_256x256.png") },
                "resnet": { "name": "1. SAR-ResNet (Vitale)", "img": load_b64("Vitale_SAR_ResNet_p8_img2_256x256.png") },
                "gt": { "name": "2. Эталон Ground Truth", "img": load_b64("Vitale_SAR_ResNet_p8_img0_256x256.png") }
            },
            "metrics": {
                "ENL": "42.5",
                "EPI": "0.82",
                "PSNR": "28.4 dB",
                "SSIM": "0.824"
            }
        },
        {
            "id": "merlin",
            "paper": "Dalsasso et al. (arXiv:2110.13148v2, IEEE TGRS 2021)",
            "title": "MERLIN: Реальный снимок TerraSAR-X плотины Сер-Понсон во Французских Альпах",
            "desc": "Реальный однопроходный снимок TerraSAR-X (X-диапазон). Авторы сравнивают предложенную слепую сверточную сеть MERLIN с вейвлетным методом SAR-BM3D и моделью Speckle2Void.",
            "strip_img": load_b64("strip_merlin.png"),
            "pairs": {
                "noisy": { "name": "0. Исходный снимок со спеклом", "img": load_b64("MERLIN_Dalsasso_p9_img0_512x512.png") },
                "merlin": { "name": "1. MERLIN Blind-Spot CNN", "img": load_b64("MERLIN_Dalsasso_p9_img3_512x512.png") },
                "bm3d": { "name": "2. SAR-BM3D", "img": load_b64("MERLIN_Dalsasso_p9_img1_512x512.png") },
                "s2v": { "name": "3. Speckle2Void", "img": load_b64("MERLIN_Dalsasso_p9_img2_512x512.png") }
            },
            "metrics": {
                "ENL": "28.4",
                "EPI": "0.72",
                "Bias": "+0.12 dB",
                "Ratio Var": "0.24"
            }
        },
        {
            "id": "restormer",
            "paper": "Perera et al. (arXiv:2201.09355v1, IEEE 2022)",
            "title": "SAR-Restormer: Сравнение PPB, BM3D, CNN и трансформера на реальном SAR",
            "desc": "Реальный радиолокационный снимок. Авторы сравнивают классический нелокальный фильтр PPB, вейвлетный SAR-BM3D, сверточный SAR-CNN, дилатированный ID-CNN и предложенный трансформер SAR-Restormer.",
            "strip_img": load_b64("strip_restormer.png"),
            "pairs": {
                "noisy": { "name": "0. Исходный снимок со спеклом", "img": load_b64("Restormer_SAR_p4_img0_512x512.jpeg") },
                "restormer": { "name": "1. SAR-Restormer (Трансформер)", "img": load_b64("Restormer_SAR_p4_img5_512x512.jpeg") },
                "bm3d": { "name": "2. SAR-BM3D", "img": load_b64("Restormer_SAR_p4_img2_512x512.jpeg") },
                "cnn": { "name": "3. SAR-CNN", "img": load_b64("Restormer_SAR_p4_img3_512x512.jpeg") },
                "idcnn": { "name": "4. ID-CNN", "img": load_b64("Restormer_SAR_p4_img4_512x512.jpeg") },
                "ppb": { "name": "5. PPB Filter", "img": load_b64("Restormer_SAR_p4_img1_512x512.jpeg") }
            },
            "metrics": {
                "ENL": "35.8",
                "EPI": "0.81",
                "PSNR": "29.1 dB",
                "SSIM": "0.845"
            }
        },
        {
            "id": "sim2real",
            "paper": "Al-Husseini et al. (arXiv:2601.01541v1, 2026)",
            "title": "Sim2Real: Оптический эталон vs Capella Space SAR vs SEDRUNet",
            "desc": "Прямое попиксельное сопоставление реального радиолокационного снимка Capella Space сверхвысокого разрешения, оптического эталона Ground Truth и результата нейросети SEDRUNet.",
            "strip_img": load_b64("strip_sim2real.png"),
            "pairs": {
                "noisy": { "name": "0. Реальный радар Capella Space", "img": load_b64("Sim2Real_2026_p6_img1_308x308.png") },
                "sedrunet": { "name": "1. Восстановленный SEDRUNet", "img": load_b64("Sim2Real_2026_p6_img2_308x308.png") },
                "gt": { "name": "2. Оптический Ground Truth", "img": load_b64("Sim2Real_2026_p6_img0_308x308.png") }
            },
            "metrics": {
                "ENL": "31.2",
                "EPI": "0.79",
                "PSNR": "27.8 dB",
                "SSIM": "0.812"
            }
        }
    ]

    return app_data


# =========================================================================
# 7. СБОРКА АВТОНОМНОГО HTML
# =========================================================================

from html_template import HTML_TEMPLATE

def build_advanced_html(data: dict[str, Any], output_path: str) -> None:
    json_str = json.dumps(data)
    html_content = HTML_TEMPLATE.replace("__DATA_JSON__", json_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[HTML SUCCESS] Автономный интерактивный файл записан: {output_path}")


def main() -> None:
    print("=" * 70)
    print("ЗАПУСК КОМПЛЕКСНОЙ ГЕНЕРАЦИИ СРАВНИТЕЛЬНОГО АНАЛИЗАТОРА SAR")
    print(f"Вычислительное устройство: {device}")
    print("=" * 70)

    app_data = process_all_quadrants()
    out_files = [
        r"D:\ПРОФ\ВУЗ\ВТСК\SAR\sar_advanced_comparison.html",
        r"D:\ПРОФ\ВУЗ\ВТСК\SAR\sar_visualization.html",
        r"D:\ПРОФ\ВУЗ\ВТСК\SAR\index.html",
    ]
    for out_html in out_files:
        build_advanced_html(app_data, out_html)
    print("=" * 70)
    print("ВСЕ РАБОТЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    print("=" * 70)


if __name__ == "__main__":
    main()
