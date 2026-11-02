# -*- coding: utf-8 -*-
"""Автономный сервер Sentinel-1 SAR с GPU-ускорением (RTX 3080)
для произвольных конвейеров фильтрации любой длины,
включая методы НИР (Robust PCA IALM, DIP-Moran, DIP-SARP, DIP-RFF, DIP-Pure)
и переключение сезонов (Весна, Лето, Осень, Зима).

Запуск:
    D:\\Programs\\Python\\python.exe sar_server.py [порт, по умолчанию 8080]
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm

sys.stdout.reconfigure(encoding='utf-8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[СЕРВЕР] Вычислительное устройство: {device}")
if torch.cuda.is_available():
    print(f"[СЕРВЕР] Графический процессор: {torch.cuda.get_device_name(0)}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(BASE_DIR, "weights")
paper_dir = os.path.join(BASE_DIR, "paper_results")
data_cache_dir = os.path.join(BASE_DIR, "data_cache")
nir_weights_dir = r"D:\ПРОФ\ВУЗ\ВТСК\НИР\DATA\DIP_TASK\weights"

# -------------------------------------------------------------------------
# 1. АРХИТЕКТУРЫ СЕТЕЙ
# -------------------------------------------------------------------------
from train_sar_models import SAR_ResNet, SAR_MERLIN_BlindSpot, SAR_Restormer
from build_full_benchmark import (
    filter_boxcar_mean, filter_median, filter_lee, filter_enhanced_lee,
    filter_kuan, filter_frost, filter_enhanced_frost, filter_bilateral,
    filter_gamma_map, JointFTVSARP, compute_metrics, array_to_png_base64
)

# DIP с GroupNorm (H2 Moran, H4 SARP)
class ConvBlockGN(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='reflect')
        self.gn = nn.GroupNorm(8, out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x).float()).to(x.dtype))

class DIPUNetGN(nn.Module):
    def __init__(self, in_c: int = 1, out_c: int = 1, base_filters: int = 64, depth: int = 5):
        super().__init__()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up = nn.ModuleList()
        ch = in_c
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.enc.append(nn.Sequential(ConvBlockGN(ch, out_ch), ConvBlockGN(out_ch, out_ch)))
            ch = out_ch
        for i in range(depth - 1, 0, -1):
            in_ch_dec = base_filters * (2 ** i)
            out_ch_dec = base_filters * (2 ** (i - 1))
            self.up.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.up_convs.append(ConvBlockGN(in_ch_dec, out_ch_dec))
            self.dec.append(nn.Sequential(ConvBlockGN(out_ch_dec * 2, out_ch_dec), ConvBlockGN(out_ch_dec, out_ch_dec)))
        self.final = nn.Conv2d(base_filters, out_c, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = []
        for i, l in enumerate(self.enc):
            x = l(x)
            if i < 4:
                s.append(x)
                x = self.pool(x)
        for i, (u, uc, l) in enumerate(zip(self.up, self.up_convs, self.dec)):
            skip = s[-(i + 1)]
            x = uc(u(x))
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = l(torch.cat([x, skip], dim=1))
        return self.final(x)

# DIP Pure (H1)
class DIPUNetPure(nn.Module):
    def __init__(self, in_c: int = 1, out_c: int = 1, base_filters: int = 64, depth: int = 5):
        super().__init__()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up = nn.ModuleList()
        ch = in_c
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.enc.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            ch = out_ch
        for i in range(depth - 1, 0, -1):
            in_ch_dec = base_filters * (2 ** i)
            out_ch_dec = base_filters * (2 ** (i - 1))
            self.up.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.up_convs.append(nn.Conv2d(in_ch_dec, out_ch_dec, 3, padding=1, padding_mode='reflect'))
            self.dec.append(nn.Sequential(
                nn.Conv2d(out_ch_dec * 2, out_ch_dec, 3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch_dec, out_ch_dec, 3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.final = nn.Conv2d(base_filters, out_c, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = []
        for i, l in enumerate(self.enc):
            x = l(x)
            if i < 4:
                s.append(x)
                x = self.pool(x)
        for i, (u, uc, l) in enumerate(zip(self.up, self.up_convs, self.dec)):
            skip = s[-(i + 1)]
            x = uc(u(x))
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = l(torch.cat([x, skip], dim=1))
        return self.final(x)

# Random Fourier Features + U-Net для RFF-DIP
class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features: int = 2, num_features: int = 16, sigma: float = 8.0):
        super().__init__()
        b_mat = torch.randn(num_features, in_features) * sigma
        self.B = nn.Parameter(b_mat, requires_grad=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * torch.einsum("nc,bchw->bnhw", self.B, coords)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)

class SARUNetRFF(nn.Module):
    def __init__(self, in_ch: int = 32, out_ch: int = 1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2))
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2))
        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = F.interpolate(e2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)

# -------------------------------------------------------------------------
# 2. ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА ВЕСОВ МОДЕЛЕЙ
# -------------------------------------------------------------------------
model_res = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device).eval()
model_res.load_state_dict(torch.load(os.path.join(weights_dir, "sar_resnet_vitale.pth"), map_location=device))

model_mer = SAR_MERLIN_BlindSpot(in_channels=1, num_features=48).to(device).eval()
model_mer.load_state_dict(torch.load(os.path.join(weights_dir, "sar_merlin_blindspot.pth"), map_location=device))

model_rst = SAR_Restormer(in_channels=1, base_dim=32).to(device).eval()
model_rst.load_state_dict(torch.load(os.path.join(weights_dir, "sar_restormer.pth"), map_location=device))

solver_ftv = JointFTVSARP(alpha=1.25, K=11, device=device)

# Загрузка моделей НИР
model_h1 = None
p_h1 = os.path.join(nir_weights_dir, "h1_pure_dip.pt")
if os.path.exists(p_h1):
    try:
        model_h1 = DIPUNetPure().to(device).eval()
        model_h1.load_state_dict(torch.load(p_h1, map_location=device)["state_dict"])
        print("[СЕРВЕР] DIP-Pure (H1) веса загружены.")
    except Exception as e:
        print(f"[СЕРВЕР ПРЕДУПРЕЖДЕНИЕ] H1 DIP: {e}")

model_h2 = None
p_h2 = os.path.join(nir_weights_dir, "h2_moran_dip.pt")
if os.path.exists(p_h2):
    try:
        model_h2 = DIPUNetGN().to(device).eval()
        model_h2.load_state_dict(torch.load(p_h2, map_location=device)["state_dict"])
        print("[СЕРВЕР] DIP-Moran (H2) веса загружены.")
    except Exception as e:
        print(f"[СЕРВЕР ПРЕДУПРЕЖДЕНИЕ] H2 Moran DIP: {e}")

model_h4 = None
p_h4 = os.path.join(nir_weights_dir, "h4_sarp_dip.pt")
if os.path.exists(p_h4):
    try:
        model_h4 = DIPUNetGN().to(device).eval()
        model_h4.load_state_dict(torch.load(p_h4, map_location=device)["state_dict"])
        print("[СЕРВЕР] DIP-SARP (H4) веса загружены.")
    except Exception as e:
        print(f"[СЕРВЕР ПРЕДУПРЕЖДЕНИЕ] H4 SARP DIP: {e}")

# -------------------------------------------------------------------------
# 3. МАТЕМАТИЧЕСКИЕ ОПЕРАТОРЫ И АЛГОРИТМЫ
# -------------------------------------------------------------------------

def run_dl(im_np: np.ndarray, model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        scale = float(np.percentile(im_np, 99.5))
        if scale < 1e-4: scale = float(im_np.max())
        scale = max(scale, 1e-4)
        t = torch.from_numpy((im_np / scale).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            out = model(t)
        res = out.squeeze().float().cpu().numpy() * scale
    return np.maximum(res.astype(np.float32), 1e-6)

def run_dip_eval(im_np: np.ndarray, model: nn.Module) -> np.ndarray:
    """Вычисление отклика DIP с паддингом до кратности 32."""
    H, W = im_np.shape
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32
    scale = float(np.percentile(im_np, 99.5))
    if scale < 1e-4: scale = float(im_np.max())
    scale = max(scale, 1e-4)
    norm = (im_np / scale).astype(np.float32)
    padded = np.pad(norm, ((0, pad_h), (0, pad_w)), mode='reflect')
    t = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(t)
        res = out.squeeze().float().cpu().numpy()[:H, :W] * scale
    return np.maximum(res.astype(np.float32), 1e-6)

def run_rpca_ialm(im_np: np.ndarray, max_iter: int = 12, tol: float = 5e-4) -> tuple[np.ndarray, np.ndarray]:
    """Inexact ALM (IALM) метод робастных главных компонент: Y = L + S."""
    Y = torch.from_numpy(im_np.astype(np.float32)).to(device)
    M, N = Y.shape
    lmbda = 1.0 / np.sqrt(max(M, N))
    norm_two = torch.linalg.matrix_norm(Y, ord=2)
    norm_inf = torch.max(torch.abs(Y)) / lmbda
    dual_norm = max(norm_two.item(), norm_inf.item())
    Y_dual = Y / max(dual_norm, 1e-6)

    mu = 1.25 / max(norm_two.item(), 1e-6)
    mu_bar = mu * 1e7
    rho = 1.5

    L = torch.zeros_like(Y)
    S = torch.zeros_like(Y)
    norm_F = torch.linalg.norm(Y, ord='fro')

    for _ in range(max_iter):
        temp_L = Y - S + (1.0 / mu) * Y_dual
        U, S_val, Vh = torch.linalg.svd(temp_L, full_matrices=False)
        thresh = 1.0 / mu
        S_thresh = torch.clamp(S_val - thresh, min=0.0)
        L = torch.mm(U, torch.mm(torch.diag(S_thresh), Vh))

        temp_S = Y - L + (1.0 / mu) * Y_dual
        tau_S = lmbda / mu
        S = torch.sign(temp_S) * torch.clamp(torch.abs(temp_S) - tau_S, min=0.0)

        Z = Y - L - S
        Y_dual = Y_dual + mu * Z
        mu = min(mu * rho, mu_bar)

        err = torch.linalg.norm(Z, ord='fro') / (norm_F + 1e-8)
        if err.item() < tol:
            break

    L_np = torch.clamp(L, min=1e-6).cpu().numpy()
    S_np = S.cpu().numpy()
    return L_np, S_np

def run_rff_dip_single(im_np: np.ndarray, n_iter: int = 25) -> np.ndarray:
    """Random Fourier Features DIP с оптимизацией вариационной структуры."""
    H, W = im_np.shape
    y_g, x_g = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing="ij")
    coords = torch.stack([x_g, y_g], dim=0).unsqueeze(0)
    rff = RandomFourierFeatures(in_features=2, num_features=16, sigma=8.0).to(device)
    net = SARUNetRFF(in_ch=32, out_ch=1).to(device)
    z = rff(coords)

    t_im = torch.clamp(torch.from_numpy(im_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device), min=1e-6)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    psi_target = 0.255036

    for _ in range(n_iter):
        opt.zero_grad()
        p_log = net(z)
        p_R = torch.exp(p_log)
        l_nll = (p_log + t_im / p_R).mean()
        res_log = torch.log(t_im) - p_log
        diff_x = res_log[:, :, :, 1:] - res_log[:, :, :, :-1]
        diff_y = res_log[:, :, 1:, :] - res_log[:, :, :-1, :]
        gamma_1 = 0.5 * (diff_x.pow(2).mean() + diff_y.pow(2).mean())
        loss = l_nll + 2.0 * (gamma_1 - psi_target).pow(2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = torch.exp(net(z))
        scale = t_im.mean() / out.mean()
        out = out * scale
    return torch.clamp(out.squeeze(), min=1e-6).cpu().numpy()

def run_ftv(im_np: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(im_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = solver_ftv.solve(t, n_iter=15)
    return out[0, 0].cpu().numpy()

OPERATORS = {
    # 1. Классические локальные фильтры
    "BXC": lambda x: filter_boxcar_mean(x, w=5),
    "MED": lambda x: filter_median(x, w=5),
    "LEE": lambda x: filter_lee(x, w=5, looks=4.4),
    "ENH_LEE": lambda x: filter_enhanced_lee(x, w=5, looks=4.4),
    "KUA": lambda x: filter_kuan(x, w=5, looks=4.4),
    "FRO": lambda x: filter_frost(x, w=5, damping=2.0),
    "ENH_FRO": lambda x: filter_enhanced_frost(x, w=5, looks=4.4, damping=2.0),
    "BIL": lambda x: filter_bilateral(x),
    "GMAP": lambda x: filter_gamma_map(x, w=5, looks=4.4),

    # 2. Вариационные и глубокие нейросетевые модели (из статей)
    "FTV": run_ftv,
    "RES": lambda x: run_dl(x, model_res),
    "MER": lambda x: run_dl(x, model_mer),
    "RST": lambda x: run_dl(x, model_rst),

    # 3. Инновации из НИР (RPCA, DIP-Moran, DIP-SARP, DIP-Pure, DIP-RFF)
    "RPCA": lambda x: run_rpca_ialm(x)[0],
    "RPCA_S": lambda x: np.maximum(im_np_raw_diff := np.abs(run_rpca_ialm(x)[1]), 1e-6),
    "DIP_MORAN": lambda x: run_dip_eval(x, model_h2) if model_h2 is not None else filter_gamma_map(x, w=5),
    "DIP_SARP": lambda x: run_dip_eval(x, model_h4) if model_h4 is not None else filter_gamma_map(x, w=5),
    "DIP_PURE": lambda x: run_dip_eval(x, model_h1) if model_h1 is not None else filter_boxcar_mean(x, w=5),
    "DIP_RFF": lambda x: run_rff_dip_single(x, n_iter=25),
}

# -------------------------------------------------------------------------
# 4. КЭШ ДАННЫХ И СЕЗОНОВ
# -------------------------------------------------------------------------
RAW_CACHE = {}

BENCH_CACHE = {
    "vitale": {
        "raw": np.array(Image.open(os.path.join(paper_dir, "Vitale_SAR_ResNet_p8_img1_256x256.png")).convert("L")).astype(np.float32) / 255.0,
        "gt": np.array(Image.open(os.path.join(paper_dir, "Vitale_SAR_ResNet_p8_img0_256x256.png")).convert("L")).astype(np.float32) / 255.0,
    },
    "merlin": {
        "raw": np.array(Image.open(os.path.join(paper_dir, "MERLIN_Dalsasso_p9_img0_512x512.png")).convert("L")).astype(np.float32) / 255.0,
        "gt": None,
    },
    "restormer": {
        "raw": np.array(Image.open(os.path.join(paper_dir, "Restormer_SAR_p4_img0_512x512.jpeg")).convert("L")).astype(np.float32) / 255.0,
        "gt": None,
    },
    "sim2real": {
        "raw": np.array(Image.open(os.path.join(paper_dir, "Sim2Real_2026_p6_img1_308x308.png")).convert("L")).astype(np.float32) / 255.0,
        "gt": np.array(Image.open(os.path.join(paper_dir, "Sim2Real_2026_p6_img0_308x308.png")).convert("L")).astype(np.float32) / 255.0,
    }
}

GIS_KEYS = ["benevento", "rome", "serre_poncon", "forest", "city", "water", "desert", "mountain"]

for k in GIS_KEYS:
    # 1. Проверяем сезонный файл
    seasons_p = os.path.join(data_cache_dir, f"seasons_{k}.npz")
    raw_p = os.path.join(data_cache_dir, f"raw_{k}.npz")

    sc_dict = {"seasons": {}}

    if os.path.exists(seasons_p):
        s_data = np.load(seasons_p)
        for s in ["winter", "spring", "summer", "autumn"]:
            if f"{s}_vv" in s_data:
                sc_dict["seasons"][s] = {
                    "vv": s_data[f"{s}_vv"],
                    "vh": s_data[f"{s}_vh"],
                    "date": str(s_data[f"{s}_date"]),
                }
        # Устанавливаем дефолтное состояние (весна)
        default_s = "spring" if "spring" in sc_dict["seasons"] else list(sc_dict["seasons"].keys())[0]
        sc_dict["vv"] = sc_dict["seasons"][default_s]["vv"]
        sc_dict["vh"] = sc_dict["seasons"][default_s]["vh"]
        sc_dict["date"] = sc_dict["seasons"][default_s]["date"]
        sc_dict["gt"] = None

    elif os.path.exists(raw_p):
        d = np.load(raw_p)
        gt_lin = d["vv_gt"] if ("vv_gt" in d and d["vv_gt"] is not None and getattr(d["vv_gt"], "ndim", 0) > 0) else None
        sc_dict["vv"] = d["vv"]
        sc_dict["vh"] = d["vh"]
        sc_dict["gt"] = gt_lin
        sc_dict["date"] = "2023-04-15"

    RAW_CACHE[k] = sc_dict

print(f"[СЕРВЕР] Инициализирован кэш GIS полигонов ({len(RAW_CACHE)}/8): {list(RAW_CACHE.keys())}")
for k, v in RAW_CACHE.items():
    s_keys = list(v.get("seasons", {}).keys())
    print(f"  • {k:13s}: сезоны = {s_keys}, форма = {v['vv'].shape}")

# -------------------------------------------------------------------------
# 5. КОМПОЗИТЫ И ЦВЕТОВОЕ ОТОБРАЖЕНИЕ
# -------------------------------------------------------------------------

def rgb_composite_base64_fixed(vv_db: np.ndarray, vh_db: np.ndarray, ratio_db: np.ndarray) -> str:
    """Физически откалиброванный поляриметрический RGB-композит без синего насыщения."""
    r = np.clip((vv_db - (-25.0)) / (0.0 - (-25.0)), 0.0, 1.0)
    g = np.clip((vh_db - (-32.0)) / (-8.0 - (-32.0)), 0.0, 1.0)
    b = np.clip((ratio_db - 2.0) / (18.0 - 2.0), 0.0, 1.0)
    rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=-1)
    rgb_8u = (rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb_8u, mode="RGB").convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def execute_pipeline(im_start: np.ndarray, stages: list[str]) -> np.ndarray:
    curr = im_start.copy()
    for s in stages:
        op = s.strip().upper()
        if op in OPERATORS:
            curr = OPERATORS[op](curr)
    return curr

def generate_layer_b64(vv_arr: np.ndarray, vh_arr: np.ndarray, layer_type: str) -> str:
    """Генерация base64 изображения для заданного слоя."""
    if layer_type == "rgb":
        vv_db = 10.0 * np.log10(np.maximum(vv_arr, 1e-6))
        vh_db = 10.0 * np.log10(np.maximum(vh_arr, 1e-6))
        ratio_db = vv_db - vh_db
        return rgb_composite_base64_fixed(vv_db, vh_db, ratio_db)
    elif layer_type == "ratio":
        vv_db = 10.0 * np.log10(np.maximum(vv_arr, 1e-6))
        vh_db = 10.0 * np.log10(np.maximum(vh_arr, 1e-6))
        ratio_db = vv_db - vh_db
        return array_to_png_base64(ratio_db, 0.0, 20.0, cmap_name="viridis")
    elif layer_type == "vh":
        vh_db = 10.0 * np.log10(np.maximum(vh_arr, 1e-6))
        return array_to_png_base64(vh_db, -32.0, -8.0)
    else: # "vv"
        vv_db = 10.0 * np.log10(np.maximum(vv_arr, 1e-6))
        return array_to_png_base64(vv_db, -25.0, 0.0)

# -------------------------------------------------------------------------
# 6. HTTP ОБРАБОТЧИК ЗАПРОСОВ
# -------------------------------------------------------------------------

class SARRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/pipeline":
            t0 = time.perf_counter()
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                req = json.loads(body)
                mode = req.get("mode", "gis") # "gis" or "bench"
                target_id = req.get("target_id", "forest")
                season = req.get("season", "spring")
                stages = req.get("stages", ["FTV", "RES"])
                layer = req.get("layer", "rgb")

                if mode == "bench":
                    if target_id not in BENCH_CACHE:
                        raise ValueError(f"Неизвестный бенчмарк: '{target_id}'. Доступны: {list(BENCH_CACHE.keys())}")
                    b_data = BENCH_CACHE[target_id]
                    im_raw = b_data["raw"]
                    im_gt = b_data["gt"]
                    im_filt = execute_pipeline(im_raw, stages) if stages else im_raw
                    m = compute_metrics(im_raw, im_filt, gt_lin=im_gt)
                    norm = np.clip(im_filt, 0.0, 1.0)
                    u8 = (norm * 255.0).astype(np.uint8)
                    img = Image.fromarray(u8, mode="L")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                    date_str = "Benchmark"
                else:
                    # GIS mode
                    if target_id not in RAW_CACHE:
                        raise ValueError(f"Неизвестный полигон: '{target_id}'. Доступны: {list(RAW_CACHE.keys())}")
                    sc_data = RAW_CACHE[target_id]

                    # Извлечение сезонных данных
                    if "seasons" in sc_data and season in sc_data["seasons"]:
                        vv_input = sc_data["seasons"][season]["vv"]
                        vh_input = sc_data["seasons"][season]["vh"]
                        date_str = sc_data["seasons"][season]["date"]
                    else:
                        vv_input = sc_data["vv"]
                        vh_input = sc_data["vh"]
                        date_str = sc_data.get("date", "2023-04-15")

                    # Выполнение пайплайна
                    if not stages or stages == ["RAW"]:
                        im_vv_filt = vv_input.copy()
                        im_vh_filt = vh_input.copy()
                    else:
                        im_vv_filt = execute_pipeline(vv_input, stages)
                        im_vh_filt = execute_pipeline(vh_input, stages)

                    # Формирование визуального слоя
                    if layer == "rgb":
                        vv_db = 10.0 * np.log10(np.maximum(im_vv_filt, 1e-6))
                        vh_db = 10.0 * np.log10(np.maximum(im_vh_filt, 1e-6))
                        ratio_db = vv_db - vh_db
                        b64 = rgb_composite_base64_fixed(vv_db, vh_db, ratio_db)
                        m = compute_metrics(vv_input, im_vv_filt, gt_lin=sc_data.get("gt"))
                    elif layer == "ratio":
                        vv_db = 10.0 * np.log10(np.maximum(im_vv_filt, 1e-6))
                        vh_db = 10.0 * np.log10(np.maximum(im_vh_filt, 1e-6))
                        ratio_db = vv_db - vh_db
                        b64 = array_to_png_base64(ratio_db, 0.0, 20.0, cmap_name="viridis")
                        m = compute_metrics(vv_input, im_vv_filt, gt_lin=sc_data.get("gt"))
                    elif layer == "res_ratio":
                        res_map = vv_input / np.maximum(im_vv_filt, 1e-6)
                        b64 = array_to_png_base64(res_map, 0.0, 2.0, cmap_name="coolwarm")
                        m = compute_metrics(vv_input, im_vv_filt, gt_lin=sc_data.get("gt"))
                    else:
                        im_raw = vv_input if layer != "vh" else vh_input
                        im_filt = im_vv_filt if layer != "vh" else im_vh_filt
                        fl_db = 10.0 * np.log10(np.maximum(im_filt, 1e-6))
                        vmin = -25.0 if layer == "vv" else -32.0
                        vmax = 0.0 if layer == "vv" else -8.0
                        b64 = array_to_png_base64(fl_db, vmin, vmax)
                        m = compute_metrics(im_raw, im_filt, gt_lin=sc_data.get("gt"))

                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                resp = {
                    "status": "ok",
                    "compute_time_ms": elapsed_ms,
                    "target_id": target_id,
                    "season": season,
                    "date": date_str,
                    "stages": stages,
                    "layer": layer,
                    "metrics": m,
                    "img": b64
                }
                def default_json(o):
                    if isinstance(o, (np.floating, np.integer)):
                        return float(o)
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    return str(o)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(resp, default=default_json).encode('utf-8'))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                err_resp = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(err_resp).encode('utf-8'))
        else:
            super().do_POST()

def main():
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    server_address = ('', port)
    httpd = HTTPServer(server_address, SARRequestHandler)
    print(f"\n[СЕРВЕР] Готов к приему запросов на http://localhost:{port}/")
    print(f"[СЕРВЕР] Доступные операторы ({len(OPERATORS)}): {list(OPERATORS.keys())}")
    httpd.serve_forever()

if __name__ == "__main__":
    main()
