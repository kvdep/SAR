# -*- coding: utf-8 -*-
"""Автономный сервер Sentinel-1 SAR с GPU-ускорением (RTX 3080)
для произвольных конвейеров фильтрации любой длины,
включая параметры фильтров (kwargs), метод НИР (Robust PCA IALM),
модель MERLIN CNN, и переключение сезонов (Весна, Лето, Осень, Зима).

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

# Ограничение видеопамяти VRAM в соответствии с gpu_vram_coordination.md (не более 2.0 ГБ)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.20, device=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[СЕРВЕР] Вычислительное устройство: {device}")
if torch.cuda.is_available():
    print(f"[СЕРВЕР] Графический процессор: {torch.cuda.get_device_name(0)}")
    print(f"[СЕРВЕР] Лимит VRAM установлен: 20% (~2048 МБ) для параллельной работы агентов.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(BASE_DIR, "weights")
paper_dir = os.path.join(BASE_DIR, "paper_results")
data_cache_dir = os.path.join(BASE_DIR, "data_cache")

# -------------------------------------------------------------------------
# 1. АРХИТЕКТУРЫ СЕТЕЙ
# -------------------------------------------------------------------------
from train_sar_models import SAR_ResNet, SAR_MERLIN_CNN, SAR_Restormer
from build_full_benchmark import (
    filter_boxcar_mean, filter_median, filter_lee, filter_enhanced_lee,
    filter_kuan, filter_frost, filter_enhanced_frost, filter_bilateral,
    filter_gamma_map, JointFTVSARP, compute_metrics, array_to_png_base64
)

# -------------------------------------------------------------------------
# 2. ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА ВЕСОВ МОДЕЛЕЙ
# -------------------------------------------------------------------------
model_res = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device).eval()
p_res = os.path.join(weights_dir, "sar_resnet_vitale.pth")
if os.path.exists(p_res):
    model_res.load_state_dict(torch.load(p_res, map_location=device))
    print(f"[СЕРВЕР] SAR-ResNet веса загружены: {p_res}")

model_mer = SAR_MERLIN_CNN(in_channels=1, num_features=36).to(device).eval()
p_mer = os.path.join(weights_dir, "sar_merlin_cnn.pth")
if os.path.exists(p_mer):
    model_mer.load_state_dict(torch.load(p_mer, map_location=device))
    print(f"[СЕРВЕР] Каноническая модель MERLIN CNN загружена: {p_mer}")

model_rst = SAR_Restormer(in_channels=1, base_dim=32).to(device).eval()
p_rst = os.path.join(weights_dir, "sar_restormer.pth")
if os.path.exists(p_rst):
    model_rst.load_state_dict(torch.load(p_rst, map_location=device))
    print(f"[СЕРВЕР] SAR-Restormer веса загружены: {p_rst}")

solver_ftv_default = JointFTVSARP(alpha=1.25, K=11, device=device)

# -------------------------------------------------------------------------
# 3. МАТЕМАТИЧЕСКИЕ ОПЕРАТОРЫ И АЛГОРИТМЫ
# -------------------------------------------------------------------------

def run_dl(im_np: np.ndarray, model: nn.Module) -> np.ndarray:
    """Безопасный нейросетевой вывод с 2D оконным блочным разбиением (окно Ханна),
    гарантирующий потребление VRAM < 250 МБ и исключающий ошибки CUDA Out of Memory."""
    H, W = im_np.shape
    scale = float(np.percentile(im_np, 99.5))
    if scale < 1e-4:
        scale = float(im_np.max())
    scale = max(scale, 1e-4)
    norm_im = np.clip(im_np / scale, 0.0, 10.0).astype(np.float32)

    # Если размер компактный и не трансформер - прямой быстрый проход
    if H <= 256 and W <= 256 and not isinstance(model, SAR_Restormer):
        t = torch.from_numpy(norm_im).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(t)
            res = out.squeeze().float().cpu().numpy() * scale
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.maximum(np.nan_to_num(res, nan=1e-6).astype(np.float32), 1e-6)

    # Оконная блочная обработка 256x256 с шагом 224 и перекрытием 32
    tile_size = 256
    stride = 224
    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride
    if H + pad_h < tile_size:
        pad_h = tile_size - H
    if W + pad_w < tile_size:
        pad_w = tile_size - W

    padded = np.pad(norm_im, ((0, pad_h), (0, pad_w)), mode='reflect')
    H_pad, W_pad = padded.shape

    w_y = np.hanning(tile_size + 2)[1:-1].astype(np.float32)
    w_x = np.hanning(tile_size + 2)[1:-1].astype(np.float32)
    window_2d = np.outer(w_y, w_x)
    window_t = torch.from_numpy(window_2d).to(device)

    canvas = torch.zeros((H_pad, W_pad), dtype=torch.float32, device=device)
    weight_canvas = torch.zeros((H_pad, W_pad), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for r in range(0, H_pad - tile_size + 1, stride):
            for c in range(0, W_pad - tile_size + 1, stride):
                tile = padded[r:r+tile_size, c:c+tile_size]
                t_tile = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
                if isinstance(model, SAR_Restormer):
                    out_t = model(t_tile)
                else:
                    with torch.amp.autocast('cuda'):
                        out_t = model(t_tile)
                out_t = out_t.squeeze()
                canvas[r:r+tile_size, c:c+tile_size] += out_t * window_t
                weight_canvas[r:r+tile_size, c:c+tile_size] += window_t

    res_padded = (canvas / torch.clamp(weight_canvas, min=1e-6)).cpu().numpy()
    res = res_padded[:H, :W] * scale
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.maximum(np.nan_to_num(res, nan=1e-6).astype(np.float32), 1e-6)

def run_rpca_ialm(im_np: np.ndarray, max_iter: int = 12, tol: float = 5e-4) -> tuple[np.ndarray, np.ndarray]:
    """Inexact ALM (IALM) метод робастных главных компонент: Y = L + S.
    L - гладкий низкоранговый фон ландшафта, S - разреженные точечные цели.
    """
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

def run_ftv(im_np: np.ndarray, alpha: float = 1.25, n_iter: int = 15, K: int = 11) -> np.ndarray:
    if abs(alpha - 1.25) < 1e-3 and K == 11:
        solver = solver_ftv_default
    else:
        solver = JointFTVSARP(alpha=alpha, K=K, device=device)
    t = torch.from_numpy(im_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = solver.solve(t, n_iter=n_iter)
    return out[0, 0].cpu().numpy()

OPERATORS = {
    # 1. Классические локальные фильтры с поддержкой kwargs
    "BXC": lambda x, **kw: filter_boxcar_mean(x, w=int(kw.get("w", 5))),
    "MED": lambda x, **kw: filter_median(x, w=int(kw.get("w", 5))),
    "LEE": lambda x, **kw: filter_lee(x, w=int(kw.get("w", 5)), looks=float(kw.get("looks", 4.4))),
    "ENH_LEE": lambda x, **kw: filter_enhanced_lee(x, w=int(kw.get("w", 5)), looks=float(kw.get("looks", 4.4)), k=float(kw.get("k", 1.0))),
    "KUA": lambda x, **kw: filter_kuan(x, w=int(kw.get("w", 5)), looks=float(kw.get("looks", 4.4))),
    "FRO": lambda x, **kw: filter_frost(x, w=int(kw.get("w", 5)), damping=float(kw.get("damping", 2.0))),
    "ENH_FRO": lambda x, **kw: filter_enhanced_frost(x, w=int(kw.get("w", 5)), looks=float(kw.get("looks", 4.4)), damping=float(kw.get("damping", 2.0))),
    "BIL": lambda x, **kw: filter_bilateral(x, w=int(kw.get("w", 5)), sigma_s=float(kw.get("sigma_s", 2.0)), sigma_r=float(kw.get("sigma_r", 0.15))),
    "GMAP": lambda x, **kw: filter_gamma_map(x, w=int(kw.get("w", 5)), looks=float(kw.get("looks", 4.4))),

    # 2. Вариационный метод дробного порядка FTV-SARP
    "FTV": lambda x, **kw: run_ftv(x, alpha=float(kw.get("alpha", 1.25)), n_iter=int(kw.get("n_iter", 15)), K=int(kw.get("K", 11))),

    # 3. Глубокие нейросетевые модели
    "RES": lambda x, **kw: run_dl(x, model_res),
    "MER": lambda x, **kw: run_dl(x, model_mer),
    "RST": lambda x, **kw: run_dl(x, model_rst),

    # 4. Аналитический метод НИР (Robust PCA IALM)
    "RPCA": lambda x, **kw: run_rpca_ialm(x, max_iter=int(kw.get("max_iter", 12)), tol=float(kw.get("tol", 5e-4)))[0],
    "RPCA_S": lambda x, **kw: np.maximum(np.abs(run_rpca_ialm(x, max_iter=int(kw.get("max_iter", 12)), tol=float(kw.get("tol", 5e-4)))[1]), 1e-6),
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
    raw_p = os.path.join(data_cache_dir, f"raw_{k}.npz")
    seasons_p = os.path.join(data_cache_dir, f"seasons_{k}.npz")

    if not os.path.exists(raw_p):
        continue

    d = np.load(raw_p)
    base_vv = d["vv"].astype(np.float32)
    base_vh = d["vh"].astype(np.float32)
    gt_lin = d["vv_gt"] if ("vv_gt" in d and d["vv_gt"] is not None and getattr(d["vv_gt"], "ndim", 0) > 0) else None
    H_base, W_base = base_vv.shape

    sc_dict = {
        "vv": base_vv,
        "vh": base_vh,
        "gt": gt_lin,
        "date": "15 апреля 2023 (Весна)",
        "seasons": {
            "spring": {
                "vv": base_vv,
                "vh": base_vh,
                "date": "15 апреля 2023 (Весна)",
            }
        }
    }

    if os.path.exists(seasons_p):
        s_data = np.load(seasons_p)
        ref_vv_mean = float(base_vv.mean())
        ref_vh_mean = float(base_vh.mean())
        label_map = {
            "winter": "15 января 2023 (Зима)",
            "summer": "15 июля 2023 (Лето)",
            "autumn": "15 октября 2023 (Осень)",
        }

        for s in ["winter", "summer", "autumn"]:
            if f"{s}_vv" in s_data:
                s_vv_raw = s_data[f"{s}_vv"].astype(np.float32)
                s_vh_raw = s_data[f"{s}_vh"].astype(np.float32)

                # Билинейная передискретизация на каноническую сетку базового слоя
                if s_vv_raw.shape != (H_base, W_base):
                    scale_y = H_base / s_vv_raw.shape[0]
                    scale_x = W_base / s_vv_raw.shape[1]
                    s_vv = ndimage.zoom(s_vv_raw, (scale_y, scale_x), order=1)
                    s_vh = ndimage.zoom(s_vh_raw, (scale_y, scale_x), order=1)
                else:
                    s_vv = s_vv_raw
                    s_vh = s_vh_raw

                s_vv = np.maximum(s_vv, 1e-6)
                s_vh = np.maximum(s_vh, 1e-6)

                # Калибровка динамического диапазона относительно канонического эталона
                calib_vv = s_vv * (ref_vv_mean / max(float(s_vv.mean()), 1e-6))
                calib_vh = s_vh * (ref_vh_mean / max(float(s_vh.mean()), 1e-6))

                sc_dict["seasons"][s] = {
                    "vv": calib_vv.astype(np.float32),
                    "vh": calib_vh.astype(np.float32),
                    "date": label_map.get(s, str(s_data.get(f"{s}_date", s))),
                }

    RAW_CACHE[k] = sc_dict

print(f"[СЕРВЕР] Инициализирован кэш GIS полигонов ({len(RAW_CACHE)}/8): {list(RAW_CACHE.keys())}")

# -------------------------------------------------------------------------
# 5. КОМПОЗИТЫ И ЦВЕТОВОЕ ОТОБРАЖЕНИЕ
# -------------------------------------------------------------------------

def rgb_composite_base64_calibrated(vv_db: np.ndarray, vh_db: np.ndarray, ratio_db: np.ndarray) -> str:
    """Точно откалиброванный поляриметрический RGB-композит:
    R = clip((VV - (-18)) / (-2 - (-18)), 0, 1)
    G = clip((VH - (-24)) / (-8 - (-24)), 0, 1)
    B = clip((Ratio - 2) / (14 - 2), 0, 1)
    Сохраняет естественные радиолокационные тени, крутые склоны и дюны в синем спектре.
    """
    r = np.clip((vv_db - (-18.0)) / (-2.0 - (-18.0)), 0.0, 1.0)
    g = np.clip((vh_db - (-24.0)) / (-8.0 - (-24.0)), 0.0, 1.0)
    b = np.clip((ratio_db - 2.0) / (14.0 - 2.0), 0.0, 1.0)
    rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=-1)
    rgb_8u = (rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb_8u, mode="RGB").convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def execute_pipeline(im_start: np.ndarray, stages: list[Any]) -> np.ndarray:
    curr = im_start.copy()
    for s in stages:
        if isinstance(s, dict):
            op = s.get("op", "").strip().upper()
            kw = s.get("kwargs", {})
        else:
            op = str(s).strip().upper()
            kw = {}
        if op in OPERATORS:
            curr = OPERATORS[op](curr, **kw)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curr

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
                        raise ValueError(f"Неизвестный эталон: '{target_id}'. Доступны: {list(BENCH_CACHE.keys())}")
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
                        b64 = rgb_composite_base64_calibrated(vv_db, vh_db, ratio_db)
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
