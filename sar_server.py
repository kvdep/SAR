# -*- coding: utf-8 -*-
"""Автономный сервер Sentinel-1 SAR с GPU-ускорением (RTX 3080)
для произвольных конвейеров фильтрации любой длины.

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
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[СЕРВЕР] Вычислительное устройство: {device}")
if torch.cuda.is_available():
    print(f"[СЕРВЕР] Графический процессор: {torch.cuda.get_device_name(0)}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(BASE_DIR, "weights")
paper_dir = os.path.join(BASE_DIR, "paper_results")
data_cache_dir = os.path.join(BASE_DIR, "data_cache")

# Загрузка моделей и фильтров
from train_sar_models import SAR_ResNet, SAR_MERLIN_BlindSpot, SAR_Restormer
from build_full_benchmark import (
    filter_boxcar_mean, filter_median, filter_lee, filter_enhanced_lee,
    filter_kuan, filter_frost, filter_enhanced_frost, filter_bilateral,
    filter_gamma_map, JointFTVSARP, compute_metrics, array_to_png_base64,
    rgb_composite_base64
)

model_res = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device).eval()
model_res.load_state_dict(torch.load(os.path.join(weights_dir, "sar_resnet_vitale.pth"), map_location=device))

model_mer = SAR_MERLIN_BlindSpot(in_channels=1, num_features=48).to(device).eval()
model_mer.load_state_dict(torch.load(os.path.join(weights_dir, "sar_merlin_blindspot.pth"), map_location=device))

model_rst = SAR_Restormer(in_channels=1, base_dim=32).to(device).eval()
model_rst.load_state_dict(torch.load(os.path.join(weights_dir, "sar_restormer.pth"), map_location=device))

solver_ftv = JointFTVSARP(alpha=1.25, K=11, device=device)

def run_dl(im_np: np.ndarray, model) -> np.ndarray:
    with torch.no_grad():
        scale = float(np.percentile(im_np, 99.5))
        if scale < 1e-4: scale = float(im_np.max())
        scale = max(scale, 1e-4)
        t = torch.from_numpy((im_np / scale).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            out = model(t)
        res = out.squeeze().cpu().numpy() * scale
    return np.maximum(res, 1e-6)

def run_ftv(im_np: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(im_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = solver_ftv.solve(t, n_iter=15)
    return out[0, 0].cpu().numpy()

OPERATORS = {
    "BXC": lambda x: filter_boxcar_mean(x, w=5),
    "MED": lambda x: filter_median(x, w=5),
    "LEE": lambda x: filter_lee(x, w=5, looks=4.4),
    "ENH_LEE": lambda x: filter_enhanced_lee(x, w=5, looks=4.4),
    "KUA": lambda x: filter_kuan(x, w=5, looks=4.4),
    "FRO": lambda x: filter_frost(x, w=5, damping=2.0),
    "ENH_FRO": lambda x: filter_enhanced_frost(x, w=5, looks=4.4, damping=2.0),
    "BIL": lambda x: filter_bilateral(x),
    "GMAP": lambda x: filter_gamma_map(x, w=5, looks=4.4),
    "FTV": run_ftv,
    "RES": lambda x: run_dl(x, model_res),
    "MER": lambda x: run_dl(x, model_mer),
    "RST": lambda x: run_dl(x, model_rst),
}

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

# Загрузка ВСЕХ 8 геопривязанных сцен строго из data_cache/raw_{k}.npz
GIS_KEYS = ["benevento", "rome", "serre_poncon", "forest", "city", "water", "desert", "mountain"]
for k in GIS_KEYS:
    p = os.path.join(data_cache_dir, f"raw_{k}.npz")
    if os.path.exists(p):
        d = np.load(p)
        gt_lin = d["vv_gt"] if ("vv_gt" in d and d["vv_gt"] is not None and getattr(d["vv_gt"], "ndim", 0) > 0) else None
        RAW_CACHE[k] = { "vv": d["vv"], "vh": d["vh"], "gt": gt_lin }
    else:
        print(f"[СЕРВЕР ОШИБКА] Файл полигона {p} не найден!")

print(f"[СЕРВЕР] Инициализирован кэш GIS полигонов ({len(RAW_CACHE)}/8): {list(RAW_CACHE.keys())}")
print(f"[СЕРВЕР] Инициализирован кэш бенчмарков: {list(BENCH_CACHE.keys())}")

def execute_pipeline(im_start: np.ndarray, stages: list[str]) -> np.ndarray:
    curr = im_start.copy()
    for s in stages:
        op = s.strip().upper()
        if op in OPERATORS:
            curr = OPERATORS[op](curr)
    return curr

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
                stages = req.get("stages", ["FTV", "RES"])
                layer = req.get("layer", "rgb")

                if mode == "bench":
                    if target_id not in BENCH_CACHE:
                        raise ValueError(f"Неизвестный бенчмарк: '{target_id}'. Доступны: {list(BENCH_CACHE.keys())}")
                    b_data = BENCH_CACHE[target_id]
                    im_raw = b_data["raw"]
                    im_gt = b_data["gt"]
                    im_filt = execute_pipeline(im_raw, stages)
                    m = compute_metrics(im_raw, im_filt, gt_lin=im_gt)
                    norm = np.clip(im_filt, 0.0, 1.0)
                    u8 = (norm * 255.0).astype(np.uint8)
                    img = Image.fromarray(u8, mode="L")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                else:
                    # GIS mode
                    if target_id not in RAW_CACHE:
                        raise ValueError(f"Неизвестный полигон: '{target_id}'. Доступны: {list(RAW_CACHE.keys())}")
                    sc_data = RAW_CACHE[target_id]
                    
                    if layer == "rgb":
                        im_vv_filt = execute_pipeline(sc_data["vv"], stages)
                        im_vh_filt = execute_pipeline(sc_data["vh"], stages)
                        vv_db = 10.0 * np.log10(np.maximum(im_vv_filt, 1e-6))
                        vh_db = 10.0 * np.log10(np.maximum(im_vh_filt, 1e-6))
                        ratio_db = vv_db - vh_db
                        b64 = rgb_composite_base64(vv_db, vh_db, ratio_db)
                        m = compute_metrics(sc_data["vv"], im_vv_filt, gt_lin=sc_data.get("gt"))
                    elif layer == "ratio":
                        im_vv_filt = execute_pipeline(sc_data["vv"], stages)
                        im_vh_filt = execute_pipeline(sc_data["vh"], stages)
                        vv_db = 10.0 * np.log10(np.maximum(im_vv_filt, 1e-6))
                        vh_db = 10.0 * np.log10(np.maximum(im_vh_filt, 1e-6))
                        ratio_db = vv_db - vh_db
                        b64 = array_to_png_base64(ratio_db, 2.0, 14.0)
                        m = compute_metrics(sc_data["vv"], im_vv_filt, gt_lin=sc_data.get("gt"))
                    elif layer == "res_ratio":
                        im_vv_filt = execute_pipeline(sc_data["vv"], stages)
                        res_map = sc_data["vv"] / np.maximum(im_vv_filt, 1e-6)
                        b64 = array_to_png_base64(res_map, 0.0, 2.0, cmap_name="coolwarm")
                        m = compute_metrics(sc_data["vv"], im_vv_filt, gt_lin=sc_data.get("gt"))
                    else:
                        im_raw = sc_data["vv"] if layer != "vh" else sc_data["vh"]
                        im_filt = execute_pipeline(im_raw, stages)
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
                    "stages": stages,
                    "layer": layer,
                    "metrics": m,
                    "img": b64
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port: int = 8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SARRequestHandler)
    print(f"[СЕРВЕР] Запущен на http://localhost:{port}/")
    print(f"[СЕРВЕР] Доступен API конвейера: POST http://localhost:{port}/api/pipeline")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[СЕРВЕР] Остановлен.")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
