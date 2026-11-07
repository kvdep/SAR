# -*- coding: utf-8 -*-
"""Автономный сервер Sentinel-1 SAR с GPU-ускорением.
Полностью инкапсулирован:
1. Использует единый реестр FilterRegistry и модуль filters_suite для исполнения фильтров.
2. Использует единый реестр полигонов SCENES_REGISTRY и load_scene_rasters для загрузки данных.
3. Полностью исключает дублирование архитектур нейросетей, весов и математических операторов.

Запуск:
    python sar_server.py [порт, по умолчанию 8080]
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

sys.stdout.reconfigure(encoding='utf-8')

# Подключение конфигурации и реестра
try:
    from .config import (
        ROOT_DIR, DATA_DIR, DATA_CACHE_DIR, PAPER_DIR, WEIGHTS_DIR,
        RESULTS_DIR, HTML_DIR, SCENES_REGISTRY, load_scene_rasters,
        get_available_temporal_keys
    )
    from .filter_registry import (
        FilterRegistry, compute_metrics, array_to_webp_base64,
        rgb_composite_webp_base64, array_to_png_base64, rgb_composite_base64
    )
    import filters_suite
except ImportError:
    from config import (
        ROOT_DIR, DATA_DIR, DATA_CACHE_DIR, PAPER_DIR, WEIGHTS_DIR,
        RESULTS_DIR, HTML_DIR, SCENES_REGISTRY, load_scene_rasters,
        get_available_temporal_keys
    )
    from filter_registry import (
        FilterRegistry, compute_metrics, array_to_webp_base64,
        rgb_composite_webp_base64, array_to_png_base64, rgb_composite_base64
    )
    import filters_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[СЕРВЕР] Вычислительное устройство: {device}")
if torch.cuda.is_available():
    print(f"[СЕРВЕР] Графический процессор: {torch.cuda.get_device_name(0)}")

# Псевдонимы коротких кодов интерфейса к идентификаторам FilterRegistry
ALIASES: Dict[str, str] = {
    "bxc": "boxcar",
    "med": "median",
    "lee": "lee",
    "enh_lee": "enhanced_lee",
    "kua": "kuan",
    "fro": "frost",
    "enh_fro": "enhanced_frost",
    "bil": "bilateral",
    "gmap": "gamma_map",
    "ftv": "joint_ftv",
    "res": "resnet",
    "mer": "mer_dense",
    "mer_dense": "mer_dense",
    "mer_cbam": "mer_cbam",
    "mer_res": "mer_res",
    "mer_base": "mer_base",
    "mer_blind": "mer_blind",
    "mer_gammamle": "mer_gammamle",
    "mer_extremedeep": "mer_extremedeep",
    "rst": "restormer",
    "rst_v2": "restormer",
    "yolo26": "yolo26_s",
    "yolo26_s": "yolo26_s",
    "yolo26_n": "yolo26_n",
    "rpca": "rpca_ialm",
    "rpca_s": "rpca_sparse",
}

# Кэш эталонных снимков из научных статей (arXiv)
BENCH_CACHE: Dict[str, Dict[str, Any]] = {}
BENCH_SPECS = {
    "vitale": {
        "raw": "Vitale_SAR_ResNet_p8_img0_256x256.png",
        "gt": "Vitale_SAR_ResNet_p8_img3_256x256.png",
    },
    "merlin": {
        "raw": "MERLIN_Dalsasso_p9_img0_512x512.png",
        "gt": None,
    },
    "restormer": {
        "raw": "Restormer_SAR_p4_img0_512x512.jpeg",
        "gt": None,
    },
    "sim2real": {
        "raw": "Sim2Real_2026_p6_img1_308x308.png",
        "gt": "Sim2Real_2026_p6_img0_308x308.png",
    }
}

for b_id, b_info in BENCH_SPECS.items():
    raw_p = os.path.join(PAPER_DIR, b_info["raw"])
    gt_p = os.path.join(PAPER_DIR, b_info["gt"]) if b_info["gt"] else None
    if os.path.exists(raw_p):
        raw_arr = np.array(Image.open(raw_p).convert("L")).astype(np.float32) / 255.0
        gt_arr = np.array(Image.open(gt_p).convert("L")).astype(np.float32) / 255.0 if (gt_p and os.path.exists(gt_p)) else None
        BENCH_CACHE[b_id] = {"raw": raw_arr, "gt": gt_arr}

print(f"[СЕРВЕР] Инициализирован кэш эталонов статей: {list(BENCH_CACHE.keys())}")
print(f"[СЕРВЕР] Доступно полигонов в SCENES_REGISTRY: {len(SCENES_REGISTRY)}")
print(f"[СЕРВЕР] Зарегистрировано фильтров в FilterRegistry: {len(FilterRegistry.list_all())}")


def execute_pipeline(im_vv: np.ndarray, im_vh: Optional[np.ndarray], stages: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Последовательно применяет цепочку стадий фильтрации к поляризациям VV и VH."""
    curr_vv = im_vv.copy()
    curr_vh = im_vh.copy() if im_vh is not None else im_vv.copy()

    for s in stages:
        if isinstance(s, dict):
            op = str(s.get("op", "")).strip().lower()
            kw = s.get("kwargs", {})
        else:
            op = str(s).strip().lower()
            kw = {}

        filter_key = ALIASES.get(op, op)
        if filter_key in FilterRegistry.list_all():
            spec = FilterRegistry.get(filter_key)
            curr_vv, curr_vh = spec.execute(curr_vv, curr_vh, **kw)
        else:
            print(f"[СЕРВЕР ПРЕДУПРЕЖДЕНИЕ] Неизвестный оператор '{op}', пропуск этапа.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return curr_vv, curr_vh


class SARRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HTML_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/filters":
            meta = FilterRegistry.get_metadata()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(meta, ensure_ascii=False).encode('utf-8'))
        elif self.path == "/api/scenes":
            scenes_data = {k: sc.to_dict() for k, sc in SCENES_REGISTRY.items()}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(scenes_data, ensure_ascii=False).encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path in ("/api/pipeline", "/api/custom_pipeline"):
            t0 = time.perf_counter()
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')

            try:
                req = json.loads(body)
                mode = req.get("mode", "gis")  # "gis" | "bench"
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

                    if stages and stages != ["RAW"]:
                        filt_vv, _ = execute_pipeline(im_raw, im_raw, stages)
                    else:
                        filt_vv = im_raw.copy()

                    m = compute_metrics(im_raw, filt_vv, gt_lin=im_gt)
                    norm = np.clip(filt_vv, 0.0, 1.0)
                    u8 = (norm * 255.0).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(u8, mode="L").save(buf, format="PNG")
                    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                    date_str = "Научный эталон (arXiv)"
                else:
                    # GIS mode
                    if target_id not in SCENES_REGISTRY:
                        raise ValueError(f"Неизвестный полигон: '{target_id}'. Доступны: {list(SCENES_REGISTRY.keys())}")

                    sc = SCENES_REGISTRY[target_id]
                    vv_in, vh_in, bbox, vv_gt, vh_gt = load_scene_rasters(sc, temporal_key=season)

                    if stages and stages != ["RAW"]:
                        filt_vv, filt_vh = execute_pipeline(vv_in, vh_in, stages)
                    else:
                        filt_vv = vv_in.copy()
                        filt_vh = vh_in.copy()

                    vv_db = 10.0 * np.log10(np.maximum(filt_vv, 1e-6))
                    vh_db = 10.0 * np.log10(np.maximum(filt_vh, 1e-6))
                    ratio_db = vv_db - vh_db

                    if layer == "rgb":
                        b64 = rgb_composite_base64(vv_db, vh_db, ratio_db)
                    elif layer == "ratio":
                        b64 = array_to_png_base64(ratio_db, 0.0, 20.0, cmap_name="viridis")
                    elif layer == "res_ratio":
                        res_map = vv_in / np.maximum(filt_vv, 1e-6)
                        b64 = array_to_png_base64(res_map, 0.0, 2.0, cmap_name="coolwarm")
                    elif layer == "vh":
                        b64 = array_to_png_base64(vh_db, -32.0, -8.0)
                    else:  # "vv"
                        b64 = array_to_png_base64(vv_db, -25.0, 0.0)

                    m = compute_metrics(vv_in, filt_vv, gt_lin=(vv_gt if season == "spring" else None))
                    date_str = f"Полигон: {sc.title} | Срез: {season}"

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
    print("=" * 70)
    print(f"[СЕРВЕР] Готов к приему запросов: http://localhost:{port}/")
    print(f"[СЕРВЕР] Обслуживаемый каталог HTML: {HTML_DIR}")
    print(f"[СЕРВЕР] Доступные фильтры: {len(FilterRegistry.list_all())} алгоритмов")
    print("=" * 70)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
