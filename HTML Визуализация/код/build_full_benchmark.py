# -*- coding: utf-8 -*-
"""Главный исполнительный модуль сборки интерактивного HTML-бенчмарка Sentinel-1 SAR.
Полностью отвязан от файловой системы: все пути и полигоны импортируются из config.py.
Все алгоритмы фильтрации подключаются динамически через единый реестр FilterRegistry.

Сборка запускается командой:
    python build_full_benchmark.py

Результат:
    Генерируется единственный канонический файл:
    HTML Визуализация/html/index.html
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

# Подключение конфигурации и реестра
try:
    from .config import (
        ROOT_DIR, DATA_DIR, DATA_CACHE_DIR, PAPER_DIR, WEIGHTS_DIR,
        RESULTS_DIR, HTML_DIR, OUTPUT_HTML_PATHS, SCENES_REGISTRY,
        WEBP_QUALITY, BENCHMARK_RUNS, load_scene_rasters, get_available_temporal_keys
    )
    from .filter_registry import (
        FilterRegistry, compute_metrics, array_to_webp_base64,
        rgb_composite_webp_base64, array_to_png_base64, rgb_composite_base64,
        benchmark_filter
    )
    from .html_template import HTML_TEMPLATE
    import filters_suite
except ImportError:
    from config import (
        ROOT_DIR, DATA_DIR, DATA_CACHE_DIR, PAPER_DIR, WEIGHTS_DIR,
        RESULTS_DIR, HTML_DIR, OUTPUT_HTML_PATHS, SCENES_REGISTRY,
        WEBP_QUALITY, BENCHMARK_RUNS, load_scene_rasters, get_available_temporal_keys
    )
    from filter_registry import (
        FilterRegistry, compute_metrics, array_to_webp_base64,
        rgb_composite_webp_base64, array_to_png_base64, rgb_composite_base64,
        benchmark_filter
    )
    from html_template import HTML_TEMPLATE
    import filters_suite

sys.stdout.reconfigure(encoding='utf-8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 1. ОБРАБОТКА ЭТАЛОНОВ ИЗ СТАТЕЙ (arXiv)
# -----------------------------------------------------------------------------

def load_b64(fname: str) -> str:
    full_p = os.path.join(PAPER_DIR, fname)
    if not os.path.exists(full_p):
        return ""
    with open(full_p, "rb") as f:
        data = f.read()
    ext = os.path.splitext(fname)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def process_bench_case(
    bench_id: str,
    paper_name: str,
    title: str,
    desc: str,
    strip_fname: str,
    input_fname: str,
    paper_outputs: Dict[str, Tuple[str, str]],
    gt_fname: Optional[str] = None
) -> Dict[str, Any]:
    """Обрабатывает один эталонный тестовый случай из научной статьи."""
    print(f"\n[ЭТАЛОН] Обработка тестового случая: {bench_id} ({paper_name})...")

    inp_path = os.path.join(PAPER_DIR, input_fname)
    im_in_u8 = Image.open(inp_path).convert("L")
    im_in = np.array(im_in_u8).astype(np.float32) / 255.0

    im_gt = None
    if gt_fname:
        gt_path = os.path.join(PAPER_DIR, gt_fname)
        if os.path.exists(gt_path):
            im_gt = np.array(Image.open(gt_path).convert("L")).astype(np.float32) / 255.0

    pairs_dict = {}

    # Запуск всех методов из реестра FilterRegistry на эталонном изображении
    all_filters = FilterRegistry.list_all()
    for p_id, spec in all_filters.items():
        try:
            out_vv, _ = spec.execute(im_in, im_in)
            out_vv = np.clip(out_vv, 0.0, 1.0)
            u8 = (out_vv * 255.0).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(u8, mode="L").save(buf, format="PNG")
            b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            m = compute_metrics(im_in, out_vv, gt_lin=im_gt, looks=4.4)
            m["time_ms"] = benchmark_filter(spec, im_in, im_in, n_runs=BENCHMARK_RUNS)

            pairs_dict[p_id] = {
                "name": spec.name,
                "img": b64_str,
                "metrics": m
            }
        except Exception as e:
            print(f"  [ПРЕДУПРЕЖДЕНИЕ] Ошибка выполнения {p_id} на эталоне {bench_id}: {e}")

    # Добавление результатов, опубликованных авторами статей
    for p_key, (p_label, p_file) in paper_outputs.items():
        full_p = os.path.join(PAPER_DIR, p_file)
        if os.path.exists(full_p):
            p_im_u8 = Image.open(full_p).convert("L")
            p_arr = np.array(p_im_u8).astype(np.float32) / 255.0
            p_gt_arg = im_gt if (im_gt is not None and im_gt.shape == p_arr.shape) else None
            m_paper = compute_metrics(im_in, p_arr, gt_lin=p_gt_arg, looks=4.4)
            m_paper["time_ms"] = 0.0
            pairs_dict[p_key] = {
                "name": p_label,
                "img": load_b64(p_file),
                "metrics": m_paper
            }

    if gt_fname and "gt" not in pairs_dict:
        m_gt = compute_metrics(im_in, im_gt, gt_lin=im_gt, looks=4.4) if im_gt is not None else {}
        m_gt["time_ms"] = 0.0
        pairs_dict["gt"] = {
            "name": "Эталон: Ground Truth",
            "img": load_b64(gt_fname),
            "metrics": m_gt
        }

    return {
        "id": bench_id,
        "paper_name": paper_name,
        "title": title,
        "desc": desc,
        "strip_img": load_b64(strip_fname),
        "input_img": load_b64(input_fname),
        "gt_img": load_b64(gt_fname) if gt_fname else None,
        "pairs": pairs_dict
    }


# -----------------------------------------------------------------------------
# 2. ГЛАВНЫЙ ЦИКЛ ГЕНЕРАЦИИ
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("СБОРКА АВТОНОМНОГО SAR АНАЛИЗАТОРА: РЕЕСТР ПОЛИГОНОВ x РЕЕСТР ФИЛЬТРОВ")
    print("=" * 70)
    print(f"[ИНФОРМАЦИЯ] Корень проекта: {ROOT_DIR}")
    print(f"[ИНФОРМАЦИЯ] Каталог данных: {DATA_DIR}")
    print(f"[ИНФОРМАЦИЯ] Каталог весов: {WEIGHTS_DIR}")
    print(f"[ИНФОРМАЦИЯ] Устройство: {device}")
    if torch.cuda.is_available():
        print(f"[ИНФОРМАЦИЯ] GPU: {torch.cuda.get_device_name(0)}")

    all_filters = FilterRegistry.list_all()
    print(f"[ИНФОРМАЦИЯ] Зарегистрировано фильтров в реестре: {len(all_filters)}")
    for fid, spec in all_filters.items():
        print(f"  - [{spec.category}] {fid}: {spec.name}")

    app_data: Dict[str, Any] = {}

    # 1. Обработка всех зарегистрированных полигонов
    print(f"\n[ИНФОРМАЦИЯ] Зарегистрировано полигонов в реестре: {len(SCENES_REGISTRY)}")
    for key, sc in SCENES_REGISTRY.items():
        print(f"\n---> Обработка полигона: {sc.title} ({key})...")
        temporal_keys = get_available_temporal_keys(sc)
        print(f"  Доступные временные срезы / даты ({len(temporal_keys)}): {temporal_keys}")

        seasons_dict: Dict[str, Any] = {}

        for t_key in temporal_keys:
            print(f"  -> Расчет фильтрации для среза: '{t_key}'...")
            try:
                vv_raw, vh_raw, bbox, vv_gt, vh_gt = load_scene_rasters(sc, temporal_key=t_key)
            except Exception as e:
                print(f"     [ОШИБКА ЗАГРУЗКИ] {e}, пропуск среза.")
                continue

            has_gt = vv_gt is not None
            s_pipe_dict: Dict[str, Any] = {}

            # Запуск каждого зарегистрированного фильтра
            for p_id, spec in all_filters.items():
                out_vv, out_vh = spec.execute(vv_raw, vh_raw)

                vv_db = 10.0 * np.log10(np.maximum(out_vv, 1e-6))
                vh_db = 10.0 * np.log10(np.maximum(out_vh, 1e-6))
                ratio_db = vv_db - vh_db
                res_ratio = vv_raw / np.maximum(out_vv, 1e-8) if p_id != "raw" else np.ones_like(vv_raw)

                m = compute_metrics(
                    vv_raw, out_vv,
                    gt_lin=(vv_gt if (t_key == "spring" and has_gt) else None),
                    looks=4.4
                )
                m["time_ms"] = benchmark_filter(spec, vv_raw, vh_raw, n_runs=BENCHMARK_RUNS)

                s_pipe_dict[p_id] = {
                    "name": spec.name,
                    "metrics": m,
                    "layers": {
                        "vv": array_to_webp_base64(vv_db, -25.0, 0.0, quality=WEBP_QUALITY),
                        "vh": array_to_webp_base64(vh_db, -32.0, -8.0, quality=WEBP_QUALITY),
                        "ratio": array_to_webp_base64(ratio_db, 0.0, 20.0, cmap_name="viridis", quality=WEBP_QUALITY),
                        "rgb": rgb_composite_webp_base64(vv_db, vh_db, ratio_db, quality=WEBP_QUALITY),
                        "res_ratio": array_to_webp_base64(res_ratio, 0.0, 2.5, cmap_name="coolwarm", quality=WEBP_QUALITY),
                    }
                }

            # Добавление эталона Ground Truth если доступен
            if t_key == "spring" and has_gt:
                gt_vv_db = 10.0 * np.log10(np.maximum(vv_gt, 1e-6))
                gt_vh_db = 10.0 * np.log10(np.maximum(vh_gt, 1e-6))
                gt_ratio_db = gt_vv_db - gt_vh_db
                gt_res = vv_raw / np.maximum(vv_gt, 1e-8)

                m_gt = compute_metrics(vv_raw, vv_gt, gt_lin=vv_gt, looks=4.4)
                m_gt["time_ms"] = 0.0
                s_pipe_dict["multitemp_gt"] = {
                    "name": "17. GEE Multi-Temporal Ground Truth (Эталон)",
                    "metrics": m_gt,
                    "layers": {
                        "vv": array_to_webp_base64(gt_vv_db, -25.0, 0.0, quality=WEBP_QUALITY),
                        "vh": array_to_webp_base64(gt_vh_db, -32.0, -8.0, quality=WEBP_QUALITY),
                        "ratio": array_to_webp_base64(gt_ratio_db, 0.0, 20.0, cmap_name="viridis", quality=WEBP_QUALITY),
                        "rgb": rgb_composite_webp_base64(gt_vv_db, gt_vh_db, gt_ratio_db, quality=WEBP_QUALITY),
                        "res_ratio": array_to_webp_base64(gt_res, 0.0, 2.5, cmap_name="coolwarm", quality=WEBP_QUALITY),
                    }
                }

            seasons_dict[t_key] = s_pipe_dict

        # Координатные границы Leaflet: [[lat_min, lon_min], [lat_max, lon_max]]
        leaflet_bounds = [[sc.bbox[1], sc.bbox[0]], [sc.bbox[3], sc.bbox[2]]]
        default_temporal = "spring" if "spring" in seasons_dict else list(seasons_dict.keys())[0]

        app_data[key] = {
            "title": sc.title,
            "desc": sc.desc,
            "center": sc.center,
            "zoom": sc.zoom,
            "bounds": leaflet_bounds,
            "shape": list(seasons_dict[default_temporal]["raw"]["layers"]["vv"].__len__()),  # информационная длина
            "pipelines": seasons_dict[default_temporal],
            "seasons": seasons_dict,
        }
        print(f"  [ГОТОВО] Полигон {key}: {len(seasons_dict[default_temporal])} фильтров x {len(seasons_dict)} срезов.")

    # 2. Обработка эталонов статей (arXiv)
    print("\n[ИНФОРМАЦИЯ] Обработка эталонных тестовых случаев из научных статей...")
    app_data["_benchmarks"] = [
        process_bench_case(
            bench_id="vitale",
            paper_name="Vitale et al. (arXiv:2104.09350v2, IEEE JSTARS 2021)",
            title="Беневенто: Синтетический спекл vs CNN-модели (SAR-ResNet, SAR-CNN)",
            desc="Сравнение на реалистичном синтетическом изображении с известным Ground Truth. Тестирование сохранения тонких линий, мостов и текстур.",
            strip_fname="strip_vitale.png",
            input_fname="Vitale_SAR_ResNet_p8_img0_256x256.png",
            paper_outputs={
                "paper_resnet": ("Авторы: SAR-ResNet (Vitale et al.)", "Vitale_SAR_ResNet_p8_img1_256x256.png"),
                "paper_sar_cnn": ("Базовый: SAR-CNN (Chierchia et al.)", "Vitale_SAR_ResNet_p8_img2_256x256.png"),
                "paper_lee": ("Базовый: Lee Filter 5x5", "Vitale_SAR_ResNet_p8_img4_256x256.png")
            },
            gt_fname="Vitale_SAR_ResNet_p8_img3_256x256.png"
        ),
        process_bench_case(
            bench_id="merlin",
            paper_name="Dalsasso et al. (arXiv:2110.13148v2, IEEE TGRS 2021)",
            title="Сер-Понсон: Blind-Spot архитектура MERLIN на реальном Sentinel-1",
            desc="Реальный горный полигон водохранилища Сер-Понсон в Альпах. Сравнение самообучающейся сети MERLIN со Speckle2Void и SAR-BM3D.",
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
            desc="Реальный снимок. Авторы сравнивают нелокальный фильтр PPB, вейвлетный SAR-BM3D, сверточный SAR-CNN, дилатированный ID-CNN и трансформер SAR-Restormer.",
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
            desc="Прямое попиксельное сопоставление реального снимка Capella Space сверхвысокого разрешения, оптического эталона Ground Truth и результата нейросети SEDRUNet.",
            strip_fname="strip_sim2real.png",
            input_fname="Sim2Real_2026_p6_img1_308x308.png",
            paper_outputs={
                "paper_sedrunet": ("Авторы: SEDRUNet (Al-Husseini et al. 2026)", "Sim2Real_2026_p6_img2_308x308.png"),
                "gt": ("Эталон: Оптический снимок высокого разрешения", "Sim2Real_2026_p6_img0_308x308.png")
            },
            gt_fname="Sim2Real_2026_p6_img0_308x308.png"
        )
    ]

    # 3. Сериализация в единый JSON и внедрение в HTML шаблон
    print("\n[ИНФОРМАЦИЯ] Сериализация данных в JSON...")
    data_json = json.dumps(app_data)

    print("[ИНФОРМАЦИЯ] Внедрение данных в HTML_TEMPLATE...")
    html_content = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    # 4. Сохранение единственного канонического файла index.html
    os.makedirs(HTML_DIR, exist_ok=True)
    canonical_output = OUTPUT_HTML_PATHS[0]
    with open(canonical_output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[УСПЕХ] Сгенерирован канонический HTML-файл:")
    print(f"       -> {canonical_output}")
    print(f"       -> Размер: {os.path.getsize(canonical_output):,} байт")
    print(f"[УСПЕХ] Дубликаты исключены. В репозитории поддерживается строго 1 канонический HTML.")


if __name__ == "__main__":
    main()
