# -*- coding: utf-8 -*-
"""Скрипт полного комбинаторного перебора длинных конвейеров фильтрации:
- 1-шаговые (12 базовых операторов)
- 2-шаговые (12 x 12 = 144 комбинации)
- 3-шаговые (12 x 12 x 12 = 1728 комбинаций)
Оценка по полному набору метрик: PSNR, SSIM, ENL, EPI, Bias, Var(r), Composite Score.
Тестирование на эталоне Vitale et al. (с известным Ground Truth) и на реальном лесе РФ.
"""

import os
import sys
import time
import json
import math
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ИНФОРМАЦИЯ] Устройство: {device}")
if torch.cuda.is_available():
    print(f"[ИНФОРМАЦИЯ] GPU: {torch.cuda.get_device_name(0)}")

# =========================================================================
# 1. ЗАГРУЗКА МОДЕЛЕЙ И ФИЛЬТРОВ
# =========================================================================
from train_sar_models import SAR_ResNet, SAR_MERLIN_BlindSpot, SAR_Restormer
from build_full_benchmark import (
    filter_boxcar_mean, filter_median, filter_lee, filter_enhanced_lee,
    filter_kuan, filter_frost, filter_enhanced_frost, filter_bilateral,
    filter_gamma_map, JointFTVSARP
)

weights_dir = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\weights"
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
    t = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = solver_ftv.solve(t, n_iter=15)
    return out[0, 0].cpu().numpy()

# 12 базовых операторов
OPERATORS = {
    "BXC": ("Boxcar 5x5", lambda x: filter_boxcar_mean(x, w=5)),
    "MED": ("Median 5x5", lambda x: filter_median(x, w=5)),
    "LEE": ("Lee 5x5", lambda x: filter_lee(x, w=5, looks=4.4)),
    "ENH_LEE": ("EnhLee 5x5", lambda x: filter_enhanced_lee(x, w=5, looks=4.4)),
    "KUA": ("Kuan 5x5", lambda x: filter_kuan(x, w=5, looks=4.4)),
    "FRO": ("Frost 5x5", lambda x: filter_frost(x, w=5, damping=2.0)),
    "ENH_FRO": ("EnhFrost 5x5", lambda x: filter_enhanced_frost(x, w=5, looks=4.4, damping=2.0)),
    "BIL": ("Bilateral", lambda x: filter_bilateral(x)),
    "GMAP": ("GammaMAP 5x5", lambda x: filter_gamma_map(x, w=5, looks=4.4)),
    "FTV": ("FTV-SARP", run_ftv),
    "RES": ("SAR-ResNet", lambda x: run_dl(x, model_res)),
    "MER": ("MERLIN", lambda x: run_dl(x, model_mer)),
}

# =========================================================================
# 2. ВЫЧИСЛЕНИЕ МЕТРИК
# =========================================================================

def eval_metrics(im_raw: np.ndarray, im_filt: np.ndarray, im_gt: np.ndarray | None = None, looks: float = 4.4) -> dict:
    im_c = np.maximum(im_filt, 1e-6)
    raw_c = np.maximum(im_raw, 1e-6)

    # Однородные области для ENL
    mu = ndimage.uniform_filter(im_c, size=7)
    sq = ndimage.uniform_filter(im_c**2, size=7)
    sig = np.sqrt(np.maximum(sq - mu**2, 1e-8))
    ci = sig / (mu + 1e-8)
    cu = 1.0 / math.sqrt(looks)
    homo = (ci < 1.15 * cu)
    if np.sum(homo) > 50:
        enl = float(np.mean(im_c[homo])**2 / (np.var(im_c[homo]) + 1e-9))
    else:
        enl = float(np.mean(im_c)**2 / (np.var(im_c) + 1e-9))

    # EPI
    l_raw = ndimage.laplace(raw_c).flatten()
    l_filt = ndimage.laplace(im_c).flatten()
    std_r, std_f = l_raw.std(), l_filt.std()
    if std_r > 1e-8 and std_f > 1e-8:
        epi = float(np.corrcoef(l_raw, l_filt)[0, 1])
    else:
        epi = 1.0

    # Bias
    bias_db = float(10.0 * np.log10(max(1e-8, float(np.mean(im_c))) / max(1e-8, float(np.mean(raw_c)))))

    # Ratio
    r = np.clip(raw_c / im_c, 0.0, 30.0)
    var_r = float(np.var(r))

    res = {
        "enl": round(enl, 2),
        "epi": round(epi, 4),
        "bias_db": round(bias_db, 3),
        "var_r": round(var_r, 4),
    }

    if im_gt is not None:
        gt_c = np.maximum(im_gt, 1e-6)
        mse = float(np.mean((gt_c - im_c)**2))
        dr = float(gt_c.max() - gt_c.min())
        psnr = float(10.0 * np.log10((dr**2) / (mse + 1e-9))) if mse > 0 else 99.9

        mu_x = float(np.mean(gt_c))
        mu_y = float(np.mean(im_c))
        sig_x = float(np.var(gt_c))
        sig_y = float(np.var(im_c))
        sig_xy = float(np.mean((gt_c - mu_x) * (im_c - mu_y)))
        c1 = (0.01 * dr)**2
        c2 = (0.03 * dr)**2
        ssim = float(((2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2)))

        res["psnr"] = round(psnr, 2)
        res["ssim"] = round(ssim, 4)
        res["score"] = round(psnr + 20.0 * ssim + 10.0 * epi - 5.0 * abs(bias_db), 2)
    else:
        res["score"] = round(enl * 0.1 + 10.0 * epi - 5.0 * abs(bias_db), 2)

    return res

# =========================================================================
# 3. ОСНОВНОЙ ЦИКЛ ПОИСКА
# =========================================================================

def main():
    p_noisy = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\paper_results\Vitale_SAR_ResNet_p8_img1_256x256.png"
    p_gt = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\paper_results\Vitale_SAR_ResNet_p8_img0_256x256.png"
    im_raw = np.array(Image.open(p_noisy).convert("L")).astype(np.float32) / 255.0
    im_gt = np.array(Image.open(p_gt).convert("L")).astype(np.float32) / 255.0

    print("=" * 80)
    print("ЭКСПЕРИМЕНТ 1: ПОЛНЫЙ ПЕРЕБОР 1-, 2- И 3-ШАГОВЫХ ПАЙПЛАЙНОВ (БЕНЧМАРК VITALE)")
    print(f"Размер изображения: {im_raw.shape}, Базовых операторов: {len(OPERATORS)}")
    print("=" * 80)

    # Базовые метрики входного зашумленного изображения
    m_raw = eval_metrics(im_raw, im_raw, im_gt)
    print(f"Входной сигнал: PSNR={m_raw['psnr']} дБ, SSIM={m_raw['ssim']}, ENL={m_raw['enl']}, EPI={m_raw['epi']}\n")

    # ШАГ 1: 1-шаговые пайплайны
    t0 = time.time()
    cache_1step = {}
    results_1step = []
    print("[1/3] Вычисление 1-шаговых операторов (N=12)...")
    for k, (name, fn) in OPERATORS.items():
        out = fn(im_raw)
        cache_1step[k] = out
        m = eval_metrics(im_raw, out, im_gt)
        results_1step.append({
            "pipeline": k,
            "name": name,
            "steps": 1,
            **m
        })
    results_1step.sort(key=lambda x: x["score"], reverse=True)
    print(f"  Вычислено 12 вариантов за {time.time()-t0:.2f} с.")

    # ШАГ 2: 2-шаговые пайплайны (12 x 12 = 144)
    t0 = time.time()
    cache_2step = {}
    results_2step = []
    print("\n[2/3] Вычисление 2-шаговых комбинаций (12 x 12 = 144)...")
    for k1, out1 in cache_1step.items():
        for k2, (name2, fn2) in OPERATORS.items():
            pipe_key = f"{k1} -> {k2}"
            out2 = fn2(out1)
            cache_2step[pipe_key] = out2
            m = eval_metrics(im_raw, out2, im_gt)
            results_2step.append({
                "pipeline": pipe_key,
                "name": f"{OPERATORS[k1][0]} -> {name2}",
                "steps": 2,
                **m
            })
    results_2step.sort(key=lambda x: x["score"], reverse=True)
    print(f"  Вычислено 144 комбинации за {time.time()-t0:.2f} с.")

    # ШАГ 3: 3-шаговые пайплайны (144 x 12 = 1728)
    t0 = time.time()
    results_3step = []
    print("\n[3/3] Вычисление 3-шаговых комбинаций (144 x 12 = 1728)...")
    count = 0
    for pipe2_key, out2 in cache_2step.items():
        for k3, (name3, fn3) in OPERATORS.items():
            pipe_key = f"{pipe2_key} -> {k3}"
            out3 = fn3(out2)
            m = eval_metrics(im_raw, out3, im_gt)
            results_3step.append({
                "pipeline": pipe_key,
                "steps": 3,
                **m
            })
            count += 1
            if count % 300 == 0:
                print(f"    Прогресс: {count} / 1728 ({time.time()-t0:.1f} с)...")
    results_3step.sort(key=lambda x: x["score"], reverse=True)
    print(f"  Вычислено 1728 комбинаций за {time.time()-t0:.2f} с.")

    # Сохранение полных результатов в JSON
    full_output = {
        "raw": m_raw,
        "results_1step": results_1step,
        "results_2step": results_2step,
        "results_3step": results_3step,
    }
    with open("pipeline_search_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print("\n[ГОТОВО] Все результаты сохранены в pipeline_search_results.json")

    # =====================================================================
    # ВАЛИДАЦИЯ ТОП-КОНВЕЙЕРОВ НА РЕАЛЬНОМ ЛЕСЕ (data_cache/raw_forest.npz)
    # =====================================================================
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ 2: ВАЛИДАЦИЯ ТОП-КОНВЕЙЕРОВ НА РЕАЛЬНОМ ЛЕСЕ РФ")
    print("=" * 80)

    raw_forest = np.load(r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_forest.npz")["vv"]
    m_raw_forest = eval_metrics(raw_forest, raw_forest, None)
    print(f"Исходный лес: ENL={m_raw_forest['enl']}, EPI={m_raw_forest['epi']}")

    # Выберем топ-5 из 1-шаговых, топ-10 из 2-шаговых и топ-10 из 3-шаговых
    selected_pipelines = []
    for item in results_1step[:5]:
        selected_pipelines.append((item["pipeline"], 1, item["score"]))
    for item in results_2step[:10]:
        selected_pipelines.append((item["pipeline"], 2, item["score"]))
    for item in results_3step[:10]:
        selected_pipelines.append((item["pipeline"], 3, item["score"]))

    forest_eval_results = []
    for pipe_str, steps, bench_score in selected_pipelines:
        ops = pipe_str.split(" -> ")
        curr = raw_forest
        for op in ops:
            curr = OPERATORS[op][1](curr)
        m_f = eval_metrics(raw_forest, curr, None)
        forest_eval_results.append({
            "pipeline": pipe_str,
            "steps": steps,
            "bench_score": bench_score,
            **m_f
        })

    with open("forest_top_pipelines.json", "w", encoding="utf-8") as f:
        json.dump(forest_eval_results, f, ensure_ascii=False, indent=2)
    print("[ГОТОВО] Валидация на лесе сохранена в forest_top_pipelines.json")

if __name__ == "__main__":
    main()
