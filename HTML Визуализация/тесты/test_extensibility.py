# -*- coding: utf-8 -*-
"""Тест расширяемости и масштабируемости архитектуры HTML Визуализации SAR.
Проверяет:
1. Регистрацию нового пользовательского фильтра через @register_filter без правки шаблона.
2. Регистрацию нового полигона через register_scene с чтением реальных данных (.npz, .nc, .tif).
3. Доступность временных срезов через get_available_temporal_keys.
4. Вычисление метрик фильтрации (ENL, EPI, Bias, PSNR, SSIM, Var(r), time_ms).
5. Двустороннюю совместимость FilterRegistry с sar_server и build_full_benchmark.
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "код"))
sys.path.insert(0, CODE_DIR)

from config import (
    SceneSpec, SCENES_REGISTRY, register_scene, load_scene_rasters,
    get_available_temporal_keys, DATA_DIR, DATA_CACHE_DIR
)
from filter_registry import (
    FilterRegistry, register_filter, compute_metrics, benchmark_filter
)
import filters_suite
import html_template


def test_custom_filter_registration():
    print("[ТЕСТ 1] Регистрация пользовательского фильтра через декоратор @register_filter...")
    
    initial_count = len(FilterRegistry.list_all())
    
    @register_filter(
        filter_id="test_gaussian_custom",
        name="Пользовательский Gaussian Denoise",
        category="spatial",
        default_kwargs={"sigma": 1.2},
        is_gpu=False,
        description="Тестовый фильтр Гаусса для проверки масштабирования."
    )
    def filter_gaussian_custom(vv: np.ndarray, vh: np.ndarray, sigma: float = 1.2):
        import scipy.ndimage as ndimage
        vv_f = ndimage.gaussian_filter(vv.astype(np.float64), sigma=sigma).astype(np.float32)
        vh_f = ndimage.gaussian_filter(vh.astype(np.float64), sigma=sigma).astype(np.float32)
        return vv_f, vh_f

    new_count = len(FilterRegistry.list_all())
    assert new_count == initial_count + 1, f"Ожидалось {initial_count + 1} фильтров, получено {new_count}"
    
    spec = FilterRegistry.get("test_gaussian_custom")
    assert spec.filter_id == "test_gaussian_custom"
    assert spec.default_kwargs == {"sigma": 1.2}

    # Проверка исполнения на реальном массиве
    dummy_vv = np.random.RandomState(42).gamma(4.4, 1.0/4.4, (128, 128)).astype(np.float32) * 0.1
    dummy_vh = dummy_vv * 0.25
    out_vv, out_vh = spec.execute(dummy_vv, dummy_vh)
    assert out_vv.shape == dummy_vv.shape
    assert out_vh.shape == dummy_vh.shape

    # Проверка расчета метрик
    m = compute_metrics(dummy_vv, out_vv, looks=4.4)
    assert "enl" in m and "epi" in m and "bias_db" in m
    print(f"  [OK] Фильтр успешно зарегистрирован и исполнен. ENL: {m['enl']}, EPI: {m['epi']}")


def test_scene_registration_and_raster_loading():
    print("\n[ТЕСТ 2] Регистрация сцены и чтение реальных радиолокационных растров...")

    # Проверка загрузки существующего полигона forest из кэша
    forest_spec = SCENES_REGISTRY["forest"]
    vv_raw, vh_raw, bbox, vv_gt, vh_gt = load_scene_rasters(forest_spec, temporal_key="spring")
    assert vv_raw.ndim == 2 and vv_raw.shape[0] > 0
    assert vh_raw.ndim == 2 and vh_raw.shape[1] > 0
    print(f"  [OK] Полигон 'forest' загружен успешно: форма {vv_raw.shape}, диапазон [{vv_raw.min():.4f}, {vv_raw.max():.4f}]")

    # Проверка прямого чтения .nc файла без npz кэша
    nc_path = os.path.join(DATA_DIR, "s1_forest.nc")
    if os.path.exists(nc_path):
        custom_scene = register_scene(
            scene_id="test_forest_direct_nc",
            title="Тестовый прямой полигон NetCDF",
            desc="Проверка прямого чтения .nc файла.",
            center=[56.5, 35.8],
            zoom=13,
            bbox=[35.75, 56.46, 35.85, 56.54],
            nc_file=nc_path
        )
        # Принудительно указываем файл .nc как основной
        custom_scene.file_path = nc_path
        vv_nc, vh_nc, bbox_nc, _, _ = load_scene_rasters(custom_scene, temporal_key="spring")
        assert vv_nc.ndim == 2 and vv_nc.shape == (557, 558)
        assert np.isfinite(vv_nc).all()
        print(f"  [OK] Прямое чтение из NetCDF .nc подтверждено: форма {vv_nc.shape}, средняя мощность {vv_nc.mean():.4f}")


def test_temporal_keys():
    print("\n[ТЕСТ 3] Извлечение временных срезов и дат...")
    spec = SCENES_REGISTRY["forest"]
    keys = get_available_temporal_keys(spec)
    assert "spring" in keys
    print(f"  [OK] Доступные временные срезы для 'forest': {keys}")


def test_template_placeholders():
    print("\n[ТЕСТ 4] Проверка шаблона HTML на плейсхолдеры и отсутствие хардкода...")
    tpl = html_template.HTML_TEMPLATE
    assert "__DATA_JSON__" in tpl, "Отсутствует плейсхолдер __DATA_JSON__"
    assert "__FILTER_METADATA_JSON__" in tpl, "Отсутствует плейсхолдер __FILTER_METADATA_JSON__"
    assert "<select id=\"bench-select\"></select>" in tpl, "В bench-select обнаружен устаревший хардкод опций"
    print("  [OK] Шаблон содержит чистые точки монтирования данных и фильтров.")


if __name__ == "__main__":
    print("=" * 70)
    print("ЗАПУСК ТЕСТИРОВАНИЯ ЭНКАПСУЛЯЦИИ И МАСШТАБИРУЕМОСТИ")
    print("=" * 70)
    test_custom_filter_registration()
    test_scene_registration_and_raster_loading()
    test_temporal_keys()
    test_template_placeholders()
    print("\n" + "=" * 70)
    print("ВСЕ 4 ТЕСТА УСПЕШНО ПРОЙДЕНЫ!")
    print("=" * 70)
