# -*- coding: utf-8 -*-
"""Скрипт генерации данных и интерактивной визуализации Sentinel-1 SAR.

Загружает 5 характерных квадрантов (лес, водоем, город, пустыня, гора),
применяет RTC и Gamma MAP, генерирует слои VV, VH, Ratio и RGB Composite,
и формирует автономный HTML-интерфейс на базе Leaflet.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

import matplotlib.cm as cm
import numpy as np
from PIL import Image

import run

QUADRANTS = {
    "forest": {
        "title": "Лес (Мещёрский национальный парк)",
        "desc": "Преобладает объёмное рассеяние (volume scattering) кронами деревьев. Высокий уровень VH (от -15 до -11 dB), умеренный VV, низкий ratio.",
        "bbox": [40.10, 55.45, 40.20, 55.55],
        "center": [55.50, 40.15],
        "zoom": 13,
        "date_range": ["2024-05-01", "2024-05-31"],
    },
    "water": {
        "title": "Водоём (Чёрное море)",
        "desc": "Зеркальное отражение (specular reflection) от гладкой поверхности воды в направлении от сенсора. Экстремально низкие значения VV (до -25 dB) и VH (до -35 dB), высокий ratio.",
        "bbox": [39.50, 43.40, 39.60, 43.50],
        "center": [43.45, 39.55],
        "zoom": 13,
        "date_range": ["2024-05-01", "2024-05-31"],
    },
    "city": {
        "title": "Город (Москва, урбанизированная застройка)",
        "desc": "Двойной отскок (double-bounce) от прямых углов 'стена-земля' и металлоконструкций. Максимальные значения VV (до +2 dB) и высокий уровень деполяризации.",
        "bbox": [37.55, 55.70, 37.65, 55.80],
        "center": [55.75, 37.60],
        "zoom": 13,
        "date_range": ["2024-05-01", "2024-05-31"],
    },
    "desert": {
        "title": "Пустыня (Сахара, Эрг-Шебби, Марокко)",
        "desc": "Поверхностное рассеяние (surface scattering) песчаных дюн и проникновение C-диапазона в сухой песок. Низкая шероховатость, слабый уровень кросс-поляризации VH.",
        "bbox": [-4.05, 31.10, -3.95, 31.20],
        "center": [31.15, -4.00],
        "zoom": 13,
        "date_range": ["2024-05-01", "2024-05-31"],
    },
    "mountain": {
        "title": "Гора (Эльбрус, Кавказ)",
        "desc": "Экстремальный макрорельеф: радарные эффекты укорочения склонов (foreshortening), наложения (layover) на освещенных склонах и радиотени на противоположных.",
        "bbox": [42.40, 43.30, 42.50, 43.40],
        "center": [43.35, 42.45],
        "zoom": 13,
        "date_range": ["2024-05-01", "2024-05-31"],
    },
}

KEY_PATH = r"D:\ПРОФ\ВУЗ\ВТСК\ОМЕЛА GIT\src\parsers\dem_model\key.json"


def array_to_png_base64(arr: np.ndarray, vmin: float, vmax: float, cmap_name: str | None = None) -> str:
    """Нормализует 2D массив в 8-битное изображение и кодирует в Base64 Data URL."""
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


def rgb_composite_base64(vv: np.ndarray, vh: np.ndarray, ratio: np.ndarray) -> str:
    """Генерирует композит: R - VV, G - ratio (VV - VH), B - VH."""
    # Точное соответствие Colab-ноутбукам:
    # R = VV (поверхностное рассеяние), G = VH (растительность/объемное), B = Ratio (структура)
    r = np.clip((vv - (-18.0)) / (-2.0 - (-18.0)), 0.0, 1.0)
    g = np.clip((vh - (-24.0)) / (-8.0 - (-24.0)), 0.0, 1.0)
    b = np.clip((ratio - (2.0)) / (14.0 - (2.0)), 0.0, 1.0)

    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)

    rgb = np.stack([r, g, b], axis=-1)
    rgb_8u = (rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb_8u, mode="RGB").convert("RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def process_all_quadrants() -> dict[str, Any]:
    """Выгружает данные через run.py и формирует пакет слоев для каждого квадранта."""
    results = {}

    for key, q in QUADRANTS.items():
        print(f"[INFO] Обработка квадранта: {q['title']}...")
        bbox = q["bbox"]
        geojson = {
            "type": "Feature",
            "properties": {
                "time": q["date_range"],
                "is_rectangle": True,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            },
        }

        nc_path = os.path.join(r"D:\ПРОФ\ВУЗ\ВТСК\SAR", f"s1_{key}.nc")
        if os.path.exists(nc_path):
            print(f"[INFO] Файл {nc_path} найден, загрузка из локального файла...")
            import shutil
            import tempfile
            tmp = os.path.join(
                tempfile.gettempdir(), f"load_{key}_{os.urandom(4).hex()}.nc"
            )
            shutil.copy2(nc_path, tmp)
            import xarray as xr
            with xr.open_dataset(tmp) as raw_ds:
                ds = raw_ds.load()
            if os.path.exists(tmp):
                os.remove(tmp)
        else:
            ds = run.run(
                geojson=geojson,
                output_path=nc_path,
                ee_key_path=KEY_PATH,
                ee_account="omela-earth-engine@vladerm-project-462821.iam.gserviceaccount.com",
            )

        # Берем первый доступный временной срез
        vv_2d = ds["VV"].values[0]
        vh_2d = ds["VH"].values[0]
        ratio_2d = ds["ratio"].values[0]

        # Расчет базовой статистики для информационного блока
        stats = {
            "vv_min": round(float(np.nanmin(vv_2d)), 2),
            "vv_max": round(float(np.nanmax(vv_2d)), 2),
            "vv_mean": round(float(np.nanmean(vv_2d)), 2),
            "vh_min": round(float(np.nanmin(vh_2d)), 2),
            "vh_max": round(float(np.nanmax(vh_2d)), 2),
            "vh_mean": round(float(np.nanmean(vh_2d)), 2),
            "ratio_min": round(float(np.nanmin(ratio_2d)), 2),
            "ratio_max": round(float(np.nanmax(ratio_2d)), 2),
            "ratio_mean": round(float(np.nanmean(ratio_2d)), 2),
            "shape": list(vv_2d.shape),
            "dates": [str(t)[:10] for t in ds["time"].values],
        }

        # Кодирование растровых наложений в Base64
        overlay_vv = array_to_png_base64(vv_2d, vmin=-25.0, vmax=0.0)
        overlay_vh = array_to_png_base64(vh_2d, vmin=-32.0, vmax=-8.0)
        overlay_ratio = array_to_png_base64(ratio_2d, vmin=0.0, vmax=20.0, cmap_name="viridis")
        overlay_rgb = rgb_composite_base64(vv_2d, vh_2d, ratio_2d)

        # Leaflet bounds: [[lat_min, lon_min], [lat_max, lon_max]]
        leaflet_bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]

        results[key] = {
            "title": q["title"],
            "desc": q["desc"],
            "center": q["center"],
            "zoom": q["zoom"],
            "bounds": leaflet_bounds,
            "stats": stats,
            "layers": {
                "vv": overlay_vv,
                "vh": overlay_vh,
                "ratio": overlay_ratio,
                "rgb": overlay_rgb,
            },
        }

    return results


def build_html(data: dict[str, Any], output_html_path: str) -> None:
    """Собирает автономное интерактивное веб-приложение визуализации."""
    json_data = json.dumps(data)

    html_template = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Интерактивная визуализация Sentinel-1 SAR</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body, html {{
            height: 100%;
            width: 100%;
            overflow: hidden;
            display: flex;
        }}
        #sidebar {{
            width: 380px;
            height: 100%;
            background: #1e1e24;
            color: #f0f0f2;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 12px rgba(0,0,0,0.4);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        #map-container {{
            flex: 1;
            height: 100%;
            position: relative;
        }}
        #map {{
            height: 100%;
            width: 100%;
        }}
        h1 {{
            font-size: 1.25rem;
            color: #ffffff;
            border-bottom: 2px solid #3d5afe;
            padding-bottom: 8px;
        }}
        .section {{
            background: #2b2b36;
            border-radius: 8px;
            padding: 14px;
        }}
        .section-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9fa8da;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        select, input[type="range"] {{
            width: 100%;
        }}
        select {{
            background: #181820;
            color: #ffffff;
            border: 1px solid #48485e;
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 0.95rem;
            cursor: pointer;
            outline: none;
        }}
        .control-row {{
            margin-bottom: 12px;
        }}
        .control-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-size: 0.9rem;
        }}
        .control-header label {{
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-weight: 500;
        }}
        .opacity-val {{
            font-size: 0.8rem;
            color: #9fa8da;
            font-variant-numeric: tabular-nums;
        }}
        input[type="range"] {{
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: #48485e;
            outline: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #3d5afe;
            cursor: pointer;
            border: 2px solid #ffffff;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            margin-top: 8px;
        }}
        .stats-table th, .stats-table td {{
            padding: 4px 6px;
            text-align: left;
            border-bottom: 1px solid #3a3a4c;
        }}
        .stats-table th {{
            color: #9fa8da;
        }}
        .desc-box {{
            font-size: 0.85rem;
            line-height: 1.45;
            color: #c5cae9;
            margin-top: 6px;
        }}
        .legend {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            margin-top: 4px;
        }}
        .legend-tag {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
        }}
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Sentinel-1 SAR Анализатор</h1>

        <div class="section">
            <div class="section-title">Выбор квадранта ландшафта</div>
            <select id="quadrant-select">
                <option value="forest">1. Лес (Мещёрский парк)</option>
                <option value="water">2. Водоём (Чёрное море)</option>
                <option value="city">3. Город (Москва, центр)</option>
                <option value="desert">4. Пустыня (Сахара, дюны)</option>
                <option value="mountain">5. Гора (Эльбрус, Кавказ)</option>
            </select>
            <div id="quadrant-desc" class="desc-box"></div>
        </div>

        <div class="section">
            <div class="section-title">Слои и прозрачность</div>

            <div class="control-row">
                <div class="control-header">
                    <label><input type="checkbox" id="check-sat" checked> Спутниковое фото (Optical)</label>
                    <span id="val-sat" class="opacity-val">100%</span>
                </div>
                <input type="range" id="opacity-sat" min="0" max="1" step="0.01" value="1.0">
            </div>

            <div class="control-row">
                <div class="control-header">
                    <label><input type="checkbox" id="check-rgb" checked> RGB Композит (R:VV, G:VH, B:Ratio)</label>
                    <span id="val-rgb" class="opacity-val">80%</span>
                </div>
                <input type="range" id="opacity-rgb" min="0" max="1" step="0.01" value="0.8">
            </div>

            <div class="control-row">
                <div class="control-header">
                    <label><input type="checkbox" id="check-vv"> Радар VV (вертикальная пол.)</label>
                    <span id="val-vv" class="opacity-val">0%</span>
                </div>
                <input type="range" id="opacity-vv" min="0" max="1" step="0.01" value="0.0">
            </div>

            <div class="control-row">
                <div class="control-header">
                    <label><input type="checkbox" id="check-vh"> Радар VH (кросс-поляризация)</label>
                    <span id="val-vh" class="opacity-val">0%</span>
                </div>
                <input type="range" id="opacity-vh" min="0" max="1" step="0.01" value="0.0">
            </div>

            <div class="control-row">
                <div class="control-header">
                    <label><input type="checkbox" id="check-ratio"> Ratio (VV - VH, viridis)</label>
                    <span id="val-ratio" class="opacity-val">0%</span>
                </div>
                <input type="range" id="opacity-ratio" min="0" max="1" step="0.01" value="0.0">
            </div>
        </div>

        <div class="section">
            <div class="section-title">Физические параметры (dB)</div>
            <table class="stats-table">
                <thead>
                    <tr><th>Параметр</th><th>Мин</th><th>Макс</th><th>Среднее</th></tr>
                </thead>
                <tbody id="stats-body"></tbody>
            </table>
            <div id="grid-info" style="font-size: 0.8rem; color: #9fa8da; margin-top: 8px;"></div>
        </div>
    </div>

    <div id="map-container">
        <div id="map"></div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const data = {json_data};

        const map = L.map('map', {{
            center: data.forest.center,
            zoom: data.forest.zoom,
            zoomControl: true
        }});

        // Спутниковая подложка Esri World Imagery
        const satLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 18,
            attribution: 'Tiles &copy; Esri'
        }}).addTo(map);

        let currentOverlays = {{
            vv: null,
            vh: null,
            ratio: null,
            rgb: null
        }};

        function updateQuadrant(key) {{
            const q = data[key];
            map.flyToBounds(q.bounds, {{ padding: [30, 30], duration: 1.2 }});

            document.getElementById('quadrant-desc').innerText = q.desc;

            // Удаление старых слоев
            Object.values(currentOverlays).forEach(layer => {{
                if (layer) map.removeLayer(layer);
            }});

            // Создание новых наложений
            currentOverlays.rgb = L.imageOverlay(q.layers.rgb, q.bounds).addTo(map);
            currentOverlays.vv = L.imageOverlay(q.layers.vv, q.bounds).addTo(map);
            currentOverlays.vh = L.imageOverlay(q.layers.vh, q.bounds).addTo(map);
            currentOverlays.ratio = L.imageOverlay(q.layers.ratio, q.bounds).addTo(map);

            // Применение текущих настроек видимости и прозрачности
            syncLayerState('rgb');
            syncLayerState('vv');
            syncLayerState('vh');
            syncLayerState('ratio');

            // Обновление таблицы статистики
            const s = q.stats;
            document.getElementById('stats-body').innerHTML = `
                <tr><td><strong>VV</strong></td><td>${{s.vv_min}}</td><td>${{s.vv_max}}</td><td>${{s.vv_mean}}</td></tr>
                <tr><td><strong>VH</strong></td><td>${{s.vh_min}}</td><td>${{s.vh_max}}</td><td>${{s.vh_mean}}</td></tr>
                <tr><td><strong>Ratio (VV-VH)</strong></td><td>${{s.ratio_min}}</td><td>${{s.ratio_max}}</td><td>${{s.ratio_mean}}</td></tr>
            `;
            document.getElementById('grid-info').innerText = `Разрешение: 20 м | Сетка: ${{s.shape[0]}} x ${{s.shape[1]}} px | Дата: ${{s.dates.join(', ')}}`;
        }}

        function syncLayerState(name) {{
            const check = document.getElementById('check-' + name);
            const slider = document.getElementById('opacity-' + name);
            const label = document.getElementById('val-' + name);

            const opacity = check.checked ? parseFloat(slider.value) : 0.0;
            label.innerText = Math.round(opacity * 100) + '%';

            if (currentOverlays[name]) {{
                currentOverlays[name].setOpacity(opacity);
            }}
        }}

        // Обработчики контролов
        ['rgb', 'vv', 'vh', 'ratio'].forEach(name => {{
            const check = document.getElementById('check-' + name);
            const slider = document.getElementById('opacity-' + name);

            check.addEventListener('change', () => syncLayerState(name));
            slider.addEventListener('input', () => {{
                if (!check.checked && parseFloat(slider.value) > 0) check.checked = true;
                syncLayerState(name);
            }});
        }});

        // Контрол спутникового фото
        const checkSat = document.getElementById('check-sat');
        const sliderSat = document.getElementById('opacity-sat');
        const labelSat = document.getElementById('val-sat');

        function syncSat() {{
            const opacity = checkSat.checked ? parseFloat(sliderSat.value) : 0.0;
            labelSat.innerText = Math.round(opacity * 100) + '%';
            satLayer.setOpacity(opacity);
        }}
        checkSat.addEventListener('change', syncSat);
        sliderSat.addEventListener('input', () => {{
            if (!checkSat.checked && parseFloat(sliderSat.value) > 0) checkSat.checked = true;
            syncSat();
        }});

        document.getElementById('quadrant-select').addEventListener('change', (e) => {{
            updateQuadrant(e.target.value);
        }});

        // Инициализация первого квадранта
        updateQuadrant('forest');
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"[SUCCESS] Интерактивная визуализация сохранена в: {output_html_path}")


if __name__ == "__main__":
    data_bundle = process_all_quadrants()
    out_html = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\sar_visualization.html"
    build_html(data_bundle, out_html)
