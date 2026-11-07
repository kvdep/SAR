# -*- coding: utf-8 -*-
"""Модуль централизованной конфигурации бенчмарка SAR и HTML-генератора.
Определяет пути к данным, параметры видеопамяти GPU, качество сжатия WebP,
а также реестр исследуемых полигонов (сцен) и временных срезов (дат/сезонов).

Масштабирование:
1. Добавление новой локации: вызовите register_scene(...) или добавьте запись в SCENES_CONFIG.
2. Добавление произвольных дат: укажите словарь дат в параметре dates/seasons.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. СИСТЕМНЫЕ ПУТИ И ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# -----------------------------------------------------------------------------

# Базовая директория текущего модуля: .../SAR/HTML Визуализация/код
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_VIS_DIR = os.path.dirname(MODULE_DIR)
# Корневая директория репозитория SAR: .../SAR
DEFAULT_ROOT_DIR = os.path.dirname(HTML_VIS_DIR)

ROOT_DIR = os.environ.get("SAR_ROOT_DIR", DEFAULT_ROOT_DIR)
DATA_CACHE_DIR = os.environ.get("SAR_DATA_CACHE", os.path.join(ROOT_DIR, "data_cache"))
WEIGHTS_DIR = os.environ.get("SAR_WEIGHTS_DIR", os.path.join(HTML_VIS_DIR, "веса"))
if not os.path.exists(WEIGHTS_DIR) and os.path.exists(os.path.join(ROOT_DIR, "weights")):
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
PAPER_DIR = os.environ.get("SAR_PAPER_DIR", os.path.join(ROOT_DIR, "paper_results"))
OUTPUT_HTML_DIR = os.environ.get("SAR_OUTPUT_DIR", os.path.join(HTML_VIS_DIR, "данные"))

# Файлы вывода готовых HTML-страниц
OUTPUT_HTML_PATHS = [
    os.path.join(OUTPUT_HTML_DIR, "sar_advanced_comparison.html"),
    os.path.join(OUTPUT_HTML_DIR, "sar_visualization.html"),
    os.path.join(OUTPUT_HTML_DIR, "index.html"),
    # Зеркало в корне репозитория для локального открытия
    os.path.join(ROOT_DIR, "index.html"),
]

# Вычислительные параметры
VRAM_FRACTION = float(os.environ.get("SAR_VRAM_FRACTION", "0.20"))  # Ограничение VRAM для RTX 3080 (<= 2048 МБ)
WEBP_QUALITY = int(os.environ.get("SAR_WEBP_QUALITY", "70"))        # Качество WebP сжатия растров (0-100)
BENCHMARK_RUNS = int(os.environ.get("SAR_BENCHMARK_RUNS", "3"))     # Число проходов для замера time_ms
DEFAULT_BASEMAP = os.environ.get("SAR_DEFAULT_BASEMAP", "esri")     # Оптическая подложка по умолчанию

# -----------------------------------------------------------------------------
# 2. РЕЕСТР ИССЛЕДУЕМЫХ ПОЛИГОНОВ (СЦЕН) И ВРЕМЕННЫХ СРЕЗОВ
# -----------------------------------------------------------------------------

class SceneSpec:
    """Спецификация исследуемой географической локации (полигона)."""
    def __init__(
        self,
        scene_id: str,
        title: str,
        desc: str,
        center: List[float],
        zoom: int,
        bbox: List[float],
        npz_file: Optional[str] = None,
        nc_file: Optional[str] = None,
        seasons_file: Optional[str] = None,
        custom_dates: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.scene_id = scene_id
        self.title = title
        self.desc = desc
        self.center = center
        self.zoom = zoom
        self.bbox = bbox
        self.npz_file = npz_file or os.path.join(DATA_CACHE_DIR, f"raw_{scene_id}.npz")
        self.nc_file = nc_file or os.path.join(ROOT_DIR, f"s1_{scene_id}.nc")
        self.seasons_file = seasons_file or os.path.join(DATA_CACHE_DIR, f"seasons_{scene_id}.npz")
        self.custom_dates = custom_dates or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "desc": self.desc,
            "center": self.center,
            "zoom": self.zoom,
            "bbox": self.bbox,
            "npz_file": self.npz_file,
            "nc_file": self.nc_file,
            "seasons_file": self.seasons_file,
            "custom_dates": self.custom_dates,
        }


# Глобальный словарь зарегистрированных полигонов
SCENES_REGISTRY: Dict[str, SceneSpec] = {}

def register_scene(
    scene_id: str,
    title: str,
    desc: str,
    center: List[float],
    zoom: int,
    bbox: List[float],
    npz_file: Optional[str] = None,
    nc_file: Optional[str] = None,
    seasons_file: Optional[str] = None,
    custom_dates: Optional[Dict[str, Dict[str, str]]] = None,
) -> SceneSpec:
    """Регистрирует новый полигон для включения в генерацию HTML.
    
    Пример вызова:
        register_scene(
            scene_id="kamchatka",
            title="🇷🇺 9. Вулкан Ключевская Сопка, Камчатка",
            desc="Радиолокационный мониторинг вулканического рельефа.",
            center=[56.05, 160.64],
            zoom=12,
            bbox=[160.50, 55.95, 160.80, 56.15],
            npz_file=r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache\raw_kamchatka.npz"
        )
    """
    spec = SceneSpec(
        scene_id=scene_id,
        title=title,
        desc=desc,
        center=center,
        zoom=zoom,
        bbox=bbox,
        npz_file=npz_file,
        nc_file=nc_file,
        seasons_file=seasons_file,
        custom_dates=custom_dates,
    )
    SCENES_REGISTRY[scene_id] = spec
    return spec


# -----------------------------------------------------------------------------
# 3. ИНИЦИАЛИЗАЦИЯ 8 БАЗОВЫХ ПОЛИГОНОВ
# -----------------------------------------------------------------------------

register_scene(
    scene_id="benevento",
    title="🇮🇹 1. Беневенто, Италия (Vitale et al. 2021, S1-lat 41.13, lon 14.75)",
    desc="Реальный полигон статьи Vitale et al. (IEEE JSTARS 2021) и GitHub-репозитория CNNSpeckleFilter. Реальные данные Sentinel-1 IW GRD из Google Earth Engine (8 временных проходов). Холмистый рельеф региона Кампания, оливковые рощи и историческая застройка.",
    center=[41.15, 14.75],
    zoom=13,
    bbox=[14.70, 41.10, 14.80, 41.20],
)

register_scene(
    scene_id="rome",
    title="🇮🇹 2. Рим, Италия (Vitale et al. 2021, Рис. 1: Тибр и Ватикан)",
    desc="Крупный полигон городской агломерации Рима из статьи Vitale et al. Включает излучину реки Тибр, плотную застройку исторического центра и Ватикана. Высокая плотность ярких точечных отражателей и уголковых структур.",
    center=[41.90, 12.45],
    zoom=12,
    bbox=[12.40, 41.85, 12.50, 41.95],
)

register_scene(
    scene_id="serre_poncon",
    title="🇫🇷 3. Сер-Понсон, Франция (Dalsasso et al. 2021, Рис. 9)",
    desc="Реальный полигон Альпийского водохранилища Сер-Понсон из статьи MERLIN (Dalsasso et al., IEEE TGRS 2021). Сложная граница раздела вода-суша, крутые горные склоны и радиолокационные тени (SAR foreshortening/layover).",
    center=[44.52, 6.32],
    zoom=13,
    bbox=[6.25, 44.47, 6.38, 44.57],
)

register_scene(
    scene_id="forest",
    title="🇷🇺 4. Бореальный лесной массив, Тверская область (РФ, 56.5°N, 35.8°E)",
    desc="Плотный массив смешанного и хвойного леса Восточно-Европейской равнины. Наличие развитого объема рассеяния полога леса (Volume Scattering), высокая кросс-поляризация VH и выраженный спекл-паттерн.",
    center=[56.50, 35.80],
    zoom=13,
    bbox=[35.75, 56.46, 35.85, 56.54],
)

register_scene(
    scene_id="city",
    title="🇷🇺 5. Москва, СЗАО (Строгино, Серебряный Бор, МКАД)",
    desc="Плотный урбанизированный ландшафт: высокоэтажные жилые кварталы, водная гладь Строгинского затона, излучина Москвы-реки, лесопарк Серебряный Бор и крупные транспортные магистрали. Сильный эффект двойного переотражения (Double-Bounce).",
    center=[55.80, 37.44],
    zoom=13,
    bbox=[37.38, 37.76, 37.50, 37.84],
)

register_scene(
    scene_id="water",
    title="🌊 6. Акватория Черного моря (Район Севастополя, Крым)",
    desc="Полигон водной поверхности: низкое обратное рассеяние водной глади (зеркальное отражение волн от радара) и высокая контрастность морских судов и береговой линии.",
    center=[44.60, 33.48],
    zoom=13,
    bbox=[33.40, 44.55, 33.55, 44.65],
)

register_scene(
    scene_id="desert",
    title="🏜️ 7. Полупустынный массив, Астраханская область",
    desc="Засушливый ландшафт с низкой диэлектрической проницаемостью сухого песчано-глинистого грунта. Низкий контраст текстуры и преобладание поверхностного брэгговского рассеяния.",
    center=[46.80, 47.80],
    zoom=13,
    bbox=[47.75, 46.75, 47.85, 46.85],
)

register_scene(
    scene_id="mountain",
    title="⛰️ 8. Горный массив Кавказа (Район Эльбруса, 43.3°N, 42.4°E)",
    desc="Сложный высокогорный рельеф: ледники, фирновые поля, скальные обнажения, сильный геометрический рельефный сдвиг фазы, радиолокационные тени и искажения крутых склонов.",
    center=[43.35, 42.45],
    zoom=13,
    bbox=[42.40, 43.30, 42.50, 43.40],
)

# -----------------------------------------------------------------------------
# 4. ФУНКЦИИ ЗАГРУЗКИ РАСТРОВ
# -----------------------------------------------------------------------------

def load_scene_rasters(
    scene_spec: SceneSpec,
    temporal_key: str = "spring"
) -> Tuple[np.ndarray, np.ndarray, List[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Загружает двухполяризационные растры (VV, VH) для указанной сцены и временного среза.
    
    Возвращает:
        (vv, vh, bbox, vv_gt, vh_gt)
    """
    # 1. Загрузка базового NPZ
    if os.path.exists(scene_spec.npz_file):
        npz_data = np.load(scene_spec.npz_file)
        vv_raw = npz_data["vv"].astype(np.float32)
        vh_raw = npz_data["vh"].astype(np.float32)
        bbox = list(npz_data["bbox"])
        vv_gt = npz_data["vv_gt"] if ("vv_gt" in npz_data and npz_data["vv_gt"] is not None and getattr(npz_data["vv_gt"], "ndim", 0) > 0) else None
        vh_gt = npz_data["vh_gt"] if ("vh_gt" in npz_data and npz_data["vh_gt"] is not None and getattr(npz_data["vh_gt"], "ndim", 0) > 0) else None
    else:
        raise FileNotFoundError(f"Файл полигона не найден: {scene_spec.npz_file}")

    # 2. Если запрошен временной срез, отличный от базового, ищем в файле сезонов или custom_dates
    if temporal_key != "spring" and os.path.exists(scene_spec.seasons_file):
        s_data = np.load(scene_spec.seasons_file)
        k_vv = f"{temporal_key}_vv"
        k_vh = f"{temporal_key}_vh"
        if k_vv in s_data and k_vh in s_data:
            s_vv = s_data[k_vv].astype(np.float32)
            s_vh = s_data[k_vh].astype(np.float32)
            if s_vv.shape != vv_raw.shape:
                s_vv = np.array(Image.fromarray(s_vv).resize((vv_raw.shape[1], vv_raw.shape[0]), Image.Resampling.BILINEAR))
                s_vh = np.array(Image.fromarray(s_vh).resize((vv_raw.shape[1], vv_raw.shape[0]), Image.Resampling.BILINEAR))
            return s_vv, s_vh, bbox, (vv_gt if temporal_key == "spring" else None), (vh_gt if temporal_key == "spring" else None)

    # 3. Проверка пользовательских произвольных дат (custom_dates)
    if temporal_key in scene_spec.custom_dates:
        date_info = scene_spec.custom_dates[temporal_key]
        if "npz" in date_info and os.path.exists(date_info["npz"]):
            c_data = np.load(date_info["npz"])
            c_vv = c_data["vv"].astype(np.float32)
            c_vh = c_data["vh"].astype(np.float32)
            if c_vv.shape != vv_raw.shape:
                c_vv = np.array(Image.fromarray(c_vv).resize((vv_raw.shape[1], vv_raw.shape[0]), Image.Resampling.BILINEAR))
                c_vh = np.array(Image.fromarray(c_vh).resize((vv_raw.shape[1], vv_raw.shape[0]), Image.Resampling.BILINEAR))
            return c_vv, c_vh, bbox, None, None

    return vv_raw, vh_raw, bbox, vv_gt, vh_gt


def get_available_temporal_keys(scene_spec: SceneSpec) -> List[str]:
    """Возвращает список всех доступных дат/сезонов для полигона."""
    keys = ["spring"]
    if os.path.exists(scene_spec.seasons_file):
        s_data = np.load(scene_spec.seasons_file)
        for s in ["summer", "autumn", "winter"]:
            if f"{s}_vv" in s_data:
                keys.append(s)
    for k in scene_spec.custom_dates.keys():
        if k not in keys:
            keys.append(k)
    return keys
