# -*- coding: utf-8 -*-
"""Модуль централизованной конфигурации бенчмарка SAR и HTML-генератора.
Все пути вычисляются динамически относительно расположения проекта в файловой системе.
Никакого хардкода абсолютных путей. Любые каталоги настраиваются в секции 1.

Масштабирование:
1. Добавление новой локации: вызовите register_scene(...) или добавьте запись в SCENES_REGISTRY.
2. Добавление произвольных дат: укажите словарь дат в параметре dates/seasons.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. СИСТЕМНЫЕ ПУТИ (динамическое определение относительно расположения файлов)
# -----------------------------------------------------------------------------

# Базовая директория текущего модуля: .../HTML Визуализация/код
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# Директория раздела визуализации: .../HTML Визуализация
HTML_VIS_DIR = os.path.dirname(MODULE_DIR)
# Корневая директория репозитория SAR (родительская к 'HTML Визуализация')
PROJECT_ROOT = os.path.abspath(os.path.join(HTML_VIS_DIR, ".."))

# Пользовательские переменные окружения или автоматические пути проекта:
ROOT_DIR = os.environ.get("SAR_ROOT_DIR", PROJECT_ROOT)

# Каталог входных данных (.nc, .tif, data_cache, paper_results)
DATA_DIR = os.environ.get("SAR_DATA_DIR", os.path.join(HTML_VIS_DIR, "данные"))
LOCAL_CACHE_DIR = os.path.join(HTML_VIS_DIR, "temp", "data_cache")
DATA_CACHE_DIR = os.environ.get("SAR_DATA_CACHE", LOCAL_CACHE_DIR if os.path.exists(LOCAL_CACHE_DIR) else os.path.join(PROJECT_ROOT, "temp", "data_cache"))
PAPER_DIR = os.environ.get("SAR_PAPER_DIR", os.path.join(DATA_DIR, "paper_results"))

# Каталог обученных весов нейросетевых моделей
WEIGHTS_DIR = os.environ.get("SAR_WEIGHTS_DIR", os.path.join(HTML_VIS_DIR, "веса"))

# Каталог результатов фильтрации и экспериментов (JSON, сводки)
RESULTS_DIR = os.environ.get("SAR_RESULTS_DIR", os.path.join(HTML_VIS_DIR, "результаты_фильтрации"))

# Каталог вывода единственного канонического HTML файла
HTML_DIR = os.environ.get("SAR_HTML_DIR", os.path.join(HTML_VIS_DIR, "html"))
OUTPUT_HTML_PATH = os.path.join(HTML_DIR, "index.html")
OUTPUT_HTML_PATHS = [OUTPUT_HTML_PATH]

# Параметры генерации растров
WEBP_QUALITY = int(os.environ.get("SAR_WEBP_QUALITY", "70"))        # Качество WebP сжатия (0-100)
BENCHMARK_RUNS = int(os.environ.get("SAR_BENCHMARK_RUNS", "3"))     # Проходы замера аппаратного времени time_ms
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
        file_path: Optional[str] = None,
        seasons_file: Optional[str] = None,
        custom_dates: Optional[Dict[str, Dict[str, str]]] = None,
        npz_file: Optional[str] = None,
        nc_file: Optional[str] = None,
    ):
        self.scene_id = scene_id
        self.title = title
        self.desc = desc
        self.center = center
        self.zoom = zoom
        self.bbox = bbox

        # Автоматическое определение основного файла данных (.npz, .nc, .tif)
        resolved_file = file_path or npz_file or nc_file
        if not resolved_file or not os.path.exists(resolved_file):
            candidates = [
                os.path.join(DATA_CACHE_DIR, f"raw_{scene_id}.npz"),
                os.path.join(DATA_DIR, f"s1_{scene_id}.nc"),
                os.path.join(DATA_DIR, f"{scene_id}.tif"),
                os.path.join(DATA_DIR, f"{scene_id}.nc"),
                os.path.join(DATA_DIR, f"sar_{scene_id}.tif"),
                os.path.join(DATA_DIR, f"sar_{scene_id}.nc"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    resolved_file = c
                    break
            if not resolved_file:
                resolved_file = candidates[0]

        self.file_path = resolved_file
        self.npz_file = resolved_file if resolved_file.endswith(".npz") else os.path.join(DATA_CACHE_DIR, f"raw_{scene_id}.npz")
        self.nc_file = resolved_file if resolved_file.endswith(".nc") else os.path.join(DATA_DIR, f"s1_{scene_id}.nc")

        # Автоматическое определение файла сезонных рядов
        def_seasons = seasons_file or os.path.join(DATA_CACHE_DIR, f"seasons_{scene_id}.npz")
        if not os.path.exists(def_seasons) and os.path.exists(os.path.join(DATA_DIR, f"seasons_{scene_id}.npz")):
            def_seasons = os.path.join(DATA_DIR, f"seasons_{scene_id}.npz")
        self.seasons_file = def_seasons

        self.custom_dates = custom_dates or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "desc": self.desc,
            "center": self.center,
            "zoom": self.zoom,
            "bbox": self.bbox,
            "file_path": self.file_path,
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
            title="9. Вулкан Ключевская Сопка, Камчатка",
            desc="Радиолокационный мониторинг вулканического рельефа.",
            center=[56.05, 160.64],
            zoom=12,
            bbox=[160.50, 55.95, 160.80, 56.15]
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
    title="1. Беневенто, Италия (Vitale et al. 2021, S1-lat 41.13, lon 14.75)",
    desc="Реальный полигон статьи Vitale et al. (IEEE JSTARS 2021) и репозитория CNNSpeckleFilter. Данные Sentinel-1 IW GRD из Google Earth Engine (8 временных проходов). Холмистый рельеф региона Кампания, оливковые рощи и историческая застройка.",
    center=[41.15, 14.75],
    zoom=13,
    bbox=[14.70, 41.10, 14.80, 41.20],
)

register_scene(
    scene_id="rome",
    title="2. Рим, Италия (Vitale et al. 2021, Рис. 1: Тибр и Ватикан)",
    desc="Полигон городской агломерации Рима из статьи Vitale et al. Включает излучину реки Тибр, плотную застройку исторического центра и Ватикана. Высокая плотность точечных отражателей и уголковых структур.",
    center=[41.90, 12.45],
    zoom=12,
    bbox=[12.40, 41.85, 12.50, 41.95],
)

register_scene(
    scene_id="serre_poncon",
    title="3. Сер-Понсон, Франция (Dalsasso et al. 2021, Рис. 9)",
    desc="Полигон Альпийского водохранилища Сер-Понсон из статьи MERLIN (Dalsasso et al., IEEE TGRS 2021). Граница раздела вода-суша, крутые горные склоны и радиолокационные тени (SAR foreshortening/layover).",
    center=[44.52, 6.32],
    zoom=13,
    bbox=[6.25, 44.47, 6.38, 44.57],
)

register_scene(
    scene_id="forest",
    title="4. Бореальный лесной массив, Тверская область (56.5°N, 35.8°E)",
    desc="Массив смешанного и хвойного леса Восточно-Европейской равнины. Наличие развитого объема рассеяния полога леса (Volume Scattering), высокая кросс-поляризация VH и выраженный спекл-паттерн.",
    center=[56.50, 35.80],
    zoom=13,
    bbox=[35.75, 56.46, 35.85, 56.54],
)

register_scene(
    scene_id="city",
    title="5. Москва, СЗАО (Строгино, Серебряный Бор, МКАД)",
    desc="Урбанизированный ландшафт: высокоэтажные жилые кварталы, водная гладь Строгинского затона, излучина Москвы-реки, лесопарк Серебряный Бор и крупные транспортные магистрали. Сильный эффект двойного переотражения (Double-Bounce).",
    center=[55.80, 37.44],
    zoom=13,
    bbox=[37.38, 37.76, 37.50, 37.84],
)

register_scene(
    scene_id="water",
    title="6. Акватория Черного моря (Район Севастополя, Крым)",
    desc="Полигон водной поверхности: низкое обратное рассеяние водной глади (зеркальное отражение волн от радара) и высокая контрастность морских судов и береговой линии.",
    center=[44.60, 33.48],
    zoom=13,
    bbox=[33.40, 44.55, 33.55, 44.65],
)

register_scene(
    scene_id="desert",
    title="7. Полупустынный массив, Астраханская область",
    desc="Засушливый ландшафт с низкой диэлектрической проницаемостью сухого песчано-глинистого грунта. Низкий контраст текстуры и преобладание поверхностного брэгговского рассеяния.",
    center=[46.80, 47.80],
    zoom=13,
    bbox=[47.75, 46.75, 47.85, 46.85],
)

register_scene(
    scene_id="mountain",
    title="8. Горный массив Кавказа (Район Эльбруса, 43.3°N, 42.4°E)",
    desc="Сложный высокогорный рельеф: ледники, фирновые поля, скальные обнажения, сильный геометрический рельефный сдвиг фазы, радиолокационные тени и искажения крутых склонов.",
    center=[43.35, 42.45],
    zoom=13,
    bbox=[42.40, 43.30, 42.50, 43.40],
)

# -----------------------------------------------------------------------------
def _read_raster_data(
    file_path: str,
    default_bbox: List[float]
) -> Tuple[np.ndarray, np.ndarray, List[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Читает двухполяризационные растры (.npz, .nc, .tif) без синтетических данных."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл радиолокационных данных не найден: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npz":
        npz = np.load(file_path)
        vv = npz["vv"].astype(np.float32)
        vh = npz["vh"].astype(np.float32)
        bbox = list(npz["bbox"]) if "bbox" in npz else default_bbox
        vv_gt = npz["vv_gt"] if ("vv_gt" in npz and npz["vv_gt"] is not None and getattr(npz["vv_gt"], "ndim", 0) > 0) else None
        vh_gt = npz["vh_gt"] if ("vh_gt" in npz and npz["vh_gt"] is not None and getattr(npz["vh_gt"], "ndim", 0) > 0) else None
        return vv, vh, bbox, vv_gt, vh_gt

    import rasterio
    with rasterio.open(file_path) as src:
        sub = getattr(src, "subdatasets", [])
        if sub and len(sub) >= 2:
            with rasterio.open(sub[0]) as s_vv:
                vv = s_vv.read(1).astype(np.float32)
            with rasterio.open(sub[1]) as s_vh:
                vh = s_vh.read(1).astype(np.float32)
        else:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32) if src.count >= 2 else vv.copy()

        if src.bounds is not None and src.crs is not None:
            b = src.bounds
            bbox = [float(b.left), float(b.bottom), float(b.right), float(b.top)]
        else:
            bbox = default_bbox

    vv = np.nan_to_num(vv, nan=1e-6, posinf=10.0, neginf=1e-6)
    vh = np.nan_to_num(vh, nan=1e-6, posinf=10.0, neginf=1e-6)

    # Физическая линеаризация при сохранении в логарифмической шкале дБ: I = 10^(dB / 10)
    if float(np.mean(vv)) < 0.0 or (float(np.min(vv)) < -1.0 and float(np.max(vv)) < 15.0):
        vv = np.maximum(10.0 ** (vv / 10.0), 1e-6).astype(np.float32)
        vh = np.maximum(10.0 ** (vh / 10.0), 1e-6).astype(np.float32)

    return vv, vh, bbox, None, None


def load_scene_rasters(
    scene_spec: SceneSpec,
    temporal_key: str = "spring"
) -> Tuple[np.ndarray, np.ndarray, List[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Загружает двухполяризационные растры (VV, VH) для указанной сцены и временного среза.
    
    Поддерживает: .npz, .nc, .tif.
    Возвращает: (vv, vh, bbox, vv_gt, vh_gt)
    """
    vv_raw, vh_raw, bbox, vv_gt, vh_gt = _read_raster_data(scene_spec.file_path, scene_spec.bbox)

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

    if temporal_key in scene_spec.custom_dates:
        date_info = scene_spec.custom_dates[temporal_key]
        c_path = date_info.get("file") or date_info.get("npz") or date_info.get("nc")
        if c_path and os.path.exists(c_path):
            c_vv, c_vh, _, _, _ = _read_raster_data(c_path, bbox)
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
