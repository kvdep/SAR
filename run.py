"""Модуль выгрузки и первичной обработки данных Sentinel-1 SAR.

Выполняет калибровку GRD, радиометрическую коррекцию рельефа (RTC),
временную медианную фильтрацию за 3 месяца, пространственный фильтр
Gamma MAP и расчет логарифмического отношения поляризаций ratio = VV - VH.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from shapely.geometry import shape

if TYPE_CHECKING:
    import ee

logger = logging.getLogger(__name__)

# Подавление системных предупреждений Windows и Airflow
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*POSIX-compliant.*"
)
warnings.filterwarnings("ignore", message=".*airflow.decorators.task.*")

CallableT = TypeVar("CallableT", bound=Callable[..., object])

try:
    from airflow import DAG
    try:
        from airflow.sdk import task
    except ImportError:
        from airflow.decorators import task
    AIRFLOW_ACTIVE = True
except ImportError:
    AIRFLOW_ACTIVE = False

    def task_virtualenv(
        *args: object, **kwargs: object
    ) -> Callable[[CallableT], CallableT]:
        def decorator(fn: CallableT) -> CallableT:
            fn.function = fn
            return fn
        return decorator

    class MockTask:
        @staticmethod
        def virtualenv(
            *args: object, **kwargs: object
        ) -> Callable[[CallableT], CallableT]:
            return task_virtualenv(*args, **kwargs)

    task = MockTask()  # type: ignore[assignment]

    class DAG:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> DAG:
            return self

        def __exit__(self, *args: object) -> None:
            pass


DAG_ID = "sentinel1_sar_parser"

DEFAULT_ARGS = {
    "owner": "Kidysiuk Aleksandr",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

FILE_TO_READ = os.path.join(tempfile.gettempdir(), "input_geo.json")
FILE_TO_WRITE = os.path.join(tempfile.gettempdir(), "output_data.nc")


def resolve_variable(key: str, env_name: str, default_value: str = "") -> str:
    """Извлекает строковую переменную из окружения или Airflow Variable.

    Значение никогда не логируется и не выводится в консоль.
    """
    if env_name in os.environ and os.environ[env_name]:
        return os.environ[env_name]
    # Поддержка альтернативного имени для обратной совместимости
    if env_name == "GEE_PROJ" and "GEE_PROJECT_ID" in os.environ:
        return os.environ["GEE_PROJECT_ID"]
    if (
        AIRFLOW_ACTIVE
        and "AIRFLOW_HOME" in os.environ
        and not sys.platform.startswith("win")
    ):
        try:
            from airflow.models import Variable
            return str(Variable.get(key, default_var=default_value))
        except Exception as exc:
            logger.debug("Airflow variable lookup skipped for %s: %s", key, exc)
    return default_value


EE_PROJECT = resolve_variable("gee_project_id", "GEE_PROJ", "")
EE_ACCOUNT = resolve_variable("gee_service_account", "GEE_SERVICE_ACCOUNT", "")
EE_KEY_PATH = resolve_variable(
    "gee_key_path", "GEE_KEY_PATH", "/opt/airflow/secrets/gee_key.json"
)


def gamma_map(im: np.ndarray, w: int = 3, looks: float = 5.0) -> np.ndarray:
    """Пространственный фильтр Gamma MAP для подавления спекл-шума.

    Выполняет байесовскую фильтрацию радарного сигнала в линейной шкале.
    """
    im_clipped = np.maximum(im, 1e-10)
    m = uniform_filter(im_clipped, size=w)
    v = np.maximum(uniform_filter(im_clipped**2, size=w) - m**2, 1e-10)
    ci = np.sqrt(v) / (m + 1e-9)
    cu = 1.0 / math.sqrt(looks)
    cmax = math.sqrt(2.0) * cu
    out = np.copy(im_clipped)

    hetero = (ci > cu) & (ci < cmax)
    if np.any(hetero):
        ci_h = ci[hetero]
        im_h = im_clipped[hetero]
        m_h = m[hetero]
        a = (1.0 + cu**2) / (ci_h**2 - cu**2 + 1e-9)
        t = (a - looks - 1.0) * m_h
        d = t**2 + 4.0 * a * looks * im_h * m_h
        out[hetero] = (t + np.sqrt(np.maximum(d, 0.0))) / (2.0 * a)

    out[ci <= cu] = m[ci <= cu]
    return out


def parse_dates_from_geojson(
    props: dict[str, object], root: dict[str, object]
) -> tuple[str, str]:
    """Извлекает интервал дат start_date, end_date из свойств GeoJSON."""
    time_prop = props.get("time") or root.get("time")
    if not time_prop:
        raise ValueError("Свойство 'time' отсутствует в GeoJSON.")

    if isinstance(time_prop, str):
        dt = pd.to_datetime(time_prop[:10])
        start_date = dt.strftime("%Y-%m-%d")
        end_date = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return start_date, end_date

    if isinstance(time_prop, (list, tuple)):
        if len(time_prop) == 0:
            raise ValueError("Список дат в 'time' пуст.")
        if len(time_prop) == 1:
            dt = pd.to_datetime(str(time_prop[0])[:10])
            start_date = dt.strftime("%Y-%m-%d")
            end_date = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            return start_date, end_date
        start_date = str(time_prop[0])[:10]
        end_date = str(time_prop[1])[:10]
        return start_date, end_date

    raise ValueError(f"Неподдерживаемый тип значения 'time': {type(time_prop)}")


def init_gee(
    ee_project: str = "",
    ee_account: str = "",
    ee_key_path: str = ""
) -> None:
    """Инициализирует Google Earth Engine API без вывода идентификаторов проекта."""
    import ee

    project = ee_project or EE_PROJECT or None
    account = ee_account or EE_ACCOUNT
    key_path = ee_key_path or EE_KEY_PATH

    if account and key_path and os.path.isfile(key_path):
        try:
            credentials = ee.ServiceAccountCredentials(account, key_path)
            ee.Initialize(credentials=credentials, project=project)
            return
        except Exception as exc:
            msg = f"Ошибка авторизации через Service Account: {exc}"
            raise RuntimeError(msg) from exc

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(f"Ошибка инициализации Earth Engine: {exc}") from exc


def fetch_s1_dataset(
    geojson_data: dict[str, object],
    ee_project: str = "",
    ee_account: str = "",
    ee_key_path: str = ""
) -> xr.Dataset:
    """Основной конвейер выгрузки и обработки радарных данных Sentinel-1 GRD."""
    import ee

    init_gee(ee_project=ee_project, ee_account=ee_account, ee_key_path=ee_key_path)

    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data.get("features", [])
        if not features or not isinstance(features, list):
            raise ValueError("GeoJSON FeatureCollection не содержит объектов.")
        feature = features[0]
    elif geojson_data.get("type") == "Feature":
        feature = geojson_data
    else:
        feature = {"type": "Feature", "geometry": geojson_data, "properties": {}}

    geom_dict = feature["geometry"]  # type: ignore[index]
    geom_type = geom_dict["type"]
    geom = shape(geom_dict)
    props = feature.get("properties", {})  # type: ignore[union-attr]
    is_rect = props.get("is_rectangle", False) if isinstance(props, dict) else False
    bbox = list(geom.bounds)

    start_date, end_date = parse_dates_from_geojson(
        props if isinstance(props, dict) else {},
        geojson_data
    )

    if geom_type == "Point":
        ee_geom = ee.Geometry.Point(geom_dict["coordinates"])
    elif not is_rect:
        ee_geom = ee.Geometry(geom_dict)
    else:
        ee_geom = ee.Geometry.Rectangle(bbox)

    dem_mosaic = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")
        .select("DEM")
        .mosaic()
        .unmask(0)
    )
    terrain_products = ee.Terrain.products(dem_mosaic)
    dem_slope = terrain_products.select("slope").multiply(math.pi / 180.0)
    dem_aspect = terrain_products.select("aspect").multiply(math.pi / 180.0)

    def apply_rtc(image: ee.Image) -> ee.Image:
        theta = image.select("angle").multiply(math.pi / 180.0)
        azimuth = ee.Terrain.aspect(image.select("angle")).multiply(math.pi / 180.0)

        cos_loc = (theta.cos().multiply(dem_slope.cos())).add(
            theta.sin()
            .multiply(dem_slope.sin())
            .multiply(dem_aspect.subtract(azimuth).cos())
        )
        slope_factor = (
            theta.sin()
            .divide(cos_loc.abs().max(0.1))
            .clamp(0.05, 50.0)
        )
        linear_bands = (
            ee.Image.constant(10.0)
            .pow(image.select(["VV", "VH"]).divide(10.0))
            .rename(["VV", "VH"])
        )
        rtc_img = linear_bands.multiply(slope_factor).rename(["VV", "VH"]).toFloat()
        return rtc_img.set("system:time_start", image.get("system:time_start"))

    base_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee_geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .sort("system:time_start")
    )

    timestamps_raw = base_col.aggregate_array("system:time_start").getInfo()
    if not timestamps_raw:
        warnings.warn(
            f"Снимки Sentinel-1 за период {start_date} - {end_date} не найдены.",
            stacklevel=2
        )
        return xr.Dataset()

    dt_index = pd.to_datetime(timestamps_raw, unit="ms", utc=True)
    unique_dates_df = pd.DataFrame({"ts_ms": timestamps_raw, "dt": dt_index})
    unique_dates_df["date_str"] = unique_dates_df["dt"].dt.strftime("%Y-%m-%d")
    daily_records = (
        unique_dates_df.drop_duplicates(subset=["date_str"]).to_dict("records")
    )

    all_data = []
    timestamps = []
    out_lats = None
    out_lons = None

    for record in daily_records:
        current_ms = record["ts_ms"]
        current_dt = record["dt"].tz_localize(None)
        current_date_ee = ee.Date(current_ms)
        window_start = current_date_ee.advance(-3, "month")
        window_end = current_date_ee.advance(1, "day")

        temp_col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(ee_geom)
            .filterDate(window_start, window_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .map(apply_rtc)
        )

        median_img = temp_col.median()

        if geom_type == "Point":
            stats_feat = median_img.sample(ee_geom, scale=20).first()
            if stats_feat is None:
                vv_val = np.nan
                vh_val = np.nan
            else:
                stats = stats_feat.toDictionary().getInfo()
                vv_val = stats.get("VV", np.nan)
                vh_val = stats.get("VH", np.nan)

            vv = np.array([[vv_val]], dtype=np.float32)
            vh = np.array([[vh_val]], dtype=np.float32)
            if out_lats is None:
                out_lats = np.array([bbox[1]], dtype=np.float64)
                out_lons = np.array([bbox[0]], dtype=np.float64)
        else:
            # Выгрузка через ee.data.computePixels с фиксированным порядком каналов
            struct_array = ee.data.computePixels({
                "expression": median_img.clipToBoundsAndScale(
                    geometry=ee_geom, scale=20
                ),
                "fileFormat": "NUMPY_NDARRAY",
                "bandIds": ["VV", "VH"]
            })
            vv = struct_array["VV"].astype(np.float32)
            vh = struct_array["VH"].astype(np.float32)

            if out_lats is None:
                ny, nx = vv.shape
                d_lat = (bbox[3] - bbox[1]) / float(ny)
                d_lon = (bbox[2] - bbox[0]) / float(nx)
                out_lats = np.array(
                    [bbox[3] - (i + 0.5) * d_lat for i in range(ny)],
                    dtype=np.float64
                )
                out_lons = np.array(
                    [bbox[0] + (j + 0.5) * d_lon for j in range(nx)],
                    dtype=np.float64
                )

        if vv.size > 1:
            mask_valid = ~np.isnan(vv) & (vv > 0)
            vv_f = np.full_like(vv, np.nan)
            vh_f = np.full_like(vh, np.nan)
            if np.any(mask_valid):
                vv_fill = np.nan_to_num(vv, nan=float(np.nanmedian(vv[mask_valid])))
                vh_fill = np.nan_to_num(vh, nan=float(np.nanmedian(vh[mask_valid])))
                vv_f[mask_valid] = gamma_map(vv_fill, w=3, looks=5.0)[mask_valid]
                vh_f[mask_valid] = gamma_map(vh_fill, w=3, looks=5.0)[mask_valid]
        else:
            vv_f = vv
            vh_f = vh

        vv_db = 10.0 * np.log10(np.maximum(vv_f, 1e-6))
        vh_db = 10.0 * np.log10(np.maximum(vh_f, 1e-6))

        all_data.append(np.stack([vv_db, vh_db], axis=-1))
        timestamps.append(current_dt)

    if not all_data:
        warnings.warn("Выборка пуста после обработки снимков.", stacklevel=2)
        return xr.Dataset()

    data_stack = np.stack(all_data, axis=0)

    return xr.Dataset(
        data_vars={
            "VV": (
                ["time", "latitude", "longitude"],
                data_stack[:, :, :, 0].astype(np.float32),
            ),
            "VH": (
                ["time", "latitude", "longitude"],
                data_stack[:, :, :, 1].astype(np.float32),
            ),
            "ratio": (
                ["time", "latitude", "longitude"],
                (data_stack[:, :, :, 0] - data_stack[:, :, :, 1]).astype(np.float32),
            ),
        },
        coords={
            "time": timestamps,
            "latitude": out_lats,
            "longitude": out_lons,
        },
        attrs={
            "units": "dB",
            "resolution_meters": 20,
            "description": "Sentinel-1 RTC Gamma-MAP processed SAR data",
        },
    )


def run(
    geojson: dict[str, object] | str,
    output_path: str | None = None,
    ee_project: str = "",
    ee_account: str = "",
    ee_key_path: str = ""
) -> xr.Dataset:
    """Точка входа модуля для вызова из Python скриптов и тестов."""
    if isinstance(geojson, str):
        if not os.path.exists(geojson):
            raise FileNotFoundError(f"Входной файл не найден: {geojson}")
        if os.path.getsize(geojson) == 0:
            raise ValueError(f"Входной файл пуст: {geojson}")
        with open(geojson, encoding="utf-8") as f:
            geojson_data = json.load(f)
    else:
        geojson_data = geojson

    ds = fetch_s1_dataset(
        geojson_data=geojson_data,
        ee_project=ee_project,
        ee_account=ee_account,
        ee_key_path=ee_key_path,
    )

    if output_path and len(ds.data_vars) > 0:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        temp_nc = os.path.join(
            tempfile.gettempdir(), f"s1_out_{os.getpid()}_{os.urandom(4).hex()}.nc"
        )
        try:
            ds.to_netcdf(temp_nc)
            shutil.move(temp_nc, output_path)
        finally:
            if os.path.exists(temp_nc):
                os.remove(temp_nc)

    return ds


if AIRFLOW_ACTIVE:
    with DAG(
        DAG_ID,
        description="Parser for Sentinel-1 SAR (RTC + Gamma MAP Filter)",
        default_args=DEFAULT_ARGS,
        schedule=None,
        catchup=False,
        start_date=datetime(2025, 1, 1),
    ) as dag:

        @task.virtualenv(
            task_id="extract_s1_sar_data",
            requirements=[
                "pandas",
                "numpy",
                "earthengine-api",
                "xarray",
                "shapely",
                "scipy",
                "netCDF4",
            ],
        )
        def parse_s1_data(
            file_to_read: str,
            file_to_write: str,
            ee_project: str = "",
            ee_account: str = "",
            ee_key_path: str = "",
        ) -> None:
            """Задача Airflow для выгрузки данных Sentinel-1."""
            run(
                geojson=file_to_read,
                output_path=file_to_write,
                ee_project=ee_project,
                ee_account=ee_account,
                ee_key_path=ee_key_path,
            )

        parse_task = parse_s1_data(
            file_to_read=FILE_TO_READ,
            file_to_write=FILE_TO_WRITE,
            ee_project=EE_PROJECT,
            ee_account=EE_ACCOUNT,
            ee_key_path=EE_KEY_PATH,
        )


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Sentinel-1 SAR Loader")
    cli_parser.add_argument(
        "--input", default=FILE_TO_READ, help="Путь к input_geo.json"
    )
    cli_parser.add_argument(
        "--output", default=FILE_TO_WRITE, help="Путь к output_data.nc"
    )
    cli_parser.add_argument(
        "--project",
        default="",
        help="ID проекта Google Cloud (по умолчанию из переменной GEE_PROJ)"
    )
    cli_parser.add_argument(
        "--account", default=EE_ACCOUNT, help="Email сервисного аккаунта"
    )
    cli_parser.add_argument(
        "--key", default=EE_KEY_PATH, help="Путь к JSON ключу"
    )
    args = cli_parser.parse_args()

    if not os.path.exists(args.input) or os.path.getsize(args.input) == 0:
        sample_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "time": ["2024-05-01", "2024-05-20"],
                        "is_rectangle": True,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [30.50, 50.40],
                                [30.55, 50.40],
                                [30.55, 50.45],
                                [30.50, 50.45],
                                [30.50, 50.40],
                            ]
                        ],
                    },
                }
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.input)), exist_ok=True)
        with open(args.input, "w", encoding="utf-8") as f_out:
            json.dump(sample_geojson, f_out, indent=2)

    run(
        geojson=args.input,
        output_path=args.output,
        ee_project=args.project,
        ee_account=args.account,
        ee_key_path=args.key,
    )
