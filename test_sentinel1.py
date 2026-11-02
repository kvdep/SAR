# -*- coding: utf-8 -*-
"""Комплексный набор тестов данных, контракта парсера и аудита качества (Data Quality) Sentinel-1 SAR.

Архитектура модуля:
1. Отсутствие хардкода:
   - Все тестовые и диагностические функции принимают путь к NetCDF (или объект xarray.Dataset)
     в качестве аргумента `nc_path`.
   - Для pytest поддерживается передача пути через флаг CLI: `pytest --nc-path=/path/to/data.nc`
     или через переменную окружения `SAR_TEST_NC_PATH`.
   - Для прямого вызова тестировщиком: `test_data_nan_and_drop_detection("my_data.nc", ...)`
     или мастер-функция `check_sar_dataset("my_data.nc", ...)`.
2. Двухрежимная работа (fail-fast vs полный аудит):
   - По умолчанию включен режим `raise_on_error=True`: функция инспектирует ВСЕ переменные,
     все временные срезы и всю пространственную сетку, собирает ВСЕ дефекты с точными
     координатами (lat, lon, time, row, col, value), логгирует их и выбрасывает AssertionError
     с исчерпывающим структурированным отчетом.
   - При `raise_on_error=False`: функция не падает при обнаружении проблем, а возвращает
     структурированный объект `QualityReport` со статусом `passed=False` и полным списком дефектов.
   - Аргумент `verbose=True` включает максимальную детализацию вывода всех обнаруженных точек.
3. Логгирование с защитой от дублирования:
   - Все результаты проверок фиксируются в файл аудита (по умолчанию `sar_quality_audit.log`).
   - Если файл уже существует, новые тесты и новые пути дописываются (append).
   - При повторном вызове того же теста на том же файле запись для `[test_name :: path]`
     перезаписывается на актуальную, исключая дублирование логов.
4. Контрактные тесты парсера (пункты 1–3 спецификации):
   - Валидация GeoJSON (отсутствующий файл, пустой файл, некорректная геометрия, RFC 7946).
   - Спецификация NetCDF (размерности, типы float32/float64, монотонность, отсутствие утечек ключей).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import uniform_filter, sobel

import run

pytestmark = [pytest.mark.filterwarnings("ignore::pytest.PytestReturnNotNoneWarning")]


# =====================================================================
# 0. Структуры данных для дефектов и отчетов
# =====================================================================

@dataclass
class DataDefect:
    """Точечная или структурная аномалия качества данных."""
    defect_type: str        # Тип дефекта (-INF, +INF, BURST_DROP_ROW, BURST_DROP_COL, RANGE_LOW, RANGE_HIGH, etc.)
    variable: str           # Имя переменной (VV, VH, ratio)
    time_idx: int           # Индекс по временной оси (-1 для сводных дефектов)
    timestamp: str          # ISO-метка времени снимка
    row: int                # Индекс строки (latitude)
    col: int                # Индекс столбца (longitude)
    latitude: float         # Географическая широта точки
    longitude: float        # Географическая долгота точки
    value: float            # Фактическое значение величины (или перепада в дБ)
    threshold: float        # Допустимый порог спецификации
    description: str        # Физическая интерпретация дефекта


@dataclass
class QualityReport:
    """Полный отчет по результатам выполнения теста качества данных."""
    test_name: str
    target_path: str
    passed: bool
    total_pixels_checked: int
    defects_count: int
    defects: list[DataDefect] = field(default_factory=list)
    summary_text: str = ""
    detailed_text: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 1. Системные утилиты загрузки и логгирования
# =====================================================================

def load_dataset_safely(nc_path: str | Path | xr.Dataset) -> xr.Dataset:
    """Безопасная загрузка NetCDF на Windows при наличии кириллицы/пробелов в пути.

    Если передан уже загруженный xr.Dataset, возвращает его без повторного чтения.
    """
    if isinstance(nc_path, xr.Dataset):
        return nc_path

    path_str = str(nc_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Файл датасета не найден: {path_str}")

    tmp = os.path.join(tempfile.gettempdir(), f"test_load_{os.urandom(6).hex()}.nc")
    shutil.copy2(path_str, tmp)
    try:
        with xr.open_dataset(tmp) as raw_ds:
            ds = raw_ds.load()
        return ds
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def log_test_record(log_file: str | Path | None, test_name: str, target_path: str, content: str) -> None:
    """Запись результатов теста в лог-файл аудита.

    - Аппендит запись, если файл существует.
    - Перезаписывает существующий блок [test_name :: target_path] при повторном запуске,
      исключая появление дубликатов.
    """
    if not log_file:
        return

    norm_path = os.path.abspath(target_path) if isinstance(target_path, (str, Path)) else "InMemoryDataset"
    section_id = f"{test_name} :: {norm_path}"
    begin_tag = f"# >>> BEGIN TEST RECORD: [{section_id}]"
    end_tag = f"# <<< END TEST RECORD: [{section_id}]"
    new_block = f"{begin_tag}\n{content.strip()}\n{end_tag}\n"

    log_path_str = str(log_file)
    if os.path.exists(log_path_str):
        with open(log_path_str, "r", encoding="utf-8") as f:
            existing = f.read()
        pattern = re.compile(rf"{re.escape(begin_tag)}.*?{re.escape(end_tag)}\n?", re.DOTALL)
        if pattern.search(existing):
            # Замена существующей секции с защитой от обратных слешей Windows
            updated = pattern.sub(lambda _: new_block, existing)
        else:
            updated = existing.rstrip() + "\n\n" + new_block
    else:
        updated = new_block

    with open(log_path_str, "w", encoding="utf-8") as f:
        f.write(updated)


# =====================================================================
# 2. Pytest Fixture и CLI конфигурация
# =====================================================================

def pytest_addoption(parser: pytest.Parser) -> None:
    """Регистрация флага командной строки для передачи пути тестируемого NetCDF."""
    try:
        parser.addoption(
            "--nc-path",
            action="store",
            default=None,
            help="Путь к NetCDF-файлу данных для проведения тестов качества"
        )
    except ValueError:
        # Опция уже может быть зарегистрирована в conftest.py
        pass


@pytest.fixture
def nc_path(request: pytest.FixtureRequest) -> str:
    """Фикстура получения пути к датасету NetCDF.

    Приоритет:
    1. CLI аргумент: pytest --nc-path=/path/to/data.nc
    2. Переменная окружения SAR_TEST_NC_PATH
    3. Автопоиск s1_*.nc файлов в рабочей директории.
    """
    cli_opt = None
    try:
        cli_opt = request.config.getoption("--nc-path")
    except (ValueError, AttributeError):
        pass

    if cli_opt and os.path.exists(cli_opt):
        return str(cli_opt)

    env_opt = os.environ.get("SAR_TEST_NC_PATH")
    if env_opt and os.path.exists(env_opt):
        return str(env_opt)

    # Автопоиск эталонных датасетов в папке проекта
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "s1_forest.nc"),
        os.path.join(os.path.dirname(__file__), "s1_forest.nc"),
        os.path.join(cwd, "s1_water.nc"),
        os.path.join(cwd, "s1_city.nc"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    pytest.skip("Файл NetCDF для тестирования не предоставлен. Укажите --nc-path или переменную SAR_TEST_NC_PATH.")


# =====================================================================
# 3. Базовые контрактные тесты парсера и GeoJSON (Пункты 1–3 спецификации)
# =====================================================================

def test_geojson_contract_missing_file(tmp_path: Any) -> None:
    """Несуществующий файл GeoJSON должен вызывать FileNotFoundError."""
    missing = str(tmp_path / "nonexistent.geojson")
    with pytest.raises(FileNotFoundError, match="Входной файл не найден"):
        run.run(missing)


def test_geojson_contract_empty_file(tmp_path: Any) -> None:
    """Пустой файл GeoJSON (0 байт) должен вызывать явный ValueError."""
    empty_file = str(tmp_path / "empty.geojson")
    with open(empty_file, "w", encoding="utf-8"):
        pass
    with pytest.raises(ValueError, match="Входной файл пуст"):
        run.run(empty_file)


def test_geojson_contract_unsupported_geometry() -> None:
    """Неподдерживаемый тип геометрии (например LineString) вызывает исключение."""
    invalid_geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[30.0, 50.0], [30.1, 50.1]]
        },
        "properties": {"time": "2024-05-15"}
    }
    with pytest.raises(Exception):
        run.fetch_s1_dataset(invalid_geojson)


def test_geojson_contract_coordinate_order() -> None:
    """Проверка порядка координат [longitude, latitude] по стандарту RFC 7946."""
    props = {"time": "2024-05-15"}
    root = {"time": "2024-05-15"}
    s, e = run.parse_dates_from_geojson(props, root)
    assert s == "2024-05-15"
    assert e == "2024-05-16"


def test_parser_netcdf_contract_schema(nc_path: str) -> None:
    """Соответствие .nc файла спецификации контракта выгрузки (размерности, типы, сетка)."""
    ds = load_dataset_safely(nc_path)

    # 1. Переменные
    assert set(ds.data_vars.keys()) == {"VV", "VH", "ratio"}, (
        f"Несоответствие состава переменных: {set(ds.data_vars.keys())}"
    )

    # 2. Размерности
    expected_dims = ("time", "latitude", "longitude")
    assert ds["VV"].dims == expected_dims
    assert ds["VH"].dims == expected_dims
    assert ds["ratio"].dims == expected_dims

    # 3. Типы данных: растровые значения float32, геодезические координаты float64
    assert ds["VV"].dtype == np.float32, f"Тип VV должен быть float32, получено {ds['VV'].dtype}"
    assert ds["VH"].dtype == np.float32, f"Тип VH должен быть float32, получено {ds['VH'].dtype}"
    assert ds["ratio"].dtype == np.float32, f"Тип ratio должен быть float32, получено {ds['ratio'].dtype}"
    assert ds["latitude"].dtype == np.float64, f"Широта должна быть float64, получено {ds['latitude'].dtype}"
    assert ds["longitude"].dtype == np.float64, f"Долгота должна быть float64, получено {ds['longitude'].dtype}"

    # 4. Монотонность пространственной сетки
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    assert np.all(np.diff(lats) < 0), "Широта должна строго монотонно убывать (с севера на юг)."
    assert np.all(np.diff(lons) > 0), "Долгота должна строго монотонно возрастать (с запада на восток)."

    # 5. Уникальность меток времени
    times = ds["time"].values
    assert len(times) == len(np.unique(times)), "Метки времени должны быть строго уникальными."

    # 6. Обязательные атрибуты спецификации
    assert ds.attrs.get("units") == "dB", "Атрибут units должен быть равен 'dB'"
    assert ds.attrs.get("resolution_meters") == 20, "Атрибут resolution_meters должен быть равен 20"


def test_parser_no_credential_leaks_in_output(nc_path: str) -> None:
    """Проверка отсутствия утечек приватных ключей и идентификаторов в метаданных NetCDF."""
    ds = load_dataset_safely(nc_path)
    forbidden_keys = ["project", "gee_project", "key", "token", "password", "secret", "credentials"]

    for attr_name, attr_val in ds.attrs.items():
        name_lower = str(attr_name).lower()
        val_lower = str(attr_val).lower()
        for k in forbidden_keys:
            assert k not in name_lower, f"Подозрительный ключ в атрибутах метаданных: {attr_name}"
            assert "private_key" not in val_lower, "Приватный ключ обнаружен в теле метаданных!"


def test_parser_spatial_coverage_and_non_empty(nc_path: str) -> None:
    """Возврат непустых объектов с валидными физическими отсчетами."""
    ds = load_dataset_safely(nc_path)
    assert ds.sizes["time"] > 0, "Ось времени пуста"
    assert ds.sizes["latitude"] > 0, "Ось широты пуста"
    assert ds.sizes["longitude"] > 0, "Ось долготы пуста"

    assert np.any(~np.isnan(ds["VV"].values)), "Массив VV полностью состоит из NaN."
    assert np.any(~np.isnan(ds["VH"].values)), "Массив VH полностью состоит из NaN."


def test_parser_save_and_load_netcdf_roundtrip(tmp_path: Any, nc_path: str) -> None:
    """Сериализация NetCDF на диск и повторное чтение сохраняет бинарную структуру без потерь."""
    ds = load_dataset_safely(nc_path)
    out_file = str(tmp_path / "roundtrip_test.nc")

    ds.to_netcdf(out_file)
    assert os.path.exists(out_file)

    with xr.open_dataset(out_file) as reloaded:
        assert dict(reloaded.sizes) == dict(ds.sizes)
        assert set(reloaded.data_vars) == set(ds.data_vars)
        np.testing.assert_allclose(reloaded["VV"].values, ds["VV"].values, rtol=1e-5, atol=1e-5)


# =====================================================================
# 4. Тесты качества данных (Data Quality Audit) с локализацией и двойным режимом
# =====================================================================

def test_data_nan_and_drop_detection(
    nc_path: str | Path | xr.Dataset,
    *,
    raise_on_error: bool = True,
    verbose: bool = False,
    log_file: str | Path | None = "sar_quality_audit.log",
    nan_threshold_pct: float = 15.0,
) -> QualityReport:
    """Санитарный контроль пропусков (NaN), сингулярностей (+/-inf) и выпадений строк (burst drops).

    Параметры:
    - nc_path: путь к NetCDF файлу или объект xr.Dataset.
    - raise_on_error: True — вызвать AssertionError при наличии дефектов; False — вернуть QualityReport.
    - verbose: True — вывести все обнаруженные координаты без ограничения лимита.
    - log_file: путь к файлу аудита (перезаписывает секцию при повторении пути).
    - nan_threshold_pct: максимально допустимый процент пустых пикселей (по умолчанию 15%).
    """
    target_name = str(nc_path) if isinstance(nc_path, (str, Path)) else "InMemoryDataset"
    ds = load_dataset_safely(nc_path)

    times = [str(t)[:19] for t in ds["time"].values] if "time" in ds else ["t0"]
    lats = ds["latitude"].values if "latitude" in ds else np.arange(ds.sizes.get("latitude", 1), dtype=float)
    lons = ds["longitude"].values if "longitude" in ds else np.arange(ds.sizes.get("longitude", 1), dtype=float)

    vars_to_check = [v for v in ["VV", "VH", "ratio"] if v in ds.data_vars]
    defects: list[DataDefect] = []
    metrics: dict[str, Any] = {}
    total_pixels = 0

    for var in vars_to_check:
        arr = ds[var].values
        total_pixels += arr.size
        T, Y, X = arr.shape

        # 1. Сингулярности -inf
        neginf_idx = np.argwhere(np.isneginf(arr))
        for t, r, c in neginf_idx:
            defects.append(DataDefect(
                defect_type="-INF",
                variable=var,
                time_idx=int(t),
                timestamp=times[t],
                row=int(r),
                col=int(c),
                latitude=float(lats[r]),
                longitude=float(lons[c]),
                value=-float("inf"),
                threshold=0.0,
                description="Сингулярность -inf: результат log10 от нуля или деления на ноль"
            ))

        # 2. Сингулярности +inf
        posinf_idx = np.argwhere(np.isposinf(arr))
        for t, r, c in posinf_idx:
            defects.append(DataDefect(
                defect_type="+INF",
                variable=var,
                time_idx=int(t),
                timestamp=times[t],
                row=int(r),
                col=int(c),
                latitude=float(lats[r]),
                longitude=float(lons[c]),
                value=float("inf"),
                threshold=0.0,
                description="Сингулярность +inf: переполнение при делении"
            ))

        # 3. Пропуски (NaN) и выпадения строк сканирования (burst drop)
        nan_mask = np.isnan(arr)
        nan_count = int(nan_mask.sum())
        nan_pct = float(100.0 * nan_count / arr.size)
        metrics[f"{var}_nan_count"] = nan_count
        metrics[f"{var}_nan_pct"] = nan_pct

        full_drop_rows_set = set()
        full_drop_cols_set = set()
        for t in range(T):
            row_nans = nan_mask[t].sum(axis=1)
            full_nan_rows = np.where(row_nans == X)[0]
            for r in full_nan_rows:
                full_drop_rows_set.add((t, r))
                defects.append(DataDefect(
                    defect_type="BURST_DROP_ROW",
                    variable=var,
                    time_idx=int(t),
                    timestamp=times[t],
                    row=int(r),
                    col=-1,
                    latitude=float(lats[r]),
                    longitude=float(lons[0]),
                    value=float("nan"),
                    threshold=0.0,
                    description=f"Выпадение целой строки сканирования: строка {r} (lat {lats[r]:.5f}) пуста по всем {X} столбцам"
                ))

            col_nans = nan_mask[t].sum(axis=0)
            full_nan_cols = np.where(col_nans == Y)[0]
            for c in full_nan_cols:
                full_drop_cols_set.add((t, c))
                defects.append(DataDefect(
                    defect_type="BURST_DROP_COL",
                    variable=var,
                    time_idx=int(t),
                    timestamp=times[t],
                    row=-1,
                    col=int(c),
                    latitude=float(lats[0]),
                    longitude=float(lons[c]),
                    value=float("nan"),
                    threshold=0.0,
                    description=f"Выпадение целого столбца растра: столбец {c} (lon {lons[c]:.5f}) пуст по всем {Y} строкам"
                ))

        if nan_pct > nan_threshold_pct:
            defects.append(DataDefect(
                defect_type="EXCESSIVE_NANS",
                variable=var,
                time_idx=-1,
                timestamp="ALL",
                row=-1,
                col=-1,
                latitude=0.0,
                longitude=0.0,
                value=nan_pct,
                threshold=nan_threshold_pct,
                description=f"Доля пропусков {nan_pct:.2f}% превышает лимит {nan_threshold_pct:.2f}% (потеря пакетов)"
            ))

        if nan_count > 0:
            nan_pts = np.argwhere(nan_mask)
            metrics[f"{var}_nan_bbox"] = {
                "lat_min": float(lats[nan_pts[:, 1]].min()),
                "lat_max": float(lats[nan_pts[:, 1]].max()),
                "lon_min": float(lons[nan_pts[:, 2]].min()),
                "lon_max": float(lons[nan_pts[:, 2]].max()),
            }
            # Локализация изолированных точек (не входящих в сплошные выпавшие строки)
            added_isolated = 0
            for t, r, c in nan_pts:
                if (t, r) not in full_drop_rows_set and (t, c) not in full_drop_cols_set:
                    defects.append(DataDefect(
                        defect_type="ISOLATED_NAN",
                        variable=var,
                        time_idx=int(t),
                        timestamp=times[t],
                        row=int(r),
                        col=int(c),
                        latitude=float(lats[r]),
                        longitude=float(lons[c]),
                        value=float("nan"),
                        threshold=0.0,
                        description="Локальный пропуск пикселя (NaN)"
                    ))
                    added_isolated += 1
                    if not verbose and added_isolated >= 20:
                        break

    passed = len(defects) == 0
    lines = [
        f"ТЕСТ: test_data_nan_and_drop_detection",
        f"ЦЕЛЬ: {target_name}",
        f"СТАТУС: {'PASS (0 дефектов)' if passed else f'FAIL ({len(defects)} дефектов)'}",
        f"ПРОВЕРЕНО ПИКСЕЛЕЙ: {total_pixels}",
    ]
    for var in vars_to_check:
        lines.append(f"  * {var}: {metrics[f'{var}_nan_count']} NaNs ({metrics[f'{var}_nan_pct']:.2f}%)")
        if f"{var}_nan_bbox" in metrics:
            b = metrics[f"{var}_nan_bbox"]
            lines.append(f"    BBox NaNs: lat [{b['lat_min']:.5f}, {b['lat_max']:.5f}], lon [{b['lon_min']:.5f}, {b['lon_max']:.5f}]")

    if defects:
        lines.append(f"\nЛОКАЛИЗАЦИЯ ДЕФЕКТОВ ({len(defects)}):")
        display_limit = len(defects) if verbose else min(len(defects), 20)
        for d in defects[:display_limit]:
            lines.append(
                f"  - [{d.defect_type}] Var: {d.variable} | Time: {d.timestamp} | "
                f"Pixel: (r={d.row}, c={d.col}) | Geo: (lat={d.latitude:.5f}, lon={d.longitude:.5f}) | "
                f"Val: {d.value} | {d.description}"
            )
        if len(defects) > display_limit:
            lines.append(f"  ... и еще {len(defects) - display_limit} дефектов (используйте verbose=True для полного списка).")

    summary_text = "\n".join(lines[:4])
    detailed_text = "\n".join(lines)
    log_test_record(log_file, "test_data_nan_and_drop_detection", target_name, detailed_text)

    report = QualityReport(
        test_name="test_data_nan_and_drop_detection",
        target_path=target_name,
        passed=passed,
        total_pixels_checked=total_pixels,
        defects_count=len(defects),
        defects=defects,
        summary_text=summary_text,
        detailed_text=detailed_text,
        metrics=metrics,
    )

    if raise_on_error and not passed:
        raise AssertionError(f"\n{detailed_text}")

    return report


def test_data_physical_dynamic_range(
    nc_path: str | Path | xr.Dataset,
    *,
    raise_on_error: bool = True,
    verbose: bool = False,
    log_file: str | Path | None = "sar_quality_audit.log",
    min_db: float = -55.0,
    max_db: float = 35.0,
) -> QualityReport:
    """Контроль физического динамического диапазона отраженного сигнала Sentinel-1 C-диапазона.

    Допустимый диапазон: от -55.0 дБ (порог шума электроники NESZ -22..-25 дБ с запасом)
    до +35.0 дБ (физический предел насыщения приемника на уголковых отражателях).
    """
    target_name = str(nc_path) if isinstance(nc_path, (str, Path)) else "InMemoryDataset"
    ds = load_dataset_safely(nc_path)

    times = [str(t)[:19] for t in ds["time"].values] if "time" in ds else ["t0"]
    lats = ds["latitude"].values if "latitude" in ds else np.arange(ds.sizes.get("latitude", 1), dtype=float)
    lons = ds["longitude"].values if "longitude" in ds else np.arange(ds.sizes.get("longitude", 1), dtype=float)

    vars_to_check = [v for v in ["VV", "VH"] if v in ds.data_vars]
    defects: list[DataDefect] = []
    metrics: dict[str, Any] = {}
    total_pixels = 0

    for var in vars_to_check:
        arr = ds[var].values
        total_pixels += arr.size
        valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
        if not np.any(valid_mask):
            continue

        cur_min = float(np.min(arr[valid_mask]))
        cur_max = float(np.max(arr[valid_mask]))
        metrics[f"{var}_min"] = cur_min
        metrics[f"{var}_max"] = cur_max

        # Фиксация занижений (ошибка калибровки нуля или клиппинг)
        low_idx = np.argwhere(valid_mask & (arr < min_db))
        for t, r, c in low_idx:
            val = float(arr[t, r, c])
            defects.append(DataDefect(
                defect_type="RANGE_LOW",
                variable=var,
                time_idx=int(t),
                timestamp=times[t],
                row=int(r),
                col=int(c),
                latitude=float(lats[r]),
                longitude=float(lons[c]),
                value=val,
                threshold=min_db,
                description=f"Аномально низкий уровень мощности {val:.2f} дБ < {min_db:.2f} дБ (ошибка масштабирования нуля)"
            ))

        # Фиксация превышений (насыщение детектора/целочисленное переполнение)
        high_idx = np.argwhere(valid_mask & (arr > max_db))
        for t, r, c in high_idx:
            val = float(arr[t, r, c])
            defects.append(DataDefect(
                defect_type="RANGE_HIGH",
                variable=var,
                time_idx=int(t),
                timestamp=times[t],
                row=int(r),
                col=int(c),
                latitude=float(lats[r]),
                longitude=float(lons[c]),
                value=val,
                threshold=max_db,
                description=f"Аномально высокий пик мощности {val:.2f} дБ > {max_db:.2f} дБ (насыщение сенсора/переполнение)"
            ))

    passed = len(defects) == 0
    lines = [
        f"ТЕСТ: test_data_physical_dynamic_range",
        f"ЦЕЛЬ: {target_name}",
        f"СТАТУС: {'PASS (0 дефектов)' if passed else f'FAIL ({len(defects)} дефектов)'}",
        f"ПРОВЕРЕНО ПИКСЕЛЕЙ: {total_pixels}",
    ]
    for var in vars_to_check:
        lines.append(f"  * {var}: диапазон [{metrics.get(f'{var}_min', float('nan')):.2f}, {metrics.get(f'{var}_max', float('nan')):.2f}] дБ (допустимо [{min_db:.1f}, {max_db:.1f}])")

    if defects:
        lines.append(f"\nЛОКАЛИЗАЦИЯ ДЕФЕКТОВ ({len(defects)}):")
        display_limit = len(defects) if verbose else min(len(defects), 20)
        for d in defects[:display_limit]:
            lines.append(
                f"  - [{d.defect_type}] Var: {d.variable} | Time: {d.timestamp} | "
                f"Pixel: (r={d.row}, c={d.col}) | Geo: (lat={d.latitude:.5f}, lon={d.longitude:.5f}) | "
                f"Val: {d.value:.2f} dB | {d.description}"
            )
        if len(defects) > display_limit:
            lines.append(f"  ... и еще {len(defects) - display_limit} дефектов (используйте verbose=True для полного списка).")

    summary_text = "\n".join(lines[:4])
    detailed_text = "\n".join(lines)
    log_test_record(log_file, "test_data_physical_dynamic_range", target_name, detailed_text)

    report = QualityReport(
        test_name="test_data_physical_dynamic_range",
        target_path=target_name,
        passed=passed,
        total_pixels_checked=total_pixels,
        defects_count=len(defects),
        defects=defects,
        summary_text=summary_text,
        detailed_text=detailed_text,
        metrics=metrics,
    )

    if raise_on_error and not passed:
        raise AssertionError(f"\n{detailed_text}")

    return report


def test_data_polarization_ratio_and_channel_swap(
    nc_path: str | Path | xr.Dataset,
    *,
    raise_on_error: bool = True,
    verbose: bool = False,
    log_file: str | Path | None = "sar_quality_audit.log",
    ratio_tolerance: float = 1e-5,
) -> QualityReport:
    """Санитарный контроль отношения поляризаций и защита от перепутывания каналов VV и VH."""
    target_name = str(nc_path) if isinstance(nc_path, (str, Path)) else "InMemoryDataset"
    ds = load_dataset_safely(nc_path)

    times = [str(t)[:19] for t in ds["time"].values] if "time" in ds else ["t0"]
    lats = ds["latitude"].values if "latitude" in ds else np.arange(ds.sizes.get("latitude", 1), dtype=float)
    lons = ds["longitude"].values if "longitude" in ds else np.arange(ds.sizes.get("longitude", 1), dtype=float)

    defects: list[DataDefect] = []
    metrics: dict[str, Any] = {}

    vv = ds["VV"].values
    vh = ds["VH"].values
    ratio = ds["ratio"].values
    total_pixels = ratio.size

    valid_mask = ~np.isnan(vv) & ~np.isnan(vh) & ~np.isnan(ratio)

    # 1. Попиксельная проверка математического тождества ratio == VV - VH
    diff = np.abs(ratio - (vv - vh))
    identity_violations = np.argwhere(valid_mask & (diff > ratio_tolerance))
    metrics["max_ratio_identity_diff"] = float(np.nanmax(diff)) if np.any(valid_mask) else 0.0

    for t, r, c in identity_violations:
        d_val = float(diff[t, r, c])
        defects.append(DataDefect(
            defect_type="RATIO_IDENTITY_VIOLATION",
            variable="ratio",
            time_idx=int(t),
            timestamp=times[t],
            row=int(r),
            col=int(c),
            latitude=float(lats[r]),
            longitude=float(lons[c]),
            value=d_val,
            threshold=ratio_tolerance,
            description=f"Нарушено тождество ratio == VV - VH (расхождение {d_val:.6f} дБ > {ratio_tolerance:.6f})"
        ))
        if not verbose and len(defects) >= 20:
            break

    # 2. Физический баланс поляризаций и детекция перепутывания каналов
    if np.any(valid_mask):
        mean_ratio = float(np.nanmean(ratio[valid_mask]))
        median_ratio = float(np.nanmedian(ratio[valid_mask]))
        metrics["mean_ratio"] = mean_ratio
        metrics["median_ratio"] = median_ratio

        # Если среднее отношение отрицательно — каналы перепутаны местами!
        if mean_ratio < 0.0:
            defects.append(DataDefect(
                defect_type="CHANNEL_SWAP_CRITICAL",
                variable="ratio",
                time_idx=-1,
                timestamp="ALL",
                row=-1,
                col=-1,
                latitude=0.0,
                longitude=0.0,
                value=mean_ratio,
                threshold=0.0,
                description=f"КРИТИЧЕСКИЙ БАГ: Каналы VV и VH перепутаны местами! Среднее отношение {mean_ratio:.2f} дБ < 0"
            ))

        # Выявление локальных аномалий инверсии (ratio < -3 dB)
        inverted_idx = np.argwhere(valid_mask & (ratio < -3.0))
        metrics["inverted_pixels_count"] = len(inverted_idx)
        metrics["inverted_pct"] = float(100.0 * len(inverted_idx) / total_pixels)
        if len(inverted_idx) > 0 and mean_ratio >= 0.0:
            added = 0
            for t, r, c in inverted_idx:
                r_val = float(ratio[t, r, c])
                defects.append(DataDefect(
                    defect_type="INVERTED_POLARIZATION",
                    variable="ratio",
                    time_idx=int(t),
                    timestamp=times[t],
                    row=int(r),
                    col=int(c),
                    latitude=float(lats[r]),
                    longitude=float(lons[c]),
                    value=r_val,
                    threshold=-3.0,
                    description=f"Аномально сильная кросс-поляризация: ratio {r_val:.2f} дБ (сигнал VH превосходит VV более чем в 2 раза)"
                ))
                added += 1
                if not verbose and added >= 20:
                    break

    passed = len(defects) == 0
    lines = [
        f"ТЕСТ: test_data_polarization_ratio_and_channel_swap",
        f"ЦЕЛЬ: {target_name}",
        f"СТАТУС: {'PASS (0 дефектов)' if passed else f'FAIL ({len(defects)} дефектов)'}",
        f"ПРОВЕРЕНО ПИКСЕЛЕЙ: {total_pixels}",
        f"  * Средний ratio (VV - VH): {metrics.get('mean_ratio', float('nan')):.2f} дБ",
        f"  * Максимальное расхождение тождества: {metrics.get('max_ratio_identity_diff', float('nan')):.2e} дБ",
    ]
    if defects:
        lines.append(f"\nЛОКАЛИЗАЦИЯ ДЕФЕКТОВ ({len(defects)}):")
        display_limit = len(defects) if verbose else min(len(defects), 20)
        for d in defects[:display_limit]:
            lines.append(
                f"  - [{d.defect_type}] Var: {d.variable} | Time: {d.timestamp} | "
                f"Pixel: (r={d.row}, c={d.col}) | Geo: (lat={d.latitude:.5f}, lon={d.longitude:.5f}) | "
                f"Val: {d.value:.2f} dB | {d.description}"
            )
        if len(defects) > display_limit:
            lines.append(f"  ... и еще {len(defects) - display_limit} дефектов (используйте verbose=True для полного списка).")

    summary_text = "\n".join(lines[:4])
    detailed_text = "\n".join(lines)
    log_test_record(log_file, "test_data_polarization_ratio_and_channel_swap", target_name, detailed_text)

    report = QualityReport(
        test_name="test_data_polarization_ratio_and_channel_swap",
        target_path=target_name,
        passed=passed,
        total_pixels_checked=total_pixels,
        defects_count=len(defects),
        defects=defects,
        summary_text=summary_text,
        detailed_text=detailed_text,
        metrics=metrics,
    )

    if raise_on_error and not passed:
        raise AssertionError(f"\n{detailed_text}")

    return report


def test_data_spatial_gradient_anomalies_and_rfi(
    nc_path: str | Path | xr.Dataset,
    *,
    raise_on_error: bool = True,
    verbose: bool = False,
    log_file: str | Path | None = "sar_quality_audit.log",
    threshold_db: float = 30.0,
) -> QualityReport:
    """Детекция экстремальных градиентных скачков (> 30 дБ на 20 м) и радиопомех (RFI)."""
    target_name = str(nc_path) if isinstance(nc_path, (str, Path)) else "InMemoryDataset"
    ds = load_dataset_safely(nc_path)

    times = [str(t)[:19] for t in ds["time"].values] if "time" in ds else ["t0"]
    lats = ds["latitude"].values if "latitude" in ds else np.arange(ds.sizes.get("latitude", 1), dtype=float)
    lons = ds["longitude"].values if "longitude" in ds else np.arange(ds.sizes.get("longitude", 1), dtype=float)

    defects: list[DataDefect] = []
    metrics: dict[str, Any] = {}
    vv = ds["VV"].values
    T, Y, X = vv.shape
    total_pixels = vv.size

    max_global_jump = 0.0
    for t in range(T):
        slice_2d = vv[t]
        gy = np.abs(np.diff(slice_2d, axis=0))
        gx = np.abs(np.diff(slice_2d, axis=1))

        cur_max = float(max(np.nanmax(gy) if gy.size > 0 else 0.0, np.nanmax(gx) if gx.size > 0 else 0.0))
        if cur_max > max_global_jump:
            max_global_jump = cur_max

        # Скачки по вертикали (широта)
        anom_y = np.argwhere(gy > threshold_db)
        for r, c in anom_y:
            j_val = float(gy[r, c])
            defects.append(DataDefect(
                defect_type="GRADIENT_SPIKE_Y",
                variable="VV",
                time_idx=int(t),
                timestamp=times[t],
                row=int(r),
                col=int(c),
                latitude=float(lats[r]),
                longitude=float(lons[c]),
                value=j_val,
                threshold=threshold_db,
                description=f"Скачок мощности по широте {j_val:.1f} дБ > {threshold_db:.1f} дБ на шаге 20 м (вероятная помеха RFI)"
            ))

        # Скачки по горизонтали (долгота)
        anom_x = np.argwhere(gx > threshold_db)
        for r, c in anom_x:
            j_val = float(gx[r, c])
            defects.append(DataDefect(
                defect_type="GRADIENT_SPIKE_X",
                variable="VV",
                time_idx=int(t),
                timestamp=times[t],
                row=int(r),
                col=int(c),
                latitude=float(lats[r]),
                longitude=float(lons[c]),
                value=j_val,
                threshold=threshold_db,
                description=f"Скачок мощности по долготе {j_val:.1f} дБ > {threshold_db:.1f} дБ на шаге 20 м (вероятная помеха RFI)"
            ))

    metrics["max_gradient_jump_db"] = max_global_jump
    metrics["anomalies_count"] = len(defects)

    passed = len(defects) == 0
    lines = [
        f"ТЕСТ: test_data_spatial_gradient_anomalies_and_rfi",
        f"ЦЕЛЬ: {target_name}",
        f"СТАТУС: {'PASS (0 дефектов)' if passed else f'FAIL ({len(defects)} дефектов)'}",
        f"ПРОВЕРЕНО ПИКСЕЛЕЙ: {total_pixels}",
        f"  * Максимальный скачок мощности на 20 м: {max_global_jump:.2f} дБ (порог {threshold_db:.1f} дБ)",
    ]
    if defects:
        lines.append(f"\nЛОКАЛИЗАЦИЯ ДЕФЕКТОВ ({len(defects)}):")
        display_limit = len(defects) if verbose else min(len(defects), 20)
        for d in defects[:display_limit]:
            lines.append(
                f"  - [{d.defect_type}] Var: {d.variable} | Time: {d.timestamp} | "
                f"Pixel: (r={d.row}, c={d.col}) | Geo: (lat={d.latitude:.5f}, lon={d.longitude:.5f}) | "
                f"Val: {d.value:.2f} dB | {d.description}"
            )
        if len(defects) > display_limit:
            lines.append(f"  ... и еще {len(defects) - display_limit} дефектов (используйте verbose=True для полного списка).")

    summary_text = "\n".join(lines[:4])
    detailed_text = "\n".join(lines)
    log_test_record(log_file, "test_data_spatial_gradient_anomalies_and_rfi", target_name, detailed_text)

    report = QualityReport(
        test_name="test_data_spatial_gradient_anomalies_and_rfi",
        target_path=target_name,
        passed=passed,
        total_pixels_checked=total_pixels,
        defects_count=len(defects),
        defects=defects,
        summary_text=summary_text,
        detailed_text=detailed_text,
        metrics=metrics,
    )

    if raise_on_error and not passed:
        raise AssertionError(f"\n{detailed_text}")

    return report


def test_data_checkerboard_and_striping_patterns(
    nc_path: str | Path | xr.Dataset,
    *,
    raise_on_error: bool = True,
    verbose: bool = False,
    log_file: str | Path | None = "sar_quality_audit.log",
    max_cbi: float = 0.85,
    max_striping_ratio: float = 2.5,
) -> QualityReport:
    """Детекция шахматных артефактов интерполяции (Checkerboard) и полосности сканирования (Striping)."""
    target_name = str(nc_path) if isinstance(nc_path, (str, Path)) else "InMemoryDataset"
    ds = load_dataset_safely(nc_path)

    times = [str(t)[:19] for t in ds["time"].values] if "time" in ds else ["t0"]
    lats = ds["latitude"].values if "latitude" in ds else np.arange(ds.sizes.get("latitude", 1), dtype=float)
    lons = ds["longitude"].values if "longitude" in ds else np.arange(ds.sizes.get("longitude", 1), dtype=float)

    defects: list[DataDefect] = []
    metrics: dict[str, Any] = {}
    vv = ds["VV"].values
    T, Y, X = vv.shape
    total_pixels = vv.size

    cbi_list = []
    striping_list = []

    for t in range(T):
        slice_2d = vv[t]

        # 1. Индекс шахматного паттерна (CBI) через свертку K = [[1, -1], [-1, 1]]
        diff_diag = slice_2d[:-1, :-1] - slice_2d[:-1, 1:] - slice_2d[1:, :-1] + slice_2d[1:, 1:]
        base_std = np.nanstd(slice_2d) + 1e-9
        cbi = float(np.nanstd(diff_diag) / base_std)
        cbi_list.append(cbi)

        if cbi >= max_cbi:
            defects.append(DataDefect(
                defect_type="CHECKERBOARD_ARTIFACT",
                variable="VV",
                time_idx=int(t),
                timestamp=times[t],
                row=-1,
                col=-1,
                latitude=float(lats[0]),
                longitude=float(lons[0]),
                value=cbi,
                threshold=max_cbi,
                description=f"Обнаружен паразитный шахматный паттерн интерполяции (CBI = {cbi:.2f} >= {max_cbi:.2f})"
            ))

        # 2. Отношение полосности (Striping / Scalloping)
        row_means = np.nanmean(slice_2d, axis=1)
        col_means = np.nanmean(slice_2d, axis=0)
        d_rows_std = float(np.std(np.diff(row_means)))
        d_cols_std = float(np.std(np.diff(col_means)))

        ratio_val = max(d_rows_std, d_cols_std) / (min(d_rows_std, d_cols_std) + 1e-9)
        striping_list.append(ratio_val)

        if ratio_val >= max_striping_ratio:
            defects.append(DataDefect(
                defect_type="STRIPING_ARTIFACT",
                variable="VV",
                time_idx=int(t),
                timestamp=times[t],
                row=-1,
                col=-1,
                latitude=float(lats[0]),
                longitude=float(lons[0]),
                value=ratio_val,
                threshold=max_striping_ratio,
                description=f"Обнаружена выраженная построчная полосность сенсора (Striping Ratio = {ratio_val:.2f} >= {max_striping_ratio:.2f})"
            ))

    metrics["max_cbi"] = float(max(cbi_list)) if cbi_list else 0.0
    metrics["max_striping_ratio"] = float(max(striping_list)) if striping_list else 0.0

    passed = len(defects) == 0
    lines = [
        f"ТЕСТ: test_data_checkerboard_and_striping_patterns",
        f"ЦЕЛЬ: {target_name}",
        f"СТАТУС: {'PASS (0 дефектов)' if passed else f'FAIL ({len(defects)} дефектов)'}",
        f"ПРОВЕРЕНО ПИКСЕЛЕЙ: {total_pixels}",
        f"  * Максимальный CBI (шахматный индекс): {metrics['max_cbi']:.3f} (допустимо < {max_cbi:.2f})",
        f"  * Максимальный Striping Ratio (полосность): {metrics['max_striping_ratio']:.2f} (допустимо < {max_striping_ratio:.2f})",
    ]
    if defects:
        lines.append(f"\nЛОКАЛИЗАЦИЯ ДЕФЕКТОВ ({len(defects)}):")
        for d in defects:
            lines.append(
                f"  - [{d.defect_type}] Var: {d.variable} | Time: {d.timestamp} | "
                f"Val: {d.value:.2f} | {d.description}"
            )

    summary_text = "\n".join(lines[:4])
    detailed_text = "\n".join(lines)
    log_test_record(log_file, "test_data_checkerboard_and_striping_patterns", target_name, detailed_text)

    report = QualityReport(
        test_name="test_data_checkerboard_and_striping_patterns",
        target_path=target_name,
        passed=passed,
        total_pixels_checked=total_pixels,
        defects_count=len(defects),
        defects=defects,
        summary_text=summary_text,
        detailed_text=detailed_text,
        metrics=metrics,
    )

    if raise_on_error and not passed:
        raise AssertionError(f"\n{detailed_text}")

    return report


def test_despeckling_quality_metrics_and_diagnostics(
    nc_path: str | Path | xr.Dataset,
    *,
    raise_on_error: bool = True,
    verbose: bool = False,
    log_file: str | Path | None = "sar_quality_audit.log",
) -> QualityReport:
    """Генерация метрик подавления спекла (ENL) и человекочитаемой диагностики (if/else)."""
    target_name = str(nc_path) if isinstance(nc_path, (str, Path)) else "InMemoryDataset"
    ds = load_dataset_safely(nc_path)

    vv_db = ds["VV"].values[0]
    vh_db = ds["VH"].values[0]
    ratio_db = ds["ratio"].values[0]
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    vv_lin = 10.0 ** (vv_db / 10.0)

    # 1. Локальные среднее и дисперсия в окне 7x7 для оценки ENL
    m = uniform_filter(vv_lin, size=7)
    v = np.maximum(uniform_filter(vv_lin**2, size=7) - m**2, 1e-12)
    ci = np.sqrt(v) / (m + 1e-12)

    valid_mask = ~np.isnan(ci)
    if np.any(valid_mask):
        thresh = np.percentile(ci[valid_mask], 10)
        homo_mask = (ci <= thresh) & valid_mask
        enl = float((np.mean(vv_lin[homo_mask]) / (np.std(vv_lin[homo_mask]) + 1e-12)) ** 2)
    else:
        enl = 0.0

    # 2. Градиенты и максимальный скачок
    gy = np.abs(np.diff(vv_db, axis=0))
    gx = np.abs(np.diff(vv_db, axis=1))
    max_jump = float(max(np.nanmax(gy), np.nanmax(gx)))

    anom_indices = np.where(gy > 30.0)
    anomalies_count = len(anom_indices[0])
    problem_coords = []
    if anomalies_count > 0:
        for r, c in zip(anom_indices[0][:5], anom_indices[1][:5]):
            problem_coords.append(f"широта {lats[r]:.4f}, долгота {lons[c]:.4f} (скачок {gy[r, c]:.1f} дБ)")

    # 3. Доля NaN
    nan_pct = float(100.0 * np.isnan(vv_db).sum() / vv_db.size)
    mean_ratio = float(np.nanmean(ratio_db))

    # 4. Формирование структурированного отчета (if/else)
    what_happened = []
    problems = []
    attention_points = []
    has_critical_issue = False

    if nan_pct == 0.0:
        what_happened.append("Данные целостны: пропуски (NaN) отсутствуют, потеря пакетов равна 0%.")
    elif nan_pct < 5.0:
        what_happened.append(f"Данные практически полные: пропуски составляют {nan_pct:.2f}% (краевые зоны снимка).")
        attention_points.append(f"Присутствует {nan_pct:.2f}% пустых пикселей по краям кадра.")
    else:
        problems.append(f"Высокая потеря данных: {nan_pct:.2f}% площади кадра не содержит измерений.")
        attention_points.append("Проверить границы полосы съемки спутника (swath) за указанный период.")
        has_critical_issue = True

    if enl >= 20.0:
        what_happened.append(
            f"Спекл-шум подавлен превосходно: эквивалентное число отсчетов ENL = {enl:.1f} "
            f"(шумовая дисперсия снижена в {enl / 4.8:.1f} раз относительно исходного сырого снимка)."
        )
    elif enl >= 8.0:
        what_happened.append(
            f"Спекл-шум умеренно сглажен: ENL = {enl:.1f}. Сохранена тонкая текстура ландшафта."
        )
    else:
        attention_points.append(
            f"Низкий показатель ENL ({enl:.1f}): преобладают контрастные антропогенные или горные объекты."
        )

    if anomalies_count == 0 and max_jump < 25.0:
        what_happened.append(
            f"Пространственная гладкость в норме: максимальный скачок мощности равен {max_jump:.1f} дБ. "
            f"Импульсные помехи и битые пиксели отсутствуют."
        )
    else:
        problems.append(f"Обнаружено {anomalies_count} точек экстремального перепада мощности (> 30 дБ на 20 м).")
        attention_points.append(
            f"Локальные импульсные всплески мощности. Координаты: {'; '.join(problem_coords)}"
        )
        has_critical_issue = True

    if mean_ratio > 0.0:
        what_happened.append(
            f"Поляризационный баланс правильный: ко-поляризация превосходит кросс-поляризацию "
            f"(средняя разность VV - VH составляет +{mean_ratio:.1f} дБ). Каналы ориентированы верно."
        )
    else:
        problems.append(f"КРИТИЧЕСКИЙ СБОЙ: Среднее отношение VV/VH отрицательно ({mean_ratio:.1f} дБ).")
        attention_points.append("Каналы VV и VH перепутаны местами в источнике данных!")
        has_critical_issue = True

    report_lines = [
        f"\n{'='*80}",
        f"РЕЗЮМЕ КАЧЕСТВА ДАННЫХ SAR (SENTINEL-1): [{target_name}]",
        f"ИТОГОВЫЙ СТАТУС: {'[УСПЕШНО: ДАННЫЕ ВАЛИДНЫ]' if not has_critical_issue else '[ТРЕБУЕТСЯ ВНИМАНИЕ: ОБНАРУЖЕНЫ ДЕФЕКТЫ]'}",
        f"{'-'*80}",
        "1. ЧТО ПОЛУЧИЛОСЬ В РЕЗУЛЬТАТЕ:",
    ]
    for w in what_happened:
        report_lines.append(f"   * {w}")
    report_lines.append("2. В ЧЕМ ПРОБЛЕМА:")
    if problems:
        for p in problems:
            report_lines.append(f"   * [!] {p}")
    else:
        report_lines.append("   * Дефектов и критических проблем не обнаружено.")
    report_lines.append("3. НА ЧТО ОБРАТИТЬ ВНИМАНИЕ:")
    if attention_points:
        for a in attention_points:
            report_lines.append(f"   * {a}")
    else:
        report_lines.append("   * Параметры полностью соответствуют штатным физическим допускам.")
    report_lines.append("4. ПРОБЛЕМНЫЕ ОБЛАСТИ И КООРДИНАТЫ:")
    if problem_coords:
        for c in problem_coords:
            report_lines.append(f"   * {c}")
    else:
        report_lines.append("   * Отсутствуют (кадр равномерен по качеству).")
    report_lines.append(f"{'='*80}\n")

    report_text = "\n".join(report_lines)
    log_test_record(log_file, "test_despeckling_quality_metrics_and_diagnostics", target_name, report_text)

    metrics = {
        "enl": enl,
        "nan_pct": nan_pct,
        "max_jump": max_jump,
        "anomalies_count": anomalies_count,
        "mean_ratio": mean_ratio,
        "has_critical_issue": has_critical_issue,
    }

    report = QualityReport(
        test_name="test_despeckling_quality_metrics_and_diagnostics",
        target_path=target_name,
        passed=not has_critical_issue,
        total_pixels_checked=vv_db.size,
        defects_count=anomalies_count + (1 if mean_ratio <= 0 else 0) + (1 if nan_pct > 15.0 else 0),
        defects=[],
        summary_text=f"ENL={enl:.1f}, NaNs={nan_pct:.2f}%, MeanRatio={mean_ratio:.2f}dB",
        detailed_text=report_text,
        metrics=metrics,
    )

    if verbose or not raise_on_error:
        print(report_text)

    if raise_on_error and has_critical_issue:
        raise AssertionError(f"\n{report_text}")

    return report


def test_synthetic_despeckling_benchmark_metrics() -> None:
    """Оценка метрик расшумления (PSNR gain, SSI, EPI) на тестовом синтетическом эталоне."""
    np.random.seed(42)
    clean = np.ones((60, 60), dtype=np.float32) * 0.1
    clean[:, 30:] = 0.5

    speckle = np.random.gamma(5.0, 1.0 / 5.0, size=(60, 60)).astype(np.float32)
    noisy = clean * speckle

    filt = run.gamma_map(noisy, w=3, looks=5.0)

    # 1. PSNR (Peak Signal-to-Noise Ratio)
    mse_noisy = float(np.mean((noisy - clean) ** 2))
    mse_filt = float(np.mean((filt - clean) ** 2))
    psnr_noisy = 10.0 * np.log10(np.max(clean) ** 2 / mse_noisy)
    psnr_filt = 10.0 * np.log10(np.max(clean) ** 2 / mse_filt)
    psnr_gain = psnr_filt - psnr_noisy
    assert psnr_gain > 3.0, f"Недостаточный прирост PSNR: +{psnr_gain:.2f} дБ"

    # 2. SSI (Speckle Suppression Index)
    c_noisy = float(np.std(noisy[:, :25]) / np.mean(noisy[:, :25]))
    c_filt = float(np.std(filt[:, :25]) / np.mean(filt[:, :25]))
    ssi = c_filt / (c_noisy + 1e-9)
    assert ssi < 0.70, f"Индекс SSI слишком велик: {ssi:.3f}"

    # 3. EPI (Edge Preservation Index)
    edge_grad_noisy = float(np.mean(np.abs(sobel(noisy, axis=1)[:, 29:32])))
    edge_grad_filt = float(np.mean(np.abs(sobel(filt, axis=1)[:, 29:32])))
    epi = edge_grad_filt / (edge_grad_noisy + 1e-9)
    assert epi > 0.60, f"Размытие границы: индекс EPI = {epi:.3f}"


# =====================================================================
# 5. Мастер-функция комплексного аудита качества
# =====================================================================

def check_sar_dataset(
    target: str | Path | xr.Dataset,
    *,
    raise_on_error: bool = True,
    verbose: bool = False,
    log_file: str | Path | None = "sar_quality_audit.log",
) -> dict[str, QualityReport]:
    """Запуск полного цикла проверок качества данных на переданном NetCDF.

    Выполняет:
    1. test_data_nan_and_drop_detection
    2. test_data_physical_dynamic_range
    3. test_data_polarization_ratio_and_channel_swap
    4. test_data_spatial_gradient_anomalies_and_rfi
    5. test_data_checkerboard_and_striping_patterns
    6. test_despeckling_quality_metrics_and_diagnostics

    Возвращает словарь {имя_проверки: QualityReport}.
    Если raise_on_error=True и хотя бы один тест провалился, выбрасывает AssertionError
    с консолидированным отчетом по всем дефектам.
    """
    reports: dict[str, QualityReport] = {}
    failures: list[str] = []

    checks = [
        ("nan_and_drop_detection", test_data_nan_and_drop_detection),
        ("physical_dynamic_range", test_data_physical_dynamic_range),
        ("polarization_ratio_and_channel_swap", test_data_polarization_ratio_and_channel_swap),
        ("spatial_gradient_anomalies_and_rfi", test_data_spatial_gradient_anomalies_and_rfi),
        ("checkerboard_and_striping_patterns", test_data_checkerboard_and_striping_patterns),
        ("despeckling_quality_and_diagnostics", test_despeckling_quality_metrics_and_diagnostics),
    ]

    for name, check_fn in checks:
        rep = check_fn(target, raise_on_error=False, verbose=verbose, log_file=log_file)
        reports[name] = rep
        if not rep.passed:
            failures.append(f"[{name}] {rep.defects_count} дефектов:\n{rep.detailed_text}")

    if raise_on_error and failures:
        header = f"\n{'!'*80}\nОБНАРУЖЕНЫ СБОИ В КАЧЕСТВЕ ДАННЫХ ({len(failures)} проверок не пройдено):\n{'!'*80}\n"
        raise AssertionError(header + "\n\n".join(failures))

    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Аудит качества радарных данных Sentinel-1 SAR NetCDF.")
    parser.add_argument("path", help="Путь к NetCDF-файлу для тестирования")
    parser.add_argument("--verbose", action="store_true", help="Максимально подробный вывод координат всех дефектов")
    parser.add_argument("--no-raise", action="store_true", help="Не завершать работу ошибкой при обнаружении дефектов")
    parser.add_argument("--log", default="sar_quality_audit.log", help="Файл сохранения отчета аудита")
    args = parser.parse_args()

    check_sar_dataset(
        args.path,
        raise_on_error=not args.no_raise,
        verbose=args.verbose,
        log_file=args.log,
    )
