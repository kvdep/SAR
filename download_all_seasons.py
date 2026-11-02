import os
import sys
import json
import urllib.request
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SCENES = {
    "benevento": [14.70, 41.10, 14.80, 41.20],
    "rome": [12.45, 41.85, 12.55, 41.95],
    "serre_poncon": [6.30, 44.50, 6.40, 44.60],
    "forest": [40.10, 55.45, 40.20, 55.55],
    "city": [37.55, 55.70, 37.65, 55.80],
    "water": [39.50, 43.40, 39.60, 43.50],
    "desert": [-4.05, 31.10, -3.95, 31.20],
    "mountain": [42.40, 43.30, 42.50, 43.40],
}

SEASONS = {
    "winter": ("2023-01-05T00:00:00Z", "2023-01-31T23:59:59Z", "15 января (Зима)"),
    "spring": ("2023-04-05T00:00:00Z", "2023-04-30T23:59:59Z", "15 апреля (Весна)"),
    "summer": ("2023-07-05T00:00:00Z", "2023-07-31T23:59:59Z", "15 июля (Лето)"),
    "autumn": ("2023-10-05T00:00:00Z", "2023-10-31T23:59:59Z", "15 октября (Осень)"),
}

SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"

def get_sas_token() -> str:
    req = urllib.request.urlopen(TOKEN_URL, timeout=10)
    data = json.loads(req.read().decode("utf-8"))
    return data["token"]

def main():
    print("=" * 70)
    print("ЗАГРУЗКА РЕАЛЬНЫХ СЕЗОННЫХ СРЕЗОВ SENTINEL-1 RTC (ВЕСНА, ЛЕТО, ОСЕНЬ, ЗИМА)")
    print("=" * 70)

    token = get_sas_token()
    print("[ИНФОРМАЦИЯ] SAS-токен Planetary Computer успешно получен.")

    for sc_key, bbox in SCENES.items():
        out_file = os.path.join(CACHE_DIR, f"seasons_{sc_key}.npz")
        if os.path.exists(out_file):
            print(f"[ПРОПУСК] {sc_key}: файл {out_file} уже существует.")
            continue

        print(f"\n[СЦЕНА: {sc_key.upper()}] Координаты bbox: {bbox}")
        scene_data = {"bbox": np.array(bbox, dtype=np.float64)}
        target_shape = None

        for s_key, (t_start, t_end, s_label) in SEASONS.items():
            query = {
                "collections": ["sentinel-1-rtc"],
                "bbox": bbox,
                "datetime": f"{t_start}/{t_end}",
                "limit": 1
            }
            req = urllib.request.Request(
                SEARCH_URL,
                data=json.dumps(query).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            try:
                resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))
                features = resp.get("features", [])
                if not features:
                    print(f"  [ПРЕДУПРЕЖДЕНИЕ] {s_key}: снимки не найдены в окне {t_start[:10]} - {t_end[:10]}")
                    continue

                item = features[0]
                dt = item["properties"].get("datetime", "")[:10]
                vv_href = item["assets"]["vv"]["href"]
                vh_href = item["assets"]["vh"]["href"]

                with rasterio.open(f"{vv_href}?{token}") as src_vv, rasterio.open(f"{vh_href}?{token}") as src_vh:
                    tb = transform_bounds("EPSG:4326", src_vv.crs, *bbox)
                    win = from_bounds(*tb, src_vv.transform)
                    vv = src_vv.read(1, window=win).astype(np.float32)
                    vh = src_vh.read(1, window=win).astype(np.float32)

                # Замена NaNs и отрицательных значений
                vv = np.nan_to_num(vv, nan=0.01)
                vh = np.nan_to_num(vh, nan=0.002)
                vv = np.maximum(vv, 1e-6)
                vh = np.maximum(vh, 1e-6)

                if target_shape is None:
                    target_shape = vv.shape
                else:
                    if vv.shape != target_shape:
                        import cv2
                        vv = cv2.resize(vv, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                        vh = cv2.resize(vh, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

                scene_data[f"{s_key}_vv"] = vv
                scene_data[f"{s_key}_vh"] = vh
                scene_data[f"{s_key}_date"] = dt
                print(f"  -> {s_key:6s} ({s_label}): дата={dt}, форма={vv.shape}, ср. VV={vv.mean():.4f}, ср. VH={vh.mean():.4f}")

            except Exception as exc:
                print(f"  [ОШИБКА] {sc_key} {s_key}: {exc}")

        # Проверка полноты
        has_all = all(f"{s}_vv" in scene_data for s in ["winter", "spring", "summer", "autumn"])
        if has_all:
            np.savez_compressed(out_file, **scene_data)
            print(f"[СОХРАНЕНО] {out_file} (все 4 сезона)")
        else:
            print(f"[НЕ ПОЛНОСТЬЮ] {sc_key}: собрано {len(scene_data)} полей")

if __name__ == "__main__":
    main()
