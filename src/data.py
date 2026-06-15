"""
בניית טבלת אימון ל-TCI (features + target) לשימוש חוזר מחוץ ל-Streamlit.

מקור הפיצ'רים המרחביים: data/edges_features.parquet (נוצר ב-precompute_features.py).
הנוסחה והדגימה זהות ל-build_tci_df שב-app.py — אך ללא תלות ב-Streamlit, כדי
ש-model.py יוכל לייבא מכאן בלי להריץ את כל האפליקציה.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pysolar.solar as _solar
    _PYSOLAR = True
except ImportError:
    _PYSOLAR = False

# קואורדינטות ת"א לחישוב גובה שמש
_TLV_LAT, _TLV_LON = 32.08, 34.77

EDGES_PATH = Path("data/edges_features.parquet")
CLIMATE_PATH = Path("data/climate_fallback.json")

# 7 הפיצ'רים שמנבאים TCI + עמודת היעד
FEATURE_COLS = [
    "sun_altitude", "building_height", "canopy_ratio",
    "cloud_cover", "temperature", "humidity", "shadow_angle",
]
TARGET_COL = "TCI"


def _sun_position_pool():
    """בריכת זוויות שמש אמיתיות לת"א — (altitude, azimuth) לכל שעה × 12 חודשים, מעל 10°."""
    pool = []
    for mo in range(1, 13):
        for hr in range(0, 24):
            try:
                dt = datetime(2024, mo, 15, hr, tzinfo=timezone.utc)
                alt = _solar.get_altitude(_TLV_LAT, _TLV_LON, dt)
                if alt > 10:
                    az = _solar.get_azimuth(_TLV_LAT, _TLV_LON, dt)
                    pool.append((float(alt), float(az)))
            except Exception:
                pass
    return pool


def build_tci_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    בונה n דוגמאות אימון: כל שורה = מקטע רחוב אמיתי x תנאי שמש/מזג-אוויר אקראיים.

    מחזיר DataFrame עם 7 פיצ'רים + עמודת TCI (היעד הרציף).
    דורש את data/edges_features.parquet (הרץ precompute_features.py אם חסר).
    """
    rng = np.random.default_rng(seed)

    if not EDGES_PATH.exists():
        raise FileNotFoundError(
            f"{EDGES_PATH} לא נמצא — הרץ קודם: python precompute_features.py"
        )

    # מאפייני רחוב (קבועים לכל קשת) — דגימת אינדקס אחד לשמירת הצמד גובה+חופה+כיוון
    ef = pd.read_parquet(
        EDGES_PATH, columns=["mean_building_height", "tree_canopy_ratio", "street_azimuth"]
    ).fillna(0)
    idx = rng.choice(len(ef), size=n, replace=True)
    bh = ef["mean_building_height"].values[idx]
    cr = ef["tree_canopy_ratio"].values[idx]
    street_az = ef["street_azimuth"].values[idx]  # 0°–180°, כיוון הרחוב

    # מיקום שמש — PySolar מחזיר זוגות (altitude, azimuth) לת"א
    if _PYSOLAR:
        pool = _sun_position_pool()
        if pool:
            chosen = rng.integers(0, len(pool), size=n)
            sa      = np.array([pool[i][0] for i in chosen])
            sun_az  = np.array([pool[i][1] for i in chosen])
        else:
            sa     = rng.uniform(5, 72, n)
            sun_az = rng.uniform(0, 360, n)
    else:
        # ערכים אופייניים לת"א (altitude, azimuth) ללא PySolar
        pre_baked = [(18.1,140.0),(31.4,155.0),(44.2,165.0),(55.3,175.0),
                     (64.1,185.0),(70.2,195.0),(72.1,180.0)]
        chosen = rng.integers(0, len(pre_baked), size=n)
        sa     = np.array([pre_baked[i][0] for i in chosen])
        sun_az = np.array([pre_baked[i][1] for i in chosen])

    # מזג אוויר — 12 ערכים חודשיים אמיתיים, עם fallback
    try:
        with open(CLIMATE_PATH, encoding="utf-8") as f:
            clim = json.load(f)
        temps = np.array([m["temperature"] for m in clim])
        clouds = np.array([m["cloud_cover"] for m in clim])
        humids = np.array([m.get("humidity", 70) for m in clim])
        widx = rng.integers(0, len(clim), size=n)
        temp, cloud, humid = temps[widx], clouds[widx], humids[widx]
    except Exception:
        temp = rng.uniform(13, 28, n)
        cloud = rng.uniform(0, 50, n)
        humid = rng.uniform(65, 80, n)

    # shadow_angle: זווית בין כיוון השמש לכיוון הרחוב, 0°–90°
    # 0° = שמש מקבילה לרחוב (צל לאורך הרחוב, לא מעבר) → shadow_factor=0
    # 90° = שמש ניצבת לרחוב (צל חוצה את הרחוב) → shadow_factor=1
    sun_dir = sun_az % 180          # 0°–180° (כיוון בידירקציונלי)
    diff = np.abs(sun_dir - street_az)
    shadow_angle = np.minimum(diff, 180 - diff)   # 0°–90°

    # חישוב TCI מהנוסחה האנליטית (זהה ל-app.py)
    w1, w2 = 0.6, 0.4
    shadow_factor = np.sin(np.radians(shadow_angle))
    bf = np.clip(bh / 30, 0, 1) * np.cos(np.radians(sa)) * shadow_factor
    tci = np.clip(
        1 + 9 * (sa / 80) * (1 - cloud / 100) * (1 - w1 * cr - w2 * bf),
        1, 10,
    )

    return pd.DataFrame({
        "sun_altitude":    sa,
        "building_height": bh,
        "canopy_ratio":    cr,
        "cloud_cover":     cloud,
        "temperature":     temp,
        "humidity":        humid,
        "shadow_angle":    shadow_angle,
        "TCI":             tci,
    })
