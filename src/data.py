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
    "cloud_cover", "temperature", "humidity", "azimuth",
]
TARGET_COL = "TCI"


def _sun_altitude_pool():
    """בריכת גבהי שמש אמיתיים לת"א — כל שעה x 12 חודשים, מעל 10° בלבד."""
    pool = []
    for mo in range(1, 13):
        for hr in range(0, 24):
            try:
                dt = datetime(2024, mo, 15, hr, tzinfo=timezone.utc)
                alt = _solar.get_altitude(_TLV_LAT, _TLV_LON, dt)
                if alt > 10:  # מתחת ל-10° השמש לא "מכה" — זניח לנוחות תרמית
                    pool.append(float(alt))
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

    # מאפייני רחוב (קבועים לכל קשת) — דגימת אינדקס אחד לשמירת הצמד גובה+חופה
    ef = pd.read_parquet(
        EDGES_PATH, columns=["mean_building_height", "tree_canopy_ratio"]
    ).fillna(0)
    idx = rng.choice(len(ef), size=n, replace=True)
    bh = ef["mean_building_height"].values[idx]
    cr = ef["tree_canopy_ratio"].values[idx]

    # גובה שמש — PySolar, עם fallback לערכים חודשיים אופייניים
    if _PYSOLAR:
        pool = _sun_altitude_pool()
        sa = rng.choice(pool, size=n, replace=True) if pool else rng.uniform(5, 72, n)
    else:
        pre_baked = [18.1, 31.4, 44.2, 55.3, 64.1, 70.2, 72.1, 70.1, 64.0, 55.1, 44.0]
        sa = rng.choice(pre_baked, size=n, replace=True)

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

    az = rng.uniform(0, 360, n)

    # חישוב TCI מהנוסחה האנליטית (זהה ל-app.py)
    w1, w2 = 0.6, 0.4
    bf = np.clip(bh / 30, 0, 1) * np.cos(np.radians(sa))
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
        "azimuth":         az,
        "TCI":             tci,
    })
