"""
בניית טבלת אימון ל-TCI (features + target) לשימוש חוזר מחוץ ל-Streamlit.

מקור הפיצ'רים המרחביים: data/edges_features.parquet (נוצר ב-precompute_features.py).
מקור הצללת המבנים: data/shadow_coverage.parquet (נוצר ב-precompute_shadow.py) —
כיסוי-צל אמיתי פר קשת פר שעה (מצולעי צל מ-footprint), במקום הקירוב הישן
building_height × sin(shadow_angle). זהו אות הצל היחיד בכל המערכת (אנליטי/ML/ניווט).

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
SHADOW_PATH = Path("data/shadow_coverage.parquet")
CLIMATE_PATH = Path("data/climate_fallback.json")

# שעות הייחוס + התאריך — חייבים להיות זהים ל-precompute_shadow.py
REF_DATE = (2026, 6, 27)
HOURS = [round(6.0 + 0.5 * i, 1) for i in range(27)]   # 6.0 .. 19.0

# 7 הפיצ'רים שמנבאים TCI + עמודת היעד.
# shadow_cov החליף את shadow_angle: הוא כבר מקפל בתוכו את כיוון השמש מול הרחוב,
# גובה המבנה המצל וטווח הצל — מצולע צל אמיתי במקום זווית בלבד.
FEATURE_COLS = [
    "sun_altitude", "building_height", "canopy_ratio",
    "cloud_cover", "temperature", "humidity", "shadow_cov",
]
TARGET_COL = "TCI"


def _altitude_by_hour():
    """גובה השמש לכל שעת ייחוס (זהה ל-precompute_shadow.py). fallback סטטי בלי PySolar."""
    if _PYSOLAR:
        out = []
        for h in HOURS:
            hh = int(h); mn = int(round((h - hh) * 60))
            dt = datetime(*REF_DATE, hh - 3, mn, tzinfo=timezone.utc)  # IDT=UTC+3
            out.append(max(float(_solar.get_altitude(_TLV_LAT, _TLV_LON, dt)), 0.0))
        return np.array(out)
    # קירוב לקשת קיץ אם אין PySolar
    return np.clip(-0.9 * (np.array(HOURS) - 12.75) ** 2 + 82, 0, 82)


def build_tci_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    בונה n דוגמאות אימון: כל שורה = מקטע רחוב אמיתי × שעת-ייחוס × מזג-אוויר אקראי.

    מחזיר DataFrame עם 7 פיצ'רים + עמודת TCI (היעד הרציף).
    דורש את data/edges_features.parquet ו-data/shadow_coverage.parquet.
    """
    rng = np.random.default_rng(seed)

    if not EDGES_PATH.exists():
        raise FileNotFoundError(f"{EDGES_PATH} לא נמצא — הרץ: python precompute_features.py")
    if not SHADOW_PATH.exists():
        raise FileNotFoundError(f"{SHADOW_PATH} לא נמצא — הרץ: python precompute_shadow.py")

    # מאפייני רחוב (סטטיים) + כיסוי-צל (פר שעה), ממוזגים לפי (u,v,key)
    ef = pd.read_parquet(
        EDGES_PATH, columns=["u", "v", "key", "mean_building_height", "tree_canopy_ratio"]
    ).fillna({"mean_building_height": 0.0, "tree_canopy_ratio": 0.0})
    cov = pd.read_parquet(SHADOW_PATH)
    m = ef.merge(cov, on=["u", "v", "key"], how="inner")

    hour_cols = [f"{h:.1f}" for h in HOURS]
    covmat = m[hour_cols].fillna(0.0).to_numpy()        # (n_edges, 27)
    bh_all = m["mean_building_height"].to_numpy()
    cr_all = m["tree_canopy_ratio"].to_numpy()
    alt_by_hour = _altitude_by_hour()                   # (27,)

    # דגימה: כל שורה = קשת אקראית × שעת-ייחוס אקראית
    esel = rng.integers(0, len(m), size=n)
    hsel = rng.integers(0, len(HOURS), size=n)
    cov_s = covmat[esel, hsel]
    bh    = bh_all[esel]
    cr    = cr_all[esel]
    sa    = alt_by_hour[hsel]

    # מזג אוויר — 12 ערכים חודשיים אמיתיים, עם fallback
    try:
        with open(CLIMATE_PATH, encoding="utf-8") as f:
            clim = json.load(f)
        temps = np.array([mo["temperature"] for mo in clim])
        clouds = np.array([mo["cloud_cover"] for mo in clim])
        humids = np.array([mo.get("humidity", 70) for mo in clim])
        widx = rng.integers(0, len(clim), size=n)
        temp, cloud, humid = temps[widx], clouds[widx], humids[widx]
    except Exception:
        temp = rng.uniform(13, 28, n)
        cloud = rng.uniform(0, 50, n)
        humid = rng.uniform(65, 80, n)

    # חישוב TCI מהנוסחה האנליטית (זהה ל-app.py, גרף 4): הצל = shadow_cov ישירות
    w1, w2 = 0.6, 0.4
    tci = np.clip(
        1 + 9 * (sa / 80) * (1 - cloud / 100) * (1 - w1 * cr - w2 * cov_s),
        1, 10,
    )

    return pd.DataFrame({
        "sun_altitude":    sa,
        "building_height": bh,
        "canopy_ratio":    cr,
        "cloud_cover":     cloud,
        "temperature":     temp,
        "humidity":        humid,
        "shadow_cov":      cov_s,
        "TCI":             tci,
    })
