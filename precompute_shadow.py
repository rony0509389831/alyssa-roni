# -*- coding: utf-8 -*-
"""
precompute_shadow.py — חישוב מראש של כיסוי-צל מבנים לכל קשת, פר שעה.

למה offline: כל שעה דורשת בניית ~15K מצולעי צל + חיתוך מול ~10K קשתות (~12ש').
מריצים פעם אחת, שומרים טבלה קטנה, והאפליקציה רק קוראת lookup (מיידי).

שיטה (ר' דיון M3+): כל מבנה מקבל ריבוע footprint של 20מ' בגובהו; מצולע הצל =
הריבוע + העתק נגרר במרחק h/tan(alt) בכיוון ההפוך לשמש. כיסוי קשת = אורך החיתוך
של הקשת עם איחוד הצללים, חלקי אורך הקשת. הכיוון נקבע ע"י השמש → מתקן את עצמו
לפי שעה (רחוב מקביל לשמש יוצא מואר, ניצב יוצא מוצל), בלי סף מלאכותי.

פלט: data/shadow_coverage.parquet  (u,v,key + עמודה לכל שעה "6.0".."19.0")
הרצה: python precompute_shadow.py
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, geopandas as gpd
from datetime import datetime, timezone
from pysolar import solar
from shapely.geometry import box
from shapely import affinity, STRtree
from shapely.ops import unary_union

SQUARE_HALF = 10.0                 # ריבוע 20מ'
REF_DATE = (2026, 6, 27)           # תאריך ייחוס קיץ — זהה לזה שב-app.py (גרף 4)
HOURS = [round(6.0 + 0.5 * i, 1) for i in range(27)]   # 6.0 .. 19.0
OUT = "data/shadow_coverage.parquet"
LAT, LON = 32.08, 34.77


def _load():
    # כל קשתות העיר (לא רק רוטשילד) — נדרש לניווט עירוני אחיד
    ef = gpd.read_parquet("data/edges_features.parquet").to_crs("EPSG:2039").copy()
    b = pd.read_csv("data/buildings_clean.csv")
    b = b[b.height.notna() & (b.height > 0)]   # כל המבנים בעיר
    g = gpd.GeoDataFrame(b[["height"]], geometry=gpd.points_from_xy(b.lon, b.lat),
                         crs="EPSG:4326").to_crs("EPSG:2039")
    return ef, g.geometry.x.values, g.geometry.y.values, g["height"].values


def _coverage(ef, BX, BY, BH, hour):
    h = int(hour); mn = int(round((hour - h) * 60))
    dt = datetime(*REF_DATE, h - 3, mn, tzinfo=timezone.utc)   # IDT=UTC+3
    alt = float(solar.get_altitude(LAT, LON, dt))
    az = float(solar.get_azimuth(LAT, LON, dt))
    if alt <= 3.0:                       # שמש מתחת/סמוך לאופק → הכל מוצל
        return np.ones(len(ef)), alt, az
    s = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
    tan_a = np.tan(np.radians(alt))
    dx = -s[0] * BH / tan_a; dy = -s[1] * BH / tan_a
    shadows = []
    for i in range(len(BX)):
        b0 = box(BX[i]-SQUARE_HALF, BY[i]-SQUARE_HALF, BX[i]+SQUARE_HALF, BY[i]+SQUARE_HALF)
        shadows.append(b0.union(affinity.translate(b0, dx[i], dy[i])).convex_hull)
    tree = STRtree(shadows)
    geoms = ef.geometry.values
    cov = np.zeros(len(geoms))
    for i, ln in enumerate(geoms):
        if ln.length <= 0:
            continue
        idx = tree.query(ln)
        if len(idx) == 0:
            continue
        hit = [ln.intersection(shadows[j]) for j in idx]
        hit = [x for x in hit if not x.is_empty]
        if hit:
            cov[i] = min(unary_union(hit).length / ln.length, 1.0)
    return cov, alt, az


def main():
    ef, BX, BY, BH = _load()
    print(f"קשתות={len(ef):,}  מבנים={len(BX):,}  שעות={len(HOURS)}", flush=True)
    out = ef[["u", "v", "key"]].copy()
    for hr in HOURS:
        t0 = time.time()
        cov, alt, az = _coverage(ef, BX, BY, BH, hr)
        out[f"{hr:.1f}"] = cov.round(3)
        print(f"  {hr:4.1f}h  alt={alt:5.1f}° az={az:6.1f}°  cov_avg={cov.mean()*100:4.1f}%  ({time.time()-t0:.1f}s)", flush=True)
    out.to_parquet(OUT, index=False)
    print(f"\nנשמר {OUT}  ({len(out):,} קשתות × {len(HOURS)} שעות)", flush=True)


if __name__ == "__main__":
    main()
