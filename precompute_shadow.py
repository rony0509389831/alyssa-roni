# -*- coding: utf-8 -*-
"""
precompute_shadow.py — חישוב מראש של כיסוי-צל **מאוחד** (מבנים+עצים) לכל קשת, פר שעה.

למה offline: כל שעה דורשת בניית מצולע-צל לכל אחד מ-45,783 המבנים + עצי-חופה
רלוונטיים (מתוך 230,729) + חיתוך מול ~59K קשתות. מריצים פעם אחת, שומרים טבלה
קטנה, והאפליקציה רק קוראת lookup (מיידי).

שיטה (שלב 1+2 — ר' CLAUDE.md/WORKLOG.md, שיפור אלגוריתם 2026-07-10):
- **מבנים (שלב 1):** footprint מרובע ש**גודלו** נגזר מהצפיפות המקומית (ממוצע מרחק
  ל-5 השכנים הקרובים ביותר, clip [5,20]מ' — במקום ריבוע קבוע 20×20מ' לכולם)
  ו**כיוונו** מיושר לרחוב הקרוב ביותר (street_azimuth, במקום ריבוע ישר-צירים).
- **עצים (שלב 2):** באותה שיטה בדיוק — מצולע חופת העץ האמיתי (לא ריבוע!) נגרר
  לפי גובה-עץ **משוער** (אין שדה גובה אמיתי בנתונים — טבלת buckets לפי `area_class`,
  הנחת-מודל לא-נמדדת מאותה משפחה כמו imputation גובה המבנים). לפני הגרירה, מסננים
  עצים רחוקים-מדי מכל קשת (סינון אדפטיבי, תלוי-שעה — ר' `_TREE_HEIGHT_MAX`/tan(alt))
  כדי לא לגרור אלפי עצים שבוודאות לא יכולים להגיע לשום רחוב באותה שעה.
- **איחוד:** מצולעי-הצל של מבנים ועצים לשעה זו נכנסים יחד לאותו STRtree (immutable —
  נבנה פעם אחת מרשימה שלמה), כך ש-`unary_union` על חיתוכי-הקשת פותר double-counting
  בין חפיפת עץ/מבנה על אותה קשת — בלי נוסחת-שילוב נפרדת.
- מצולע הצל של כל אובייקט = הצורה המקורית + העתק נגרר במרחק h/tan(alt) בכיוון
  ההפוך לשמש. כיסוי קשת = אורך החיתוך של הקשת עם איחוד כל הצללים, חלקי אורך הקשת.
  הכיוון נקבע ע"י השמש → מתקן את עצמו לפי שעה, בלי סף מלאכותי.

פלט: data/shadow_coverage.parquet  (u,v,key + עמודה לכל שעה "6.0".."19.0")
הרצה: python precompute_shadow.py
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, geopandas as gpd
import shapely
from datetime import datetime, timezone
from pysolar import solar
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely import affinity, STRtree
from shapely.ops import unary_union

from src.spatial import _load_trees
from src.data import REF_DATE, HOURS   # תאריך-ייחוס + שעות — מקור אמת יחיד, לא לשכפל

OUT = "data/shadow_coverage.parquet"
LAT, LON = 32.08, 34.77

# footprint דינמי למבנים (שלב 1) — במקום SQUARE_HALF=10.0 קבוע לכולם:
_HALF_MIN, _HALF_MAX = 5.0, 20.0   # מטר — רצפה/תקרה לגודל footprint
_NN_K = 5                          # ממוצע מרחק ל-5 שכנים קרובים ביותר (יציב יותר מ-1-NN גולמי)
_DENSITY_SCALE = 0.4               # half_width = clip(0.4 * nn5_avg_dist, 5, 20)

# גובה משוער לעצים לפי area_class (שלב 2) — אין שדה גובה אמיתי ב-national_canopy;
# הנחת-מודל לא-נמדדת, מאותה משפחה כמו imputation גובה המבנים (2.95*floors+5.21).
# מפתחות = תחילית area_class לפני " m" (הערך הגולמי מכיל תו replacement שבור, למשל
# "10-25 m�" — לכן משווים על התחילית הנקייה בלבד, לא על המחרוזת המלאה).
_TREE_HEIGHT_BY_CLASS = {
    "0-1": 3.0, "1-5": 4.5, "5-10": 6.0, "10-25": 8.0,
    "25-50": 10.0, "50-100": 12.0, "100+": 14.0,
}
_TREE_HEIGHT_FALLBACK = 8.0                            # אמצע הטבלה, ל-area_class חסר/לא-מוכר
_TREE_HEIGHT_MAX = max(_TREE_HEIGHT_BY_CLASS.values())  # 14.0 — לסינון-הרלוונטיות האדפטיבי
_TREE_FILTER_MARGIN_M = 15.0                            # מרווח-ביטחון: רדיוס-חופה + רוחב-buffer


def _tree_heights(area_class: pd.Series) -> np.ndarray:
    """ממפה area_class (עם/בלי תו-replacement שבור) לגובה-עץ משוער במטרים."""
    prefix = area_class.astype(str).str.split(" m", n=1).str[0]
    return prefix.map(_TREE_HEIGHT_BY_CLASS).fillna(_TREE_HEIGHT_FALLBACK).to_numpy()


def _load():
    # כל קשתות העיר (לא רק רוטשילד) — נדרש לניווט עירוני אחיד
    ef = gpd.read_parquet("data/edges_features.parquet").to_crs("EPSG:2039").copy()
    etree = STRtree(ef.geometry.values)

    # ── מבנים ──────────────────────────────────────────────────────────────
    b = pd.read_csv("data/buildings_clean.csv")
    b = b[b.height.notna() & (b.height > 0)]   # כל המבנים בעיר
    g = gpd.GeoDataFrame(b[["height"]], geometry=gpd.points_from_xy(b.lon, b.lat),
                         crs="EPSG:4326").to_crs("EPSG:2039")
    BX, BY, BH = g.geometry.x.values, g.geometry.y.values, g["height"].values

    # שלב 1א — גודל footprint לפי צפיפות מקומית: ממוצע מרחק ל-5 השכנים הקרובים
    # ביותר (יציב מול 1-NN גולמי, שסובל מזוגות-צנטרואידים כמעט-זהים ~0-1מ').
    coords = np.column_stack([BX, BY])
    dist, _idx = cKDTree(coords).query(coords, k=_NN_K + 1)   # +1: כולל את הנקודה עצמה (מרחק 0)
    nn_avg = dist[:, 1:].mean(axis=1)
    half_width = np.clip(_DENSITY_SCALE * nn_avg, _HALF_MIN, _HALF_MAX)

    # שלב 1ב — כיוון footprint לפי הרחוב הקרוב ביותר (street_azimuth הקיים כבר
    # ב-edges_features.parquet). שני הצדדים כבר ב-EPSG:2039 כאן — קריטי לחיפוש
    # nearest מרחבי-נכון (לא ב-EPSG:4326 שבו הקובץ נשמר על הדיסק).
    b_nearest_idx = etree.nearest(g.geometry.values)
    street_az = ef["street_azimuth"].to_numpy()[b_nearest_idx]   # bearing-מצפן, 0-180 מעלות

    # ── עצים (שלב 2) ──────────────────────────────────────────────────────
    trees = _load_trees()   # geometry (פוליגון אמיתי) + canopy_area_m2 + area_class, EPSG:2039
    tree_geoms = trees.geometry.to_numpy()
    tree_heights = _tree_heights(trees["area_class"])
    # חישוב חד-פעמי (לא בתוך הלולאה השעתית) — מרחק כל עץ לרחוב הקרוב ביותר, לסינון
    # אדפטיבי-לשעה בהמשך (עצים לא זזים, רק השמש; המרחק לא צריך להיגזר מחדש כל שעה).
    t_centroids = shapely.centroid(tree_geoms)
    t_nearest_idx = etree.nearest(t_centroids)
    tree_edge_dist = shapely.distance(ef.geometry.values[t_nearest_idx], t_centroids)

    return ef, BX, BY, BH, half_width, street_az, tree_geoms, tree_heights, tree_edge_dist


def _coverage(ef, BX, BY, BH, half_width, street_az,
             tree_geoms, tree_heights, tree_edge_dist, hour):
    h = int(hour); mn = int(round((hour - h) * 60))
    dt = datetime(*REF_DATE, h - 3, mn, tzinfo=timezone.utc)   # IDT=UTC+3
    alt = float(solar.get_altitude(LAT, LON, dt))
    az = float(solar.get_azimuth(LAT, LON, dt))
    if alt <= 3.0:                       # שמש מתחת/סמוך לאופק → הכל מוצל
        return np.ones(len(ef)), alt, az
    s = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
    tan_a = np.tan(np.radians(alt))

    # ── מבנים (שלב 1) ──────────────────────────────────────────────────────
    dx_b = -s[0] * BH / tan_a; dy_b = -s[1] * BH / tan_a
    shadows = []
    for i in range(len(BX)):
        hw = half_width[i]
        b0 = box(BX[i]-hw, BY[i]-hw, BX[i]+hw, BY[i]+hw)
        # street_azimuth הוא bearing-מצפן (0°=צפון-דרום, 90°=מזרח-מערב, כיוון-השעון);
        # shapely.affinity.rotate משתמש בקונבנציה המתמטית (נגד-כיוון-השעון) — לכן
        # מסובבים ב- -street_az (ולא +street_az) כדי ליישר את הריבוע לכיוון הרחוב.
        # אומת אמפירית (שישה bearings סינתטיים) לפני ריצה מלאה — ר' plan.
        b0 = affinity.rotate(b0, -street_az[i], origin="center")
        shadows.append(b0.union(affinity.translate(b0, dx_b[i], dy_b[i])).convex_hull)

    # ── עצים (שלב 2) — רק עצים שיכולים בכלל להגיע לרחוב בזווית-השמש הזו ──────
    reach = _TREE_HEIGHT_MAX / tan_a + _TREE_FILTER_MARGIN_M
    relevant = np.nonzero(tree_edge_dist <= reach)[0]
    if len(relevant):
        th = tree_heights[relevant]
        dx_t = -s[0] * th / tan_a; dy_t = -s[1] * th / tan_a
        for j, i in enumerate(relevant):
            poly = tree_geoms[i]
            shadows.append(poly.union(affinity.translate(poly, dx_t[j], dy_t[j])).convex_hull)

    # ── STRtree מאוחד מבנים+עצים לשעה זו (immutable — נבנה פעם אחת מהרשימה השלמה) ──
    shadow_rtree = STRtree(shadows)
    geoms = ef.geometry.values
    cov = np.zeros(len(geoms))
    for i, ln in enumerate(geoms):
        if ln.length <= 0:
            continue
        idx = shadow_rtree.query(ln)
        if len(idx) == 0:
            continue
        hit = [ln.intersection(shadows[j]) for j in idx]
        hit = [x for x in hit if not x.is_empty]
        if hit:
            cov[i] = min(unary_union(hit).length / ln.length, 1.0)
    return cov, alt, az, len(relevant)


def main():
    ef, BX, BY, BH, half_width, street_az, tree_geoms, tree_heights, tree_edge_dist = _load()
    print(f"קשתות={len(ef):,}  מבנים={len(BX):,}  עצים={len(tree_geoms):,}  שעות={len(HOURS)}", flush=True)
    print(f"  footprint half_width: min={half_width.min():.1f}  median={np.median(half_width):.1f}  "
          f"max={half_width.max():.1f}  (מטר)", flush=True)
    out = ef[["u", "v", "key"]].copy()
    for hr in HOURS:
        t0 = time.time()
        cov, alt, az, n_trees = _coverage(
            ef, BX, BY, BH, half_width, street_az, tree_geoms, tree_heights, tree_edge_dist, hr
        )
        out[f"{hr:.1f}"] = cov.round(3)
        print(f"  {hr:4.1f}h  alt={alt:5.1f}° az={az:6.1f}°  cov_avg={cov.mean()*100:4.1f}%  "
              f"עצים-רלוונטיים={n_trees:,}  ({time.time()-t0:.1f}s)", flush=True)
    out.to_parquet(OUT, index=False)
    print(f"\nנשמר {OUT}  ({len(out):,} קשתות × {len(HOURS)} שעות)", flush=True)


if __name__ == "__main__":
    main()
