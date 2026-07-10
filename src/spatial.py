"""
חישוב פיצ'רים מרחביים לכל קשת ברשת OSMnx — Spatial Join עם מבנים ועצים.

פלט: data/edges_features.parquet
עמודות: u, v, key, length, mean_building_height, tree_canopy_ratio, street_azimuth
"""
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

OUTPUT_PATH = Path("data/edges_features.parquet")
BUILDINGS_CSV = Path("data/buildings_clean.csv")
TREES_PATH = Path("data/national_canopy_clean.parquet")

# CRS מטרי לחישובי buffer
_CRS_METRIC = "EPSG:2039"
# buffer נפרד: מבנים 25מ' (צנטרואידים — מרכז המבנה רחוק מהציר); עצים 10מ' עם חיתוך
# פוליגונים אמיתי (5מ' פספס עצים שחופתם פרושה מעל הרחוב בעוד מרכזם מחוץ ל-buffer;
# היחס יציב כי שטח החיתוך גדל יחד עם שטח ה-buffer).
_BUILDING_BUFFER_M = 25
_CANOPY_BUFFER_M = 10


def _load_buildings(buildings_path: Path = BUILDINGS_CSV) -> gpd.GeoDataFrame:
    """טוען צנטרואידים של מבנים מ-CSV (lat/lon + height) כנקודות ב-EPSG:2039.

    ה-GeoJSON הפוליגונלי המקורי אינו נדרש — buildings_clean.csv מכיל את מרכז כל
    מבנה ואת גובהו (לאחר ניקוי + imputation). לכן ה-buffer למבנים גדול יותר.
    """
    df = pd.read_csv(buildings_path)
    df = df[df["height"].notna() & (df["height"] > 0)]
    gdf = gpd.GeoDataFrame(
        df[["height"]].copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(_CRS_METRIC)
    return gdf


def _load_trees(trees_path: Path = TREES_PATH) -> gpd.GeoDataFrame:
    """טוען פוליגוני חופת עצים מ-Parquet (geometry אמיתי) ב-EPSG:2039.

    בעבר השתמשנו בצנטרואידים (lat/lon) — אבל זה פספס עצים שחופתם פרושה מעל הרחוב
    בעוד מרכזם מחוץ ל-buffer (למשל עצי שדרות רוטשילד). עכשיו משתמשים בפוליגון עצמו
    ובחיתוך אמיתי מולו (ראה compute_edge_features).

    כולל גם canopy_area_m2/area_class (לא רק geometry) — נדרשים ב-precompute_shadow.py
    לגזירת גובה-עץ משוער לצל דינמי (שלב 2, אין שדה גובה אמיתי בנתונים).
    """
    gdf = gpd.read_parquet(trees_path)[["geometry", "canopy_area_m2", "area_class"]].to_crs(_CRS_METRIC)
    gdf = gdf[gdf.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)
    return gdf


def _compute_street_azimuth(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    מחשב bearing (0°–180°) לכל קשת מתוך ה-LineString שלה.
    0° = רחוב צפון-דרום, 90° = רחוב מזרח-מערב.
    כיוון הרחוב סימטרי (N==S), לכן mod 180.
    """
    def _bearing(line) -> float:
        coords = list(line.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        return float(np.degrees(np.arctan2(dx, dy)) % 180)

    return gdf.geometry.apply(_bearing).values


def compute_edge_features(
    G: nx.MultiDiGraph,
    buildings_path: Path = BUILDINGS_CSV,
    trees_path: Path = TREES_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    מחשב mean_building_height, tree_canopy_ratio ו-street_azimuth לכל קשת ב-G.

    תהליך:
      1. קשתות → GeoDataFrame (LineString) → EPSG:2039
      2. buffer נפרד: מבנים 25מ' (צנטרואידים), עצים 10מ' (חיתוך פוליגונים)
      3. sjoin עם צנטרואידי מבנים → ממוצע גובה לכל קשת
      4. חיתוך פוליגוני חופה עם buffer העצים → שטח חיתוך → ratio מתוך שטח buffer
      5. bearing (כיוון רחוב 0°–180°) מגיאומטריה
      6. שמירה ל-output_path

    מחזיר DataFrame עם עמודות: u, v, key, length,
                                 mean_building_height, tree_canopy_ratio, street_azimuth
    """
    print("Loading edges from graph...")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False).reset_index()
    edges_gdf = edges_gdf.to_crs(_CRS_METRIC)

    # מוסיף מזהה ייחודי לכל קשת לצורך ה-join
    edges_gdf["_eid"] = np.arange(len(edges_gdf))

    # buffer נפרד למבנים (רחב — צנטרואידים) ולעצים (צר — רגיש לשטח)
    bld_buffers = edges_gdf[["_eid"]].copy()
    bld_buffers["geometry"] = edges_gdf.geometry.buffer(_BUILDING_BUFFER_M)
    bld_buffers = gpd.GeoDataFrame(bld_buffers, geometry="geometry", crs=_CRS_METRIC)

    can_buffers = edges_gdf[["_eid"]].copy()
    can_buffers["geometry"] = edges_gdf.geometry.buffer(_CANOPY_BUFFER_M)
    can_buffers = gpd.GeoDataFrame(can_buffers, geometry="geometry", crs=_CRS_METRIC)
    canopy_areas = can_buffers.geometry.area  # m² — מכנה ל-canopy ratio

    # ── Spatial Join: מבנים (צנטרואיד בתוך buffer 25מ') ───────────────────────
    print(f"Loading buildings from {buildings_path} ...")
    buildings = _load_buildings(buildings_path)
    print(f"  {len(buildings):,} buildings with valid height")

    joined_b = gpd.sjoin(
        bld_buffers[["_eid", "geometry"]],
        buildings,
        how="left",
        predicate="intersects",
    )
    mean_heights = (
        joined_b.groupby("_eid")["height"]
        .mean()
        .rename("mean_building_height")
    )

    # ── Spatial Join: עצים (חיתוך פוליגונים אמיתי, buffer 10מ') ───────────────
    print(f"Loading tree canopy polygons from {trees_path} ...")
    trees = _load_trees(trees_path)
    print(f"  {len(trees):,} tree canopy polygons")

    _pairs = gpd.sjoin(
        can_buffers[["_eid", "geometry"]], trees,
        how="inner", predicate="intersects",
    )
    _csum = np.zeros(len(edges_gdf))
    if len(_pairs):
        # שטח החיתוך האמיתי בין ה-buffer לכל פוליגון חופה שחותך אותו
        _tree_geom = trees.geometry.loc[_pairs["index_right"].values]
        _inter = gpd.GeoSeries(_pairs.geometry.values, crs=_CRS_METRIC).intersection(
            gpd.GeoSeries(_tree_geom.values, crs=_CRS_METRIC)
        ).area.values
        np.add.at(_csum, _pairs["_eid"].values, _inter)
    canopy_sum = pd.Series(
        _csum, index=pd.Index(edges_gdf["_eid"].values, name="_eid"), name="canopy_sum"
    )

    # ── מיזוג תוצאות ────────────────────────────────────────────────────────
    result = edges_gdf[["u", "v", "key", "length", "_eid"]].copy()
    result = result.merge(mean_heights, on="_eid", how="left")
    result = result.merge(canopy_sum, on="_eid", how="left")
    result["mean_building_height"] = result["mean_building_height"].fillna(0.0)
    result["tree_canopy_ratio"] = np.clip(
        result["canopy_sum"].fillna(0.0) / canopy_areas.values,
        0.0, 1.0,
    )
    result = result.drop(columns=["canopy_sum"])

    # מחבר geometry (WGS84) לתוצאה — נדרש לציור מפת TCI
    result_geo = edges_gdf[["_eid", "geometry"]].merge(result, on="_eid").drop(columns=["_eid"])
    result_geo = gpd.GeoDataFrame(result_geo, geometry="geometry", crs=_CRS_METRIC)
    result_geo = result_geo.to_crs("EPSG:4326")
    result_geo["street_azimuth"] = _compute_street_azimuth(result_geo)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_geo.to_parquet(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    print(f"  {len(result_geo):,} edges | mean_building_height: "
          f"{result_geo['mean_building_height'].mean():.1f}m | "
          f"tree_canopy_ratio: {result_geo['tree_canopy_ratio'].mean():.3f} | "
          f"street_azimuth mean: {result_geo['street_azimuth'].mean():.1f}°")
    return result_geo
