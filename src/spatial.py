"""
חישוב פיצ'רים מרחביים לכל קשת ברשת OSMnx — Spatial Join עם מבנים ועצים.

פלט: data/edges_features.parquet
עמודות: u, v, key, length, mean_building_height, tree_canopy_ratio
"""
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

OUTPUT_PATH = Path("data/edges_features.parquet")
BUILDINGS_PATH = Path("tel_aviv_buildings.geojson")
TREES_PATH = Path("data/national_canopy_clean.parquet")

# CRS מטרי לחישובי buffer
_CRS_METRIC = "EPSG:2039"
_BUFFER_M = 5  # מרחק buffer סביב כל קשת (מטרים)


def _load_buildings(buildings_path: Path) -> gpd.GeoDataFrame:
    """טוען מבנים מ-GeoJSON, מחזיר עמודת height מ-gova_simplex_2019."""
    gdf = gpd.read_file(buildings_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 2039:
        gdf = gdf.to_crs(_CRS_METRIC)
    # בוחר עמודת גובה — מעדיף gova_simplex_2019, fallback ל-height
    if "gova_simplex_2019" in gdf.columns:
        gdf = gdf.rename(columns={"gova_simplex_2019": "height"})
    elif "height" not in gdf.columns:
        raise ValueError("buildings GeoJSON חסר עמודת גובה")
    gdf = gdf[gdf["height"].notna() & (gdf["height"] > 0)][["height", "geometry"]]
    return gdf


def _load_trees(trees_path: Path) -> gpd.GeoDataFrame:
    """טוען עצים מ-Parquet כ-GeoDataFrame של נקודות."""
    df = pd.read_parquet(trees_path, columns=["lat", "lon", "canopy_area_m2"])
    df = df.dropna(subset=["lat", "lon", "canopy_area_m2"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(_CRS_METRIC)
    return gdf[["canopy_area_m2", "geometry"]]


def compute_edge_features(
    G: nx.MultiDiGraph,
    buildings_path: Path = BUILDINGS_PATH,
    trees_path: Path = TREES_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    מחשב mean_building_height ו-tree_canopy_ratio לכל קשת ב-G.

    תהליך:
      1. קשתות → GeoDataFrame (LineString) → EPSG:2039
      2. buffer 5m סביב כל קשת
      3. sjoin עם מבנים → ממוצע גובה לכל קשת
      4. sjoin עם עצים → סכום שטח חופה → ratio מתוך שטח buffer
      5. שמירה ל-output_path

    מחזיר DataFrame עם עמודות: u, v, key, length,
                                 mean_building_height, tree_canopy_ratio
    """
    print("Loading edges from graph...")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False).reset_index()
    edges_gdf = edges_gdf.to_crs(_CRS_METRIC)

    # מוסיף מזהה ייחודי לכל קשת לצורך ה-join
    edges_gdf["_eid"] = np.arange(len(edges_gdf))
    buffers = edges_gdf.copy()
    buffers["geometry"] = edges_gdf.geometry.buffer(_BUFFER_M)
    buffer_areas = buffers.geometry.area  # m²

    # ── Spatial Join: מבנים ──────────────────────────────────────────────────
    print(f"Loading buildings from {buildings_path} ...")
    buildings = _load_buildings(buildings_path)
    print(f"  {len(buildings):,} buildings with valid height")

    joined_b = gpd.sjoin(
        buffers[["_eid", "geometry"]],
        buildings,
        how="left",
        predicate="intersects",
    )
    mean_heights = (
        joined_b.groupby("_eid")["height"]
        .mean()
        .rename("mean_building_height")
    )

    # ── Spatial Join: עצים ───────────────────────────────────────────────────
    print(f"Loading trees from {trees_path} ...")
    trees = _load_trees(trees_path)
    print(f"  {len(trees):,} tree canopy points")

    joined_t = gpd.sjoin(
        buffers[["_eid", "geometry"]],
        trees,
        how="left",
        predicate="intersects",
    )
    canopy_sum = (
        joined_t.groupby("_eid")["canopy_area_m2"]
        .sum()
        .rename("canopy_sum")
    )

    # ── מיזוג תוצאות ────────────────────────────────────────────────────────
    result = edges_gdf[["u", "v", "key", "length", "_eid"]].copy()
    result = result.merge(mean_heights, on="_eid", how="left")
    result = result.merge(canopy_sum, on="_eid", how="left")
    result["mean_building_height"] = result["mean_building_height"].fillna(0.0)
    result["tree_canopy_ratio"] = np.clip(
        result["canopy_sum"].fillna(0.0) / buffer_areas.values,
        0.0, 1.0,
    )
    result = result.drop(columns=["canopy_sum"])

    # מחבר geometry (WGS84) לתוצאה — נדרש לציור מפת TCI
    result_geo = edges_gdf[["_eid", "geometry"]].merge(result, on="_eid").drop(columns=["_eid"])
    result_geo = gpd.GeoDataFrame(result_geo, geometry="geometry", crs=_CRS_METRIC)
    result_geo = result_geo.to_crs("EPSG:4326")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_geo.to_parquet(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    print(f"  {len(result_geo):,} edges | mean_building_height: "
          f"{result_geo['mean_building_height'].mean():.1f}m | "
          f"tree_canopy_ratio: {result_geo['tree_canopy_ratio'].mean():.3f}")
    return result_geo
