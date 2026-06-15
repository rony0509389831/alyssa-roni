"""
מודול ניווט — גאוקודינג, ניווט דרך OSRM API וציור מסלול.

ניווט מהיר: OSRM demo server (Contraction Hierarchies, ~200ms).
ניווט מוצל: Dijkstra על גרף OSMnx עם משקלי TCI מ-RandomForest.
גאוקודינג: Nominatim דרך OSMnx.
"""
from pathlib import Path

import tempfile

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests

# מפנה את cache ה-geocoding לתיקיית הטמפ' של המערכת, לא לתיקיית הפרויקט
ox.settings.cache_folder = tempfile.gettempdir()

TA_LAT, TA_LON = 32.0853, 34.7818
TA_BBOX = (32.02, 34.73, 32.15, 34.85)   # (lat_min, lon_min, lat_max, lon_max)
GRAPH_PATH = Path("data/tel_aviv_walk.graphml")
WALK_SPEED_MPM = 80  # מטר לדקה — לחיזוי זמן הליכה כ-fallback

_OSRM_BASE = "https://routing.openstreetmap.de/routed-foot/route/v1/driving"

# cache ברמת המודול — נשמר לכל חיי התהליך, נמנע טעינה חוזרת (4.6s) בכל לחיצה
_GRAPH_CACHE: nx.MultiDiGraph | None = None


def load_graph() -> nx.MultiDiGraph:
    """
    טוען גרף רחובות להולכי רגל של תל אביב.

    סדר עדיפויות:
      1. cache ברמת המודול (0ms — מהשיחה השנייה ואילך)
      2. קובץ GraphML מקומי (data/tel_aviv_walk.graphml) — ~4-5s בפעם הראשונה
      3. הורדה מ-OSM דרך OSMnx, ואז שמירה לדיסק.
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE
    if GRAPH_PATH.exists():
        G = ox.load_graphml(GRAPH_PATH)
    else:
        G = ox.graph_from_place("Tel Aviv-Yafo, Israel", network_type="walk", simplify=True)
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, filepath=GRAPH_PATH)
    _GRAPH_CACHE = G
    return G


def geocode_address(address: str) -> tuple:
    """
    ממיר כתובת טקסטואלית (עברית או אנגלית) לקואורדינטות (lat, lon).

    מוסיף ', תל אביב, ישראל' אוטומטית אם הכתובת אינה כוללת "tel".
    זורק ValueError עם הודעה בעברית אם הכתובת לא נמצאה או מחוץ לאזור.
    """
    enriched = address if "tel" in address.lower() else f"{address}, תל אביב, ישראל"
    point = ox.geocode(enriched)
    if point is None:
        raise ValueError(f"הכתובת '{address}' לא נמצאה — נסה ניסוח אחר")
    lat, lon = point
    if not (TA_BBOX[0] <= lat <= TA_BBOX[2] and TA_BBOX[1] <= lon <= TA_BBOX[3]):
        raise ValueError(f"הכתובת '{address}' נמצאת מחוץ לאזור תל אביב")
    return lat, lon


def compute_route(origin_latlon: tuple, dest_latlon: tuple) -> dict:
    """
    מחשב מסלול הליכה בין שתי נקודות דרך OSRM demo server.

    OSRM משתמש ב-Contraction Hierarchies (C++) — זמן תגובה ~200ms.
    מחזיר dict עם:
      route_latlon  — רשימת (lat, lon) לאורך המסלול
      distance_m    — מרחק כולל במטרים (מ-OSRM)
      duration_min  — זמן הליכה בדקות (מ-OSRM)
    """
    lon1, lat1 = origin_latlon[1], origin_latlon[0]
    lon2, lat2 = dest_latlon[1], dest_latlon[0]
    url = f"{_OSRM_BASE}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise ValueError(f"שגיאת ניווט: {data.get('message', 'תשובה לא תקינה מהשרת')}")
    route = data["routes"][0]
    # GeoJSON מחזיר [lon, lat] — הופכים ל-(lat, lon) עבור Folium
    route_latlon = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
    distance_m = route["distance"]
    return {
        "route_latlon": route_latlon,
        "distance_m": distance_m,
        "duration_min": distance_m / WALK_SPEED_MPM,
    }


def compute_tci_weights(
    edges_df: pd.DataFrame,
    model_bundle: dict,
    sun_altitude: float,
    sun_azimuth: float,
    cloud_cover: float,
    temperature: float,
    humidity: float,
) -> tuple:
    """
    מחשב TCI לכל 58k קשתות הגרף ומחזיר (weight_dict, tci_by_uv).

    פונקציה נפרדת כדי לאפשר caching ב-Streamlit לפי תנאי מזג אוויר —
    אם השמש ומזג האוויר לא השתנו מאז הקריאה הקודמת, התוצאה חוזרת מה-cache (0ms).
    """
    ef = pd.DataFrame({
        "u":               edges_df["u"].to_numpy(),
        "v":               edges_df["v"].to_numpy(),
        "length":          edges_df["length"].to_numpy(),
        "building_height": edges_df["mean_building_height"].to_numpy(),
        "canopy_ratio":    edges_df["tree_canopy_ratio"].to_numpy(),
        "street_azimuth":  edges_df["street_azimuth"].to_numpy(),
    })
    ef["building_height"] = ef["building_height"].fillna(0.0)
    ef["canopy_ratio"]    = ef["canopy_ratio"].fillna(0.0)
    ef["street_azimuth"]  = ef["street_azimuth"].fillna(90.0)

    diff = np.abs(float(sun_azimuth) % 180 - ef["street_azimuth"])
    ef["shadow_angle"] = np.minimum(diff, 180.0 - diff)
    ef["sun_altitude"] = float(sun_altitude)
    ef["cloud_cover"]  = float(cloud_cover)
    ef["temperature"]  = float(temperature)
    ef["humidity"]     = float(humidity)

    if sun_altitude <= 0:
        ef["tci"] = 1.0
    else:
        X = ef[model_bundle["features"]].fillna(0.0)
        ef["tci"] = np.clip(model_bundle["model"].predict(X), 1.0, 10.0)

    ef["tci_weight"] = ef["tci"] * ef["length"].clip(lower=0.1)

    weight_dict = ef.groupby(["u", "v"])["tci_weight"].min().to_dict()
    tci_by_uv   = ef.groupby(["u", "v"])["tci"].mean().to_dict()
    for (u, v), w in list(weight_dict.items()):
        weight_dict.setdefault((v, u), w)
        tci_by_uv.setdefault((v, u), tci_by_uv[(u, v)])

    return weight_dict, tci_by_uv


def compute_shaded_route(
    origin_latlon: tuple,
    dest_latlon: tuple,
    weight_dict: dict,
    tci_by_uv: dict,
    G: nx.MultiDiGraph = None,
) -> dict:
    """
    מחשב מסלול הליכה מוצל בין שתי נקודות דרך Dijkstra עם משקלי TCI.

    מקבל weight_dict ו-tci_by_uv מחושבים מראש (דרך compute_tci_weights) כדי לאפשר
    caching חיצוני — הגרף ו-weight_dict נשמרים בין לחיצות ו-Dijkstra הוא הפעולה היחידה.
    """
    if G is None:
        G = load_graph()

    orig_node = ox.nearest_nodes(G, X=origin_latlon[1], Y=origin_latlon[0])
    dest_node = ox.nearest_nodes(G, X=dest_latlon[1], Y=dest_latlon[0])
    if orig_node == dest_node:
        raise ValueError("נקודת המוצא והיעד קרובות מדי — נסה כתובות מרוחקות יותר")

    try:
        path = nx.dijkstra_path(
            G, orig_node, dest_node,
            weight=lambda u, v, d: weight_dict.get((u, v), d.get("length", 50) * 5),
        )
    except nx.NetworkXNoPath:
        raise ValueError("לא נמצא מסלול בין הנקודות שהוגדרו")

    route_latlon = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

    total_dist = 0.0
    tci_vals   = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            best_k   = min(G[u][v], key=lambda k: G[u][v][k].get("length", 0))
            total_dist += G[u][v][best_k].get("length", 0.0)
        tci_val = tci_by_uv.get((u, v)) or tci_by_uv.get((v, u))
        if tci_val is not None:
            tci_vals.append(float(tci_val))

    return {
        "route_latlon": route_latlon,
        "distance_m":   total_dist,
        "duration_min": total_dist / WALK_SPEED_MPM,
        "avg_tci":      float(np.mean(tci_vals)) if tci_vals else None,
    }


def build_route_map(
    origin_latlon: tuple,
    dest_latlon: tuple,
    route_result: dict,
    color: str = "#2980b9",
) -> folium.Map:
    """
    בונה מפת Folium עם המסלול, נקודות ההתחלה והסיום.
    המפה מותאמת אוטומטית לגבולות המסלול.
    """
    m = folium.Map(location=[TA_LAT, TA_LON], zoom_start=14, tiles="CartoDB positron")

    folium.PolyLine(
        route_result["route_latlon"],
        color=color,
        weight=5,
        opacity=0.85,
        tooltip=f"מסלול: {route_result['distance_m']:.0f} מ' | {route_result['duration_min']:.0f} דקות",
    ).add_to(m)

    folium.Marker(
        location=origin_latlon,
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        popup=folium.Popup("📍 נקודת מוצא", max_width=150),
        tooltip="נקודת מוצא",
    ).add_to(m)

    folium.Marker(
        location=dest_latlon,
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        popup=folium.Popup("🏁 יעד", max_width=150),
        tooltip="יעד",
    ).add_to(m)

    lats = [p[0] for p in route_result["route_latlon"]]
    lons = [p[1] for p in route_result["route_latlon"]]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    return m
