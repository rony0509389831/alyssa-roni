"""
מודול ניווט — גאוקודינג, ניווט דרך OSRM API וציור מסלול.

ניווט: OSRM demo server (routing בC++, ~200ms).
גאוקודינג: Nominatim דרך OSMnx.
הגרף (load_graph) שמור לשלב M3 — ניווט מוצל עם משקלי קשת מותאמים.
"""
from pathlib import Path

import folium
import networkx as nx
import osmnx as ox
import requests

TA_LAT, TA_LON = 32.0853, 34.7818
TA_BBOX = (32.02, 34.73, 32.15, 34.85)   # (lat_min, lon_min, lat_max, lon_max)
GRAPH_PATH = Path("data/tel_aviv_walk.graphml")
WALK_SPEED_MPM = 80  # מטר לדקה — לחיזוי זמן הליכה כ-fallback

_OSRM_BASE = "https://routing.openstreetmap.de/routed-foot/route/v1/driving"


def load_graph() -> nx.MultiDiGraph:
    """
    טוען גרף רחובות להולכי רגל של תל אביב.

    סדר עדיפויות:
      1. קובץ GraphML מקומי (data/tel_aviv_walk.graphml) — טעינה מהירה
      2. הורדה מ-OSM דרך OSMnx (~30 שניות בפעם הראשונה), ואז שמירה לדיסק.
    """
    if GRAPH_PATH.exists():
        return ox.load_graphml(GRAPH_PATH)
    G = ox.graph_from_place(
        "Tel Aviv-Yafo, Israel",
        network_type="walk",
        simplify=True,
    )
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, filepath=GRAPH_PATH)
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


def build_route_map(
    origin_latlon: tuple,
    dest_latlon: tuple,
    route_result: dict,
) -> folium.Map:
    """
    בונה מפת Folium עם המסלול, נקודות ההתחלה והסיום.
    המפה מותאמת אוטומטית לגבולות המסלול.
    """
    m = folium.Map(location=[TA_LAT, TA_LON], zoom_start=14, tiles="CartoDB positron")

    folium.PolyLine(
        route_result["route_latlon"],
        color="#2980b9",
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
