"""
מודול ניווט — גאוקודינג, ניווט דרך OSRM API וציור מסלול.

ניווט מהיר: OSRM demo server (Contraction Hierarchies, ~200ms).
ניווט מוצל: Dijkstra על גרף OSMnx עם משקלי TCI מ-RandomForest.
גאוקודינג: Nominatim דרך OSMnx.
"""
from pathlib import Path

import math
import re
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests

try:
    import pysolar.solar as _solar
    _PYSOLAR = True
except ImportError:
    _PYSOLAR = False

# מפנה את cache ה-geocoding לתיקיית הטמפ' של המערכת, לא לתיקיית הפרויקט
ox.settings.cache_folder = tempfile.gettempdir()

TA_LAT, TA_LON = 32.0853, 34.7818
TA_BBOX = (32.02, 34.73, 32.15, 34.85)   # (lat_min, lon_min, lat_max, lon_max)
GRAPH_PATH = Path("data/tel_aviv_walk.graphml")
WALK_SPEED_MPM = 80  # מטר לדקה — לחיזוי זמן הליכה כ-fallback

_OSRM_BASE = "https://routing.openstreetmap.de/routed-foot/route/v1/driving"

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_HEADERS = {"User-Agent": "SHADY-TelAviv-Navigator/1.0"}

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# bbox לOverpass: south,west,north,east
_TA_OVERPASS_BBOX = f"({TA_BBOX[0]},{TA_BBOX[1]},{TA_BBOX[2]},{TA_BBOX[3]})"

_PHOTON_URL = "https://photon.komoot.io/api"

# cache ברמת המודול — נשמר לכל חיי התהליך, נמנע טעינה חוזרת (4.6s) בכל לחיצה
_GRAPH_CACHE: nx.MultiDiGraph | None = None

# ── כיסוי-צל מבנים מחושב מראש (precompute_shadow.py) ─────────────────────────
SHADOW_PATH = Path("data/shadow_coverage.parquet")
_COV_CACHE = None

# העדפת רחובות ירוקים בניווט: קשת מקבלת הנחת-עלות אם הרחוב שלה "ירוק".
# רחוב ירוק = ממוצע כיסוי-עצים > 35%, או שדרה מעל רף רוטשילד (24%) — כי עצי
# הטיילת המרכזית של בולווארדים (רוטשילד, ח"ן...) תת-נספרים בקשתות הכביש, אז סף
# רגיל היה מפספס אותם. ה-TCI עצמו לא מושפע — זו העדפת ניווט בלבד.
BOULEVARD_WEIGHT_FACTOR = 0.5      # קשת ברחוב ירוק עולה חצי → מועדפת חזק
CANOPY_STREET_THRESHOLD = 0.35     # סף כיסוי-עצים לרחוב רגיל
BOULEVARD_CANOPY_FLOOR = 0.24      # רף נמוך יותר לשדרות (רף רוטשילד ≈24.2%)
PEDESTRIAN_WEIGHT_FACTOR = 0.8     # שביל ייעודי להולכי-רגל מקבל הנחה 20%
EDGES_FEATURES_PATH = Path("data/edges_features.parquet")

_PREF_CACHE = None


def _preferred_edges():
    """קבוצת (u,v) של קשתות ברחובות ירוקים (כיסוי>35%, או שדרה≥24%). ממוטמן."""
    global _PREF_CACHE
    if _PREF_CACHE is not None:
        return _PREF_CACHE
    pref = set()
    try:
        uvname = {}
        for u, v, dd in load_graph().edges(data=True):
            n = dd.get("name"); n = n[0] if isinstance(n, list) else n
            uvname[(int(u), int(v))] = n; uvname[(int(v), int(u))] = n
        feat = pd.read_parquet(EDGES_FEATURES_PATH,
                               columns=["u", "v", "tree_canopy_ratio", "length"])
        feat["name"] = [uvname.get((int(u), int(v))) for u, v in zip(feat["u"], feat["v"])]
        feat = feat.dropna(subset=["name"])
        feat["_cw"] = feat["tree_canopy_ratio"].fillna(0) * feat["length"].clip(lower=1)
        grp = feat.groupby("name").agg(cw=("_cw", "sum"), L=("length", "sum"))
        canopy = grp["cw"] / grp["L"]
        green = set()
        for name, c in canopy.items():
            is_place = name.startswith(("כיכר", "שביל", "סימטת"))
            is_blvd = name.startswith("שדרות")
            if (c > CANOPY_STREET_THRESHOLD and not is_place) or (is_blvd and c >= BOULEVARD_CANOPY_FLOOR):
                green.add(name)
        for uv, n in uvname.items():
            if n in green:
                pref.add(uv)
    except Exception:
        return pref   # לא מאחסן — מאפשר ניסיון חוזר בהפעלה הבאה
    _PREF_CACHE = pref
    return pref

# מיקום השמש (גובה°, אזימוט°) לכל שעת-ייחוס ביום קיץ — לבחירת עמודת הכיסוי
# הקרובה ביותר למצב השמש הנוכחי בניווט. מקור: precompute_shadow.py (REF_DATE 27/6).
_SHADOW_HOURS = [
    ("6.0", 3.8, 64.6), ("6.5", 9.5, 68.3), ("7.0", 15.4, 71.8), ("7.5", 21.5, 75.2),
    ("8.0", 27.7, 78.5), ("8.5", 33.9, 81.9), ("9.0", 40.2, 85.3), ("9.5", 46.6, 89.0),
    ("10.0", 52.9, 93.1), ("10.5", 59.2, 98.0), ("11.0", 65.5, 104.4), ("11.5", 71.5, 113.6),
    ("12.0", 76.9, 129.3), ("12.5", 80.7, 159.7), ("13.0", 80.5, 203.0), ("13.5", 76.6, 232.1),
    ("14.0", 71.1, 247.2), ("14.5", 65.0, 256.1), ("15.0", 58.8, 262.3), ("15.5", 52.5, 267.2),
    ("16.0", 46.1, 271.3), ("16.5", 39.8, 274.9), ("17.0", 33.5, 278.4), ("17.5", 27.2, 281.7),
    ("18.0", 21.1, 285.0), ("18.5", 15.0, 288.4), ("19.0", 9.1, 291.9),
]


def load_shadow_coverage():
    """טוען (וממטמן) את טבלת כיסוי-הצל העירונית (u,v,key + עמודה לכל שעה). None אם חסר."""
    global _COV_CACHE
    if _COV_CACHE is None and SHADOW_PATH.exists():
        _COV_CACHE = pd.read_parquet(SHADOW_PATH)
    return _COV_CACHE


def _nearest_shadow_hour(sun_altitude: float, sun_azimuth: float) -> str:
    """עמודת השעה שמיקום השמש שלה (גובה+אזימוט מעגלי) הכי קרוב למצב הנוכחי."""
    best, best_d = _SHADOW_HOURS[0][0], 1e18
    for hcol, halt, haz in _SHADOW_HOURS:
        d_az = ((sun_azimuth - haz + 180) % 360) - 180
        d = (sun_altitude - halt) ** 2 + d_az ** 2
        if d < best_d:
            best_d, best = d, hcol
    return best


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


def _nominatim_search(query: str) -> tuple | None:
    """קריאה ישירה ל-Nominatim API עם מיקוד גיאוגרפי לת"א."""
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "il",
        # viewbox: lon_min,lat_max,lon_max,lat_min — פורמט Nominatim
        "viewbox": f"{TA_BBOX[1]},{TA_BBOX[2]},{TA_BBOX[3]},{TA_BBOX[0]}",
        "bounded": 0,
        "accept-language": "he,en",
    }
    resp = requests.get(_NOMINATIM_URL, params=params,
                        headers=_NOMINATIM_HEADERS, timeout=10)
    data = resp.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None


def _overpass_search(name: str) -> tuple | None:
    """חיפוש POI לפי שם ב-Overpass API (OSM) — מכסה קניונים, מסעדות, מוסדות."""
    query = (
        f"[out:json][timeout:10];"
        f"("
        f'  node["name"="{name}"]{_TA_OVERPASS_BBOX};'
        f'  way["name"="{name}"]{_TA_OVERPASS_BBOX};'
        f'  relation["name"="{name}"]{_TA_OVERPASS_BBOX};'
        f");"
        f"out center 1;"
    )
    try:
        resp = requests.post(
            _OVERPASS_URL,
            data={"data": query},
            headers=_NOMINATIM_HEADERS,
            timeout=12,
        )
        for el in resp.json().get("elements", []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lon:
                return float(lat), float(lon)
    except Exception:
        pass
    return None


def _photon_search(query: str) -> tuple | None:
    """Photon geocoder (komoot, OSM-based, fuzzy) — fallback שלישי לשמות מקומות לא מדויקים."""
    params = {
        "q": query,
        "limit": 5,
        "lat": TA_LAT,
        "lon": TA_LON,
        "zoom": 14,
        "lang": "he",
    }
    try:
        resp = requests.get(_PHOTON_URL, params=params,
                            headers=_NOMINATIM_HEADERS, timeout=6)
        for feat in resp.json().get("features", []):
            coords = feat["geometry"]["coordinates"]
            lon, lat = float(coords[0]), float(coords[1])
            if TA_BBOX[0] <= lat <= TA_BBOX[2] and TA_BBOX[1] <= lon <= TA_BBOX[3]:
                return lat, lon
    except Exception:
        pass
    return None


def geocode_address(address: str) -> tuple:
    """
    ממיר כתובת טקסטואלית (עברית או אנגלית) לקואורדינטות (lat, lon).

    שלב 1 — Nominatim (3 צורות שאילתה).
    שלב 2 — Overpass API: fallback לשמות POI (ללא מספר בית) — מכסה כל ישות
             ב-OSM עם שם מדויק, כולל אלה שNominatim מחזיר בשם שונה מעט.
    """
    lower = address.lower()
    queries = []
    if "tel" not in lower:
        queries.append(f"{address}, תל אביב")
        queries.append(f"{address}, ישראל")
    queries.append(address)

    for q in queries:
        try:
            point = _nominatim_search(q)
        except Exception:
            continue
        if point is None:
            continue
        lat, lon = point
        if TA_BBOX[0] <= lat <= TA_BBOX[2] and TA_BBOX[1] <= lon <= TA_BBOX[3]:
            return lat, lon

    # Nominatim לא מצא — ננסה Overpass לשמות POI (כתובת ללא ספרות = שם מקום)
    if not re.search(r"\d", address):
        _bare = address.strip().split(",")[0].strip()
        for _name in dict.fromkeys([address.strip(), _bare]):   # ללא כפילויות
            _pt = _overpass_search(_name)
            if _pt is not None:
                lat, lon = _pt
                if TA_BBOX[0] <= lat <= TA_BBOX[2] and TA_BBOX[1] <= lon <= TA_BBOX[3]:
                    return lat, lon

    # Photon: fuzzy geocoding — מכסה שמות חנויות/מסעדות ואיות חלופי שלא ב-OSM
    for _pq in dict.fromkeys([address, f"{address} תל אביב"]):
        _ph = _photon_search(_pq)
        if _ph is not None:
            return _ph

    raise ValueError(
        f"המיקום '{address}' לא נמצא באזור תל אביב — "
        "נסה כתובת מדויקת יותר (למשל: 'רחוב דיזינגוף 50')"
    )


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
    shade_factor: float = 1.0,
) -> tuple:
    """
    מחשב TCI לכל 58k קשתות הגרף ומחזיר (weight_dict, tci_by_uv).

    פונקציה נפרדת כדי לאפשר caching ב-Streamlit לפי תנאי מזג אוויר —
    אם השמש ומזג האוויר לא השתנו מאז הקריאה הקודמת, התוצאה חוזרת מה-cache (0ms).
    """
    ef = pd.DataFrame({
        "u":               edges_df["u"].to_numpy(),
        "v":               edges_df["v"].to_numpy(),
        "key":             edges_df["key"].to_numpy() if "key" in edges_df else 0,
        "length":          edges_df["length"].to_numpy(),
        "building_height": edges_df["mean_building_height"].to_numpy(),
        "canopy_ratio":    edges_df["tree_canopy_ratio"].to_numpy(),
    })
    ef["building_height"] = ef["building_height"].fillna(0.0)
    ef["canopy_ratio"]    = ef["canopy_ratio"].fillna(0.0)

    # כיסוי-צל מבנים: lookup לעמודת השעה הקרובה למצב השמש הנוכחי (יום קיץ ייחוס).
    # מחליף את shadow_angle הישן — כעת אות הצל זהה לזה שבמצב האנליטי ובאימון.
    cov_tbl = load_shadow_coverage()
    if cov_tbl is not None and sun_altitude > 0:
        hcol = _nearest_shadow_hour(float(sun_altitude), float(sun_azimuth))
        cov_sub = cov_tbl[["u", "v", "key", hcol]].drop_duplicates(["u", "v", "key"])
        ef = ef.merge(cov_sub, on=["u", "v", "key"], how="left")
        ef["shadow_cov"] = ef[hcol].fillna(0.0)
    else:
        ef["shadow_cov"] = 0.0

    ef["sun_altitude"] = float(sun_altitude)
    ef["cloud_cover"]  = float(cloud_cover)
    ef["temperature"]  = float(temperature)
    ef["humidity"]     = float(humidity)

    if sun_altitude <= 0:
        ef["tci"] = 1.0
    else:
        X = ef[model_bundle["features"]].fillna(0.0)
        ef["tci"] = np.clip(model_bundle["model"].predict(X), 1.0, 10.0)

    # שקלול הניווט: TCI × אורך, עם הנחת-עלות לקשתות ברחובות ירוקים.
    _pref = _preferred_edges()
    _is_pref = np.fromiter(
        ((int(u), int(v)) in _pref for u, v in zip(ef["u"], ef["v"])),
        dtype=bool, count=len(ef),
    )
    _factor = np.where(_is_pref, BOULEVARD_WEIGHT_FACTOR, 1.0)

    # העדפת שבילי הולכי-רגל ייעודיים (footway/pedestrian/path) — הנחה נוספת 20%.
    if "highway" in edges_df.columns:
        _ped_types = {"footway", "pedestrian", "path", "living_street"}
        _is_ped = edges_df["highway"].fillna("").apply(
            lambda h: (h[0] if isinstance(h, list) else str(h)) in _ped_types
        ).to_numpy()
        _factor = np.where(_is_ped, PEDESTRIAN_WEIGHT_FACTOR, 1.0) * _factor

    # shade_factor>1 מגביר הבדלי TCI: TCI=2→2^1.5≈2.8, TCI=8→8^1.5≈22.6 — פי 8 במקום פי 4
    _tci_w = np.clip(ef["tci"] ** shade_factor, 1.0, None)
    ef["tci_weight"] = _tci_w * ef["length"].clip(lower=0.1) * _factor

    _gb = ef.groupby(["u", "v"]).agg(tci_weight=("tci_weight", "min"), tci=("tci", "mean"))
    weight_dict = _gb["tci_weight"].to_dict()
    tci_by_uv   = _gb["tci"].to_dict()
    for (u, v), w in list(weight_dict.items()):
        weight_dict.setdefault((v, u), w)
        tci_by_uv.setdefault((v, u), tci_by_uv[(u, v)])

    return weight_dict, tci_by_uv


def _haversine_m(G: nx.MultiDiGraph, u, v) -> float:
    """מרחק ישר (מטרים) בין שני צמתים — היוריסטיקה admissible ל-A*."""
    lat1 = math.radians(G.nodes[u]["y"]); lon1 = math.radians(G.nodes[u]["x"])
    lat2 = math.radians(G.nodes[v]["y"]); lon2 = math.radians(G.nodes[v]["x"])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(a))


def compute_shaded_route(
    origin_latlon: tuple,
    dest_latlon: tuple,
    weight_dict: dict,
    tci_by_uv: dict,
    G: nx.MultiDiGraph = None,
) -> dict:
    """
    מחשב מסלול הליכה מוצל בין שתי נקודות דרך A* עם משקלי TCI.

    A* עם היוריסטיקת haversine ×0.5 (admissible: min_weight_per_m = TCI_min×factor_min = 0.5).
    מחזיר את אותו מסלול אופטימלי כמו Dijkstra אך עם פחות צמתים שנחקרים (~20-30% מהיר יותר).
    """
    if G is None:
        G = load_graph()

    orig_node = ox.nearest_nodes(G, X=origin_latlon[1], Y=origin_latlon[0])
    dest_node = ox.nearest_nodes(G, X=dest_latlon[1], Y=dest_latlon[0])
    if orig_node == dest_node:
        raise ValueError("נקודת המוצא והיעד קרובות מדי — נסה כתובות מרוחקות יותר")

    try:
        path = nx.astar_path(
            G, orig_node, dest_node,
            heuristic=lambda u, v: _haversine_m(G, u, v) * 0.5,
            weight=lambda u, v, d: weight_dict.get((u, v), d.get("length", 50) * 5),
        )
    except nx.NetworkXNoPath:
        raise ValueError("לא נמצא מסלול בין הנקודות שהוגדרו")

    route_latlon = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

    total_dist  = 0.0
    tci_full    = []   # N-1 ערכים, None כשאין TCI לקשת
    tci_valid   = []   # לחישוב avg_tci
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            best_k   = min(G[u][v], key=lambda k: G[u][v][k].get("length", 0))
            total_dist += G[u][v][best_k].get("length", 0.0)
        tci_val = tci_by_uv.get((u, v)) or tci_by_uv.get((v, u))
        if tci_val is not None:
            val = float(tci_val)
            tci_full.append(val)
            tci_valid.append(val)
        else:
            tci_full.append(None)

    return {
        "route_latlon": route_latlon,
        "distance_m":   total_dist,
        "duration_min": total_dist / WALK_SPEED_MPM,
        "avg_tci":      float(np.mean(tci_valid)) if tci_valid else None,
        "tci_list":     tci_full,   # תמיד אורך N-1
    }


def sun_position(nav_hour: float, now: datetime = None,
                 lat: float = 32.08, lon: float = 34.77) -> tuple:
    """
    מיקום השמש (גובה°, אזימוט°) לשעה nav_hour בתאריך של `now` (ברירת מחדל: היום).

    nav_hour בשעון מקומי ישראלי — ממיר ל-UTC תוך שמירה על DST נכון (UTC+2/+3).
    אם PySolar לא מותקן — מחזיר ברירת מחדל סבירה (45°, 180°) כמו בקוד הישן.
    """
    if not _PYSOLAR:
        return 45.0, 180.0
    if now is None:
        now = datetime.now()
    hh = int(nav_hour)
    mm = min(int(round((nav_hour - hh) * 60)), 59)
    tz_il = ZoneInfo("Asia/Jerusalem")
    dt_local = datetime(now.year, now.month, now.day, hh, mm, tzinfo=tz_il)
    dt_utc = dt_local.astimezone(timezone.utc)
    alt = float(_solar.get_altitude(lat, lon, dt_utc))
    az = float(_solar.get_azimuth(lat, lon, dt_utc))
    return alt, az


def plan_route(
    origin: str,
    dest: str,
    *,
    use_shaded: bool,
    nav_hour: float,
    geocode_fn,
    weather_fn=None,
    weights_fn=None,
    has_model: bool = False,
    graph: nx.MultiDiGraph = None,
    now: datetime = None,
    shade_factor: float = 1.0,
) -> dict:
    """
    מתזמר מסלול שלם מקלט גולמי — כל לוגיקת הניווט שהייתה ב-app.py, בלי Streamlit.

    הזרימה:
      1. גאוקודינג של מוצא ויעד (דרך geocode_fn).
      2. מצב "מהיר" → OSRM ישירות.
      3. מצב "מוצל" → אם המודל/פיצ'רים חסרים, נפילה למסלול מהיר; אחרת חישוב
         מיקום השמש לשעה הנבחרת, משקלי TCI (דרך weights_fn) ו-Dijkstra מוצל.
         אם המשקלים לא זמינים → נפילה למסלול מהיר.

    Dependency injection: geocode_fn / weather_fn / weights_fn מוזרקים מבחוץ כדי
    שה-caching של Streamlit יישאר ב-app.py — המודול הזה נשאר נקי מ-Streamlit.

    מחזיר dict:
      route_result — תוצאת המסלול (route_latlon, distance_m, duration_min, avg_tci?)
      color        — צבע לציור (ירוק=מוצל, כחול=מהיר)
      mode         — "shaded" / "fast" בפועל (אחרי fallbacks)
      fallback     — None / "model_missing" / "weights_missing" (סיבת נפילה למהיר)
    """
    origin_latlon = geocode_fn(origin)
    dest_latlon = geocode_fn(dest)

    result = {"color": "#2980b9", "mode": "fast", "fallback": None,
              "origin_latlon": origin_latlon, "dest_latlon": dest_latlon}

    if not use_shaded:
        result["route_result"] = compute_route(origin_latlon, dest_latlon)
        return result

    # מצב מוצל — דורש מודל + פיצ'רים זמינים
    if not has_model:
        result["route_result"] = compute_route(origin_latlon, dest_latlon)
        result["fallback"] = "model_missing"
        return result

    sun_alt, sun_az = sun_position(nav_hour, now=now)
    if sun_alt <= 0:
        result["route_result"] = compute_route(origin_latlon, dest_latlon)
        result["fallback"] = "night"
        return result

    weather = weather_fn() if weather_fn is not None else {
        "cloud_cover": 30.0, "temperature": 27.0, "humidity": 65.0}
    if weather.get("cloud_cover", 0) >= 80:
        result["route_result"] = compute_route(origin_latlon, dest_latlon)
        result["fallback"] = "overcast"
        return result

    # עיגול לצורך cache — RF מחושב פעם אחת לכל תנאי מזג אוויר
    alt_r = round(sun_alt / 5) * 5
    az_r = round(sun_az / 10) * 10
    cloud_r = round(weather["cloud_cover"] / 10) * 10
    temp_r = round(weather["temperature"] / 5) * 5
    hum_r = round(weather["humidity"] / 10) * 10

    shade_r = round(shade_factor * 2) / 2   # מעוגל ל-0.5 לצורך cache grouping
    wdict, tci_uv = (None, None)
    if weights_fn is not None:
        wdict, tci_uv = weights_fn(alt_r, az_r, cloud_r, temp_r, hum_r, shade_r)

    if wdict is None:
        result["route_result"] = compute_route(origin_latlon, dest_latlon)
        result["fallback"] = "weights_missing"
        return result

    result["route_result"] = compute_shaded_route(
        origin_latlon, dest_latlon, wdict, tci_uv, G=graph)
    result["color"] = "#27ae60"
    result["mode"] = "shaded"
    return result


def build_route_map(
    origin_latlon: tuple,
    dest_latlon: tuple,
    route_result: dict,
    color: str = "#2980b9",
    tci_list: list | None = None,
    fast_result: dict | None = None,
) -> folium.Map:
    """
    בונה מפת Folium עם המסלול, נקודות ההתחלה והסיום.
    tci_list: רשימת TCI פר-קשת לצביעת רמזור (ירוק/כתום/אדום).
    fast_result: תוצאת מסלול מהיר להשוואה (PolyLine כחול מקווקו).
    """
    m = folium.Map(location=[TA_LAT, TA_LON], zoom_start=14, tiles="CartoDB positron")

    pts = route_result["route_latlon"]
    if tci_list and len(tci_list) == len(pts) - 1:
        def _tci_color(tci) -> str:
            if tci is None:
                return "#95a5a6"   # אפור — לא ידוע
            if tci < 4:
                return "#27ae60"   # ירוק — מוצל
            if tci < 7:
                return "#f39c12"   # כתום — ביניים
            return "#c0392b"       # אדום — שמש

        for i, tci in enumerate(tci_list):
            folium.PolyLine(
                pts[i:i + 2],
                color=_tci_color(tci),
                weight=6,
                opacity=0.9,
                tooltip=f"TCI: {tci:.1f}" if tci is not None else "TCI לא זמין",
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
                    padding:8px 12px;border-radius:8px;border:1px solid #ccc;font-size:13px;
                    line-height:1.8;direction:rtl;">
          <b>עוצמת קרינה</b><br>
          <span style="background:#27ae60;display:inline-block;width:14px;height:14px;
                border-radius:3px;margin-left:5px;"></span>מוצל (TCI 1–4)<br>
          <span style="background:#f39c12;display:inline-block;width:14px;height:14px;
                border-radius:3px;margin-left:5px;"></span>ביניים (TCI 4–7)<br>
          <span style="background:#c0392b;display:inline-block;width:14px;height:14px;
                border-radius:3px;margin-left:5px;"></span>שמש מלאה (TCI 7–10)<br>
          <span style="display:inline-block;width:28px;height:0;
                border-top:3px dashed #2980b9;margin-left:5px;vertical-align:middle;">
          </span>מסלול מהיר (OSRM)
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
    else:
        folium.PolyLine(
            pts, color=color, weight=5, opacity=0.85,
            tooltip=f"מסלול: {route_result['distance_m']:.0f} מ' | {route_result['duration_min']:.0f} דקות",
        ).add_to(m)

    if fast_result:
        folium.PolyLine(
            fast_result["route_latlon"],
            color="#2980b9",
            weight=4,
            opacity=0.7,
            dash_array="8 5",
            tooltip=f"מהיר: {fast_result['distance_m']:.0f} מ' | {fast_result['duration_min']:.0f} דקות",
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

    all_pts = list(pts) + (list(fast_result["route_latlon"]) if fast_result else [])
    lats = [p[0] for p in all_pts]
    lons = [p[1] for p in all_pts]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    return m
