"""
מודול ניווט — גאוקודינג, ניווט דרך OSRM API וציור מסלול.

ניווט מהיר: OSRM demo server (Contraction Hierarchies, ~200ms).
ניווט מוצל: Dijkstra על גרף OSMnx עם משקלי TCI מ-RandomForest.
גאוקודינג: Nominatim דרך OSMnx.
"""
from pathlib import Path

import difflib
import math
import random
import re
import tempfile
import time
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

# מדויק (למרכוז מפה/גיאוקוד) — שונה בכוונה מ-32.08/34.77 הגס שב-weather.py/data.py/
# precompute_shadow.py (מספיק לחישוב זווית-שמש). שם נבדל כדי לא לבלבל בין השניים.
TA_LAT_PRECISE, TA_LON_PRECISE = 32.0853, 34.7818
TA_BBOX = (32.02, 34.73, 32.15, 34.85)   # (lat_min, lon_min, lat_max, lon_max)
GRAPH_PATH = Path("data/tel_aviv_walk.graphml")
WALK_SPEED_MPM = 80  # מטר לדקה — לחיזוי זמן הליכה כ-fallback

_OSRM_BASE = "https://routing.openstreetmap.de/routed-foot/route/v1/driving"


def _log_timing(label: str, elapsed: float) -> None:
    """רשת ביטחון לאבחון איטיות — מדפיס זמן קריאה חיצונית/כבדה ל-stdout
    (נראה בטרמינל שמריץ streamlit run). לא משפיע על הלוגיקה — וקריטי:
    כשל בהדפסה עצמה (למשל קונסולת Windows עם קידוד שלא תומך בעברית,
    cp1252) נבלע בשקט כאן ולעולם לא יזלוג כ-exception החוצה, כדי שלוג
    אבחון לא יהפוך בטעות חיפוש מוצלח לכישלון geocoding.
    """
    msg = f"[TIMING] {label}: {elapsed:.2f}s"
    try:
        print(msg, flush=True)
    except Exception:
        # קונסולה שלא תומכת בעברית (למשל cp1252 ב-Windows) — עדיין נדפיס
        # גרסה בטוחה-ASCII במקום לוותר על השורה לגמרי.
        try:
            print(msg.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
        except Exception:
            pass

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_HEADERS = {"User-Agent": "SHADY-TelAviv-Navigator/1.0"}

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# bbox לOverpass: south,west,north,east
_TA_OVERPASS_BBOX = f"({TA_BBOX[0]},{TA_BBOX[1]},{TA_BBOX[2]},{TA_BBOX[3]})"

_PHOTON_URL = "https://photon.komoot.io/api"

# קודי סטטוס שמצדיקים ניסיון חוזר (rate-limit/שגיאת שרת) — לא 200 עם תוצאה ריקה,
# זה "לא נמצא" לגיטימי ולא כשל חולף.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _request_with_retry(method: str, url: str, *, retries: int = 1,
                         backoff: float = 0.5, **kwargs) -> requests.Response:
    """עטיפת requests.get/post עם ניסיון-חוזר אחד על כשל-רשת חולף (timeout,
    connection error, 429/5xx) — לא על 200 עם תוצאה ריקה. שירותי הגיאוקוד
    החינמיים (Nominatim/Overpass/Photon) נוטים לחסימות/rate-limit זמניים,
    במיוחד מ-IP משותף כמו Streamlit Cloud."""
    call = requests.get if method == "get" else requests.post
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = call(url, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff)
                continue
            raise
        if resp.status_code in _RETRYABLE_STATUS and attempt < retries:
            time.sleep(backoff)
            continue
        return resp
    raise last_exc

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
# תקרת עיקוף: המסלול המוצל לא יעלה על DETOUR_CAP × אורך המסלול הקצר ביותר.
# מונע פיתולים קיצוניים ששיפור הנוחות בהם זניח (מדידה: exp 3.0 מוסיף ~+90% אורך
# עבור שיפור TCI זניח). אם המסלול חורג — מורידים את shade_factor בהדרגה עד שעומד בתקרה.
DETOUR_CAP = 1.6

# תקרת עיקוף פר-רמת-צל: (תקרת-יחס-אורך, תקרת-תוספת-דקות). לוקחים את המחמיר
# מבין השניים לכל טיול — יחס-אורך זהה (למשל 1.6×) מתורגם לתוספת-זמן שונה
# לגמרי בהתאם לאורך הטיול (על טיול קצר זה כמה דקות, על טיול ארוך זה עשרות),
# אז לא מספיק להסתמך על יחס-אורך בלבד כדי שכל הרמות ירגישו עקביות.
# רק 2 רמות (2026-07-18): "מאוזן" הוסרה — על מסלולים אמיתיים היא כמעט תמיד
# התכנסה לאותו מסלול כמו "צל" (אין מספיק הבדל אמיתי בין עוצמות-צל סמוכות
# ברוב הטיולים כדי להצדיק 3 רמות נפרדות).
# "מעט צל" הורחב (2026-07-18, מאוחר יותר אותו ערב) מ-1.10×/5-דק' ל-1.20×/10-דק' —
# אומת חי: הרחבת האחוז לבד היא no-op על טיולים מעל ~50 דק' (איבר-הדקות הקבוע
# עדיין המחמיר), צריך להגדיל את שני האיברים ביחד. על מסלול אמיתי כזה זה פתח
# shade_factor=0.3 במקום 0.2 (עיקוף ~17% במקום ~10%) וקירר את ה-TCI ב-1.4
# נקודות — שיפור אמיתי, לא שולי. על מסלולים אחרים (קפיצה חדה בלי "אמצע") זה
# לא משנה כלום — תלוי-מסלול, לא קבוע אוניברסלי.
# "הרבה צל" הורד מ-exponent 3.0 ל-2.0 (2026-07-18): אומת חי שב-3.0 ה-A* מייצר
# מסלולים משוננים עם backtrack (עד ~81מ' אחורה מהיעד) עבור שיפור-TCI זניח (~0.3
# נקודות) — יחס-המשקל TCI^3 קיצוני מדי. ב-exponent ≤2.5 המסלול חלק (0 backtrack)
# עם כמעט אותו צל. 2.0 = הערך הגבוה הבטוח (מתחת ל"מדרון" 2.5→3.0), עדיין TCI².
_TIER_DETOUR_CAPS = {
    1.0: (1.20, 10.0),   # מעט צל — הורחב מ-(1.10, 5.0)
    2.0: (1.60, 30.0),   # הרבה צל — exponent הורד מ-3.0 (ר' הערה למעלה)
}
_DEFAULT_DETOUR_CAP = (DETOUR_CAP, 30.0)   # גיבוי אם shade_r לא אחד מ-2 הערכים המוכרים
# סף חשיפה גבוהה לשמש: מקטע עם TCI מעל הסף נחשב "חשיפה גבוהה" (לחישוב תובנות המסלול).
# 5.5 = מחצית עליונה של סקאלת ה-TCI (1-10). נבחר 5.5 ולא 7 כי בפועל ה-TCI מגיע ל-~6.4
# לכל היותר בשעות אחה"צ (מגיע מעל 7 רק סביב שיא הצהריים) — סף 7 היה משאיר את המדד 0 ברוב היום.
HIGH_EXPOSURE_TCI = 5.5
# רווח-צל מינימלי שמצדיק עיקוף: אם מסלול-הצל עוקף (ארוך מהישיר) אך משפר את ה-TCI
# הממוצע בפחות מזה מול המסלול הישיר — העיקוף לא שווה, וחוזרים לישיר. פותר את
# "העיקוף חסר-ההיגיון" כשהאזור כבר קריר (למשל ב-10:00 הכל TCI~1-2, אין מה להרוויח).
# ניתן-לכוונון; 1.0 נקודה על סקאלת 1-10 = שיפור מוחשי מינימלי.
MIN_TCI_GAIN = 1.0
# תקרת פיתול-אחורה (backtrack) למסלול המוצל: מסלול שמתרחק מהיעד ביותר מזה בקטע רציף
# נחשב "מפותל" — לולאת ה-backoff תוריד לו אקספוננט עד שיתחלק. נמדד חי: מסלולים חלקים
# מגיעים ל-~34מ' לכל היותר, המפותלים ל-56-127מ' — 45 מפריד ביניהם. תלוי-מסלול (חלקים
# ב-sf=2.0 נשארים 2.0), אז לא מקפח את "הרבה צל" גלובלית. ניתן-לכוונון.
MAX_BACKTRACK_M = 45.0
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


def _nominatim_search(query: str, house_number: str | None = None) -> tuple | None:
    """קריאה ישירה ל-Nominatim API עם מיקוד גיאוגרפי לת"א.

    house_number (אופציונלי, תוקן 2026-07-18): כשמספר-בית מבוקש, מוודא ש-
    Nominatim באמת מצא בניין באותו מספר (`address.house_number` בתשובה) —
    לא רק את הרחוב. כשמספר הבית לא סביר/לא קיים (למשל "יפת 4900"), Nominatim
    נופל בשקט לרמת-רחוב (`addresstype="road"`, בלי house_number בכלל) — בלי
    הבדיקה הזו התוצאה הזו התקבלה כאילו הייתה הכתובת המדויקת."""
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "il",
        # viewbox: lon_min,lat_max,lon_max,lat_min — פורמט Nominatim
        "viewbox": f"{TA_BBOX[1]},{TA_BBOX[2]},{TA_BBOX[3]},{TA_BBOX[0]}",
        "bounded": 0,
        "accept-language": "he,en",
        "addressdetails": 1,
    }
    _t0 = time.monotonic()
    resp = _request_with_retry("get", _NOMINATIM_URL, params=params,
                                headers=_NOMINATIM_HEADERS, timeout=10)
    _log_timing(f"Nominatim '{query}'", time.monotonic() - _t0)
    data = resp.json()
    if not data:
        return None
    if house_number is not None and data[0].get("address", {}).get("house_number") != house_number:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


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
        _t0 = time.monotonic()
        resp = _request_with_retry(
            "post", _OVERPASS_URL,
            data={"data": query},
            headers=_NOMINATIM_HEADERS,
            timeout=12,
        )
        _log_timing(f"Overpass '{name}'", time.monotonic() - _t0)
        for el in resp.json().get("elements", []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lon:
                return float(lat), float(lon)
    except Exception:
        pass
    return None


def _photon_search(query: str, house_number: str | None = None) -> tuple | None:
    """Photon geocoder (komoot, OSM-based, fuzzy) — fallback שלישי לשמות מקומות לא מדויקים.

    בלי פרמטר "lang" (הוסר 2026-07-18): Photon תומך רק ב-default/de/en/fr, לא
    "he" — כשהוא נשלח, ה-API דחה כל בקשה בשקט (נבלע ע"י ה-try/except למטה),
    כך שכל השלב הזה מעולם לא עבד בפועל. house_number (אופציונלי): כמו ב-
    Nominatim, מוודא שהתוצאה היא באמת מספר-הבית המבוקש ולא רק התאמה מטושטשת
    לרחוב (Photon סובל שגיאות-כתיב, אבל לא אמור "לנחש" מספר-בית שלא התבקש)."""
    params = {
        "q": query,
        "limit": 5,
        "lat": TA_LAT_PRECISE,
        "lon": TA_LON_PRECISE,
        "zoom": 14,
    }
    try:
        _t0 = time.monotonic()
        resp = _request_with_retry("get", _PHOTON_URL, params=params,
                                    headers=_NOMINATIM_HEADERS, timeout=6)
        _log_timing(f"Photon '{query}'", time.monotonic() - _t0)
        for feat in resp.json().get("features", []):
            props = feat.get("properties", {})
            if house_number is not None and props.get("housenumber") != house_number:
                continue
            coords = feat["geometry"]["coordinates"]
            lon, lat = float(coords[0]), float(coords[1])
            if TA_BBOX[0] <= lat <= TA_BBOX[2] and TA_BBOX[1] <= lon <= TA_BBOX[3]:
                return lat, lon
    except Exception:
        pass
    return None


# "הידעת" — מוצגים לצד הודעות ההתקדמות בזמן חיפוש הכתובת (השלב עם הזמן הכי
# משתנה, כי הוא תלוי ברשת) וגם ככרטיס פתיחה ב-app.py (נבחר פעם אחת ל-session).
# מקור אמת יחיד — לא כפול בין הקבצים.
_DID_YOU_KNOW = (
    "הידעת? שטחי אספלט כהים חשופים לשמש מסוגלים להגיע לטמפרטורות קיצוניות של מעל 70 מעלות צלזיוס",
    "הידעת? הבדל הטמפרטורה של פני השטח בין מדרכה בשמש למדרכה מוצלת באותו רחוב בדיוק יכול להגיע לעד 20 עד 30 מעלות צלזיוס",
    "הידעת? מחקרים מצאו שרחובות המעוצבים עם רצף חופת עצים צפופה ומחוברת מצליחים להוריד את טמפרטורת האוויר בסביבתם בעד 5 מעלות",
    "הידעת? עצים רחבים ברחוב לא רק מקררים, אלא גם מסננים זיהום אוויר. חופת עצים צפופה לצד כבישים מפחיתה את חלקיקי הפיח והמזהמים בעד 30% עבור הולכי הרגל על המדרכה!",
    "הידעת? הליכה בשמש גורמת לנו להאיץ את הקצב באופן לא מודע כדי לברוח מהחום, מה שמגביר את הדופק והזעה.",
    "הידעת? חשיפה מצטברת לקרינת UV היא הגורם המרכזי ליותר מ-90% ממקרי סרטן העור הלא-מלנומיים ולרוב מקרי המלנומה.",
    "הידעת? כוויות שמש מרובות בילדות ובגיל ההתבגרות מכפילות את הסיכון לפתח מלנומה בשלב מאוחר יותר בחיים.",
    "הידעת? סרטן העור (שאינו מלנומה) הוא הסרטן הנפוץ ביותר בעולם ובפער עצום!",
    "הידעת? בשיא הקיץ בישראל, קרינת ה-UV חזקה כל כך שדי ב-10 עד 15 דקות בלבד של חשיפה לשמש ללא הגנה כדי לגרום לנזק מצטבר לעור ולהעלות את הסיכון לסרטן.",
    "הידעת? להגיד \"מלנומה\" ו\"סרטן עור\" זה לא אותו דבר! בעוד שסרטני עור רגילים (BCC/SCC) הם הנפוצים בעולם ומתפתחים לאט, מלנומה מתחילה בתאי הפיגמנט (המלנוציטים) ויכולה לשלוח גרורות במהירות",
    "הידעת? מהירות ההליכה הממוצעת של אדם בוגר בקצב רגיל ברחוב נעה בין 4.5 קמ\"ש ל-4.9 קמ\"ש.",
    "הידעת? וטרינרים ממליצים להוציא כלבים לטיול של 30 דקות עד שעתיים במצטבר בכל יום",
    "הידעת? מכל 5 אנשים בארה\"ב יפתח סרטן עור במהלך חייו.",
    "הידעת? רוב האנשים מורחים רק חצי מהכמות הנדרשת של קרם הגנה. כדי לקבל את ה-SPF שכתוב על האריזה, אתם צריכים למרוח כמות של כוס צ'ייסר מלאה לכל הגוף",
    "הידעת? ההנחיה הרשמית היא להימרח בקרם הגנה מחדש כל שעתיים בדיוק",
)


_STREET_NAMES_CACHE = None


# קידומות רחוב שכדאי גם להסיר, כדי שגם "רוטשילד" (בלי "שדרות") יתאים לשם המלא בגרף
_STREET_PREFIXES = ("שדרות ", "רחוב ", "דרך ", "סמטת ", "שביל ", "מעלה ", "כיכר ")


def _street_names() -> list:
    """מילון קנוני של שמות-רחובות עבריים מגרף ה-OSM — לתיקון שגיאות-כתיב. ממוטמן.

    כולל גם וריאנט בלי קידומת ("שדרות רוטשילד" → גם "רוטשילד"), כדי שמשתמש שכתב
    את שם הבולוור בלי "שדרות" (ועם שגיאת-כתיב) עדיין יתוקן נכון."""
    global _STREET_NAMES_CACHE
    if _STREET_NAMES_CACHE is None:
        names = set()
        for _, _, d in load_graph().edges(data=True):
            n = d.get("name")
            vals = n if isinstance(n, list) else [n]
            for x in vals:
                if not isinstance(x, str):
                    continue
                names.add(x)
                for p in _STREET_PREFIXES:
                    if x.startswith(p) and len(x) > len(p) + 2:
                        names.add(x[len(p):])
        # רק שמות עם אותיות עבריות (מסנן שמות לועזיים/ריקים)
        _STREET_NAMES_CACHE = [n for n in names if any("א" <= c <= "ת" for c in n)]
    return _STREET_NAMES_CACHE


# מספר בית בסוף הכתובת (עברית: "דיזנגוף 55", אולי עם אות: "55א")
_HOUSE_NUM_RE = re.compile(r"\s*(\d+\s*[א-ת]?)\s*$")


def _correct_street_spelling(address: str) -> str:
    """מתקן שגיאת-כתיב קטנה בשם רחוב מול מילון שמות ה-OSM (difflib).

    פועל **רק כשיש מספר בית** — סימן ברור לכתובת-רחוב. כך שם עסק/POI (בלי מספר,
    למשל "שוק הכרמל") לא "מתוקן" בטעות לרחוב דומה. סף גבוה (0.8): מתקן רק התאמה
    קרובה מאוד ("דיזינגוף"→"דיזנגוף"), משאיר שם לא-מוכר כמות-שהוא.
    """
    m = _HOUSE_NUM_RE.search(address or "")
    if not m:
        return address                       # אין מספר בית → לא נוגעים (POI/שם חופשי)
    name_part = address[:m.start()].strip()
    num = m.group(1).replace(" ", "")
    if not name_part:
        return address
    match = difflib.get_close_matches(name_part, _street_names(), n=1, cutoff=0.8)
    if match and match[0] != name_part:
        return f"{match[0]} {num}"
    return address


def geocode_address(address: str, on_progress=None) -> tuple:
    """
    ממיר כתובת טקסטואלית (עברית או אנגלית) לקואורדינטות (lat, lon).

    שלב 1 — Nominatim (3 צורות שאילתה).
    שלב 2 — Overpass API: fallback לשמות POI (ללא מספר בית) — מכסה כל ישות
             ב-OSM עם שם מדויק, כולל אלה שNominatim מחזיר בשם שונה מעט.
    שלב 3 — Photon: fuzzy geocoder שסובל שגיאות כתיב וחיפוש לפי שם עסק/חנות.

    on_progress(msg: str), אם הועבר, נקרא לפני כל שלב אמיתי בשרשרת ה-fallback
    (Nominatim → Overpass → Photon) — לצורך הצגת סטטוס חי ל-UI (לא משפיע על הלוגיקה).

    שלב 0 — תיקון שגיאת-כתיב בשם רחוב (כשיש מספר בית) מול מילון שמות ה-OSM, כדי
    ש-"דיזינגוף 55" (י' מיותרת) לא יחזיר כתובת שגויה מ-Nominatim.

    ולידציית מספר-בית (2026-07-18): כשיש מספר בית בכתובת, תוצאה מ-Nominatim/
    Photon שלא כוללת בדיוק את אותו מספר-בית (למשל נפילה לרמת-רחוב כי המספר
    לא סביר — "יפת 4900") נדחית, לא מתקבלת כאילו הייתה התאמה מדויקת.
    """
    address = _correct_street_spelling(address)
    lower = address.lower()
    _house_num_match = re.search(r"\d+", address)
    house_number = _house_num_match.group() if _house_num_match else None
    queries = []
    if "tel" not in lower:
        queries.append(f"{address}, תל אביב")
        queries.append(f"{address}, ישראל")
    queries.append(address)

    if on_progress:
        on_progress(f"מחפש את '{address}'... ({random.choice(_DID_YOU_KNOW)})")
    for q in queries:
        try:
            point = _nominatim_search(q, house_number=house_number)
        except Exception:
            continue
        if point is None:
            continue
        lat, lon = point
        if TA_BBOX[0] <= lat <= TA_BBOX[2] and TA_BBOX[1] <= lon <= TA_BBOX[3]:
            return lat, lon

    # Nominatim לא מצא — ננסה Overpass לשמות POI (כתובת ללא ספרות = שם מקום)
    if house_number is None:
        if on_progress:
            on_progress(f"לא נמצא ב-Nominatim — מנסה Overpass עבור '{address}'...")
        _bare = address.strip().split(",")[0].strip()
        for _name in dict.fromkeys([address.strip(), _bare]):   # ללא כפילויות
            _pt = _overpass_search(_name)
            if _pt is not None:
                lat, lon = _pt
                if TA_BBOX[0] <= lat <= TA_BBOX[2] and TA_BBOX[1] <= lon <= TA_BBOX[3]:
                    return lat, lon

    # Photon: fuzzy geocoding — מכסה שמות חנויות/מסעדות ואיות חלופי שלא ב-OSM
    if on_progress:
        on_progress(f"מנסה חיפוש מקורב (Photon) עבור '{address}'...")
    for _pq in dict.fromkeys([address, f"{address} תל אביב"]):
        _ph = _photon_search(_pq, house_number=house_number)
        if _ph is not None:
            return _ph

    raise ValueError(
        f"המיקום '{address}' לא נמצא באזור תל אביב — "
        "נסה כתובת מדויקת יותר (למשל: 'רחוב דיזינגוף 50')"
    )


def compute_route(origin_latlon: tuple, dest_latlon: tuple,
                   tci_by_uv: dict = None, G: nx.MultiDiGraph = None) -> dict:
    """
    מחשב מסלול הליכה בין שתי נקודות דרך OSRM demo server.

    OSRM משתמש ב-Contraction Hierarchies (C++) — זמן תגובה ~200ms.
    מחזיר dict עם:
      route_latlon  — רשימת (lat, lon) לאורך המסלול
      distance_m    — מרחק כולל במטרים (מ-OSRM)
      duration_min  — זמן הליכה בדקות (מ-OSRM)

    tci_by_uv (אופציונלי): כשמועבר, מצמידה TCI-פר-מקטע למסלול (ר'
    `_snap_tci_to_latlon_path`) כדי לאפשר צביעה/השוואה מול מסלול מוצל —
    מוסיף avg_tci/tci_list/high_exposure_m לתוצאה. כשל בהצמדה לא מפיל את
    חישוב המסלול עצמו (OSRM עדיין מוחזר בלי TCI).
    """
    lon1, lat1 = origin_latlon[1], origin_latlon[0]
    lon2, lat2 = dest_latlon[1], dest_latlon[0]
    url = f"{_OSRM_BASE}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    _t0 = time.monotonic()
    resp = requests.get(url, timeout=10)
    _log_timing("OSRM route", time.monotonic() - _t0)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise ValueError(f"שגיאת ניווט: {data.get('message', 'תשובה לא תקינה מהשרת')}")
    route = data["routes"][0]
    # GeoJSON מחזיר [lon, lat] — הופכים ל-(lat, lon) עבור Folium
    route_latlon = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
    distance_m = route["distance"]
    result = {
        "route_latlon": route_latlon,
        "distance_m": distance_m,
        "duration_min": distance_m / WALK_SPEED_MPM,
    }
    if tci_by_uv is not None:
        try:
            result.update(_snap_tci_to_latlon_path(route_latlon, tci_by_uv, G or load_graph()))
        except Exception:
            pass   # כשל בהצמדה לא יפיל את חישוב המסלול המהיר עצמו
    return result


def compute_edge_tci(
    edges_df: pd.DataFrame,
    model_bundle: dict,
    sun_altitude: float,
    sun_azimuth: float,
    cloud_cover: float,
    temperature: float,
    humidity: float,
) -> pd.DataFrame:
    """
    מחשב TCI פר-קשת (58k) + מקדם-העדפה (רחוב-ירוק/מדרכה) — **החלק היקר**
    (`model.predict` על 58k שורות) ש**אינו תלוי ב-shade_factor**. מחזיר DataFrame
    קומפקטי (u, v, tci, length, factor). מופרד מ-`weights_from_edge_tci` כדי שבזמן
    לולאת ה-backoff (הרבה shade_factor, אותו מזג-אוויר) ה-predict ירוץ פעם אחת בלבד.
    """
    _t0 = time.monotonic()
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

    _log_timing("edge TCI (58k edges, CPU)", time.monotonic() - _t0)
    return pd.DataFrame({
        "u":      ef["u"].to_numpy(),
        "v":      ef["v"].to_numpy(),
        "tci":    ef["tci"].to_numpy(),
        "length": ef["length"].to_numpy(),
        "factor": _factor,
    })


def weights_from_edge_tci(edge_tci: pd.DataFrame, shade_factor: float = 1.0) -> tuple:
    """
    **החלק הזול** התלוי-ב-shade_factor: מעלה את ה-TCI בחזקת shade_factor, בונה משקל
    (`tci^sf × אורך × מקדם-העדפה`) ומאחד פר-קשת. מחזיר (weight_dict, tci_by_uv).
    לא משנה את edge_tci במקום (הוא ממוטמן ב-app) — קורא בלבד דרך numpy.
    """
    tci        = edge_tci["tci"].to_numpy()
    # shade_factor>1 מגביר הבדלי TCI: TCI=2→2^1.5≈2.8, TCI=8→8^1.5≈22.6 — פי 8 במקום פי 4
    _tci_w     = np.clip(tci ** float(shade_factor), 1.0, None)
    tci_weight = _tci_w * np.clip(edge_tci["length"].to_numpy(), 0.1, None) * edge_tci["factor"].to_numpy()

    _gb = pd.DataFrame({
        "u": edge_tci["u"].to_numpy(),
        "v": edge_tci["v"].to_numpy(),
        "tci": tci,
        "tci_weight": tci_weight,
    }).groupby(["u", "v"]).agg(tci_weight=("tci_weight", "min"), tci=("tci", "mean"))
    weight_dict = _gb["tci_weight"].to_dict()
    tci_by_uv   = _gb["tci"].to_dict()
    for (u, v), w in list(weight_dict.items()):
        weight_dict.setdefault((v, u), w)
        tci_by_uv.setdefault((v, u), tci_by_uv[(u, v)])
    return weight_dict, tci_by_uv


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

    עטיפה דקה מעל compute_edge_tci (החלק היקר, אינו תלוי shade_factor) +
    weights_from_edge_tci (החלק הזול). נשמרה לתאימות-לאחור — פלט זהה לחלוטין.
    """
    edge_tci = compute_edge_tci(
        edges_df, model_bundle, sun_altitude, sun_azimuth,
        cloud_cover, temperature, humidity,
    )
    return weights_from_edge_tci(edge_tci, shade_factor)


def _haversine_latlon(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """מרחק ישר (מטרים) בין שתי נקודות lat/lon גולמיות."""
    lat1r = math.radians(lat1); lon1r = math.radians(lon1)
    lat2r = math.radians(lat2); lon2r = math.radians(lon2)
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(a))


def _haversine_m(G: nx.MultiDiGraph, u, v) -> float:
    """מרחק ישר (מטרים) בין שני צמתים — היוריסטיקה admissible ל-A*."""
    return _haversine_latlon(G.nodes[u]["y"], G.nodes[u]["x"], G.nodes[v]["y"], G.nodes[v]["x"])


def _route_backtrack(route_result: dict, dest_latlon: tuple) -> float:
    """הפיתול-אחורה המקסימלי של מסלול: אורך הקטע-הרציף הארוך ביותר שבו המסלול
    *מתרחק* מהיעד (במקום להתקרב). 0 = מסלול שמתקדם מונוטונית ליעד. משמש כמדד-חלקות
    בתקרת ה-backoff — מסלול-צל שמתפתל אחורה מדי (מדד גבוה) יורד לאקספוננט שמחליק אותו."""
    pts = route_result.get("route_latlon") or []
    if len(pts) < 2:
        return 0.0
    prev = _haversine_latlon(pts[0][0], pts[0][1], dest_latlon[0], dest_latlon[1])
    worst = 0.0
    run = 0.0
    for lat, lon in pts[1:]:
        cur = _haversine_latlon(lat, lon, dest_latlon[0], dest_latlon[1])
        if cur > prev + 2.0:            # התרחק >2מ' מהיעד (סף רעש)
            run += cur - prev
            worst = max(worst, run)
        else:
            run = 0.0
        prev = cur
    return worst


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

    _t0 = time.monotonic()
    try:
        def _edge_weight(u, v, d):
            # d ב-MultiDiGraph הוא {key: data} של כל הקשתות המקבילות — לא קשת בודדת.
            # לכן ה-fallback חייב לחלץ length דרך d.values() (d.get("length") היה מחזיר
            # תמיד None→50 קבוע, ומזער הופ-קאונט במקום מרחק כשקשת חסרה מ-weight_dict).
            w = weight_dict.get((u, v))
            if w is not None:
                return w
            return min(dd.get("length", 50) for dd in d.values()) * 5

        path = nx.astar_path(
            G, orig_node, dest_node,
            heuristic=lambda u, v: _haversine_m(G, u, v) * 0.5,
            weight=_edge_weight,
        )
    except nx.NetworkXNoPath:
        raise ValueError("לא נמצא מסלול בין הנקודות שהוגדרו")
    _log_timing("A* shaded route search", time.monotonic() - _t0)

    return _summarize_path(G, path, tci_by_uv)


def _weighted_avg_tci(tci_valid: list, seg_lens: list):
    """ממוצע TCI משוקלל לפי אורך המקטע בפועל (מטרים) — לא לפי מספר מקטעים, כדי
    שמקטע ארוך (למשל שדרה) ישפיע על הממוצע יותר ממקטע קצר (למשל סמטה), בהתאם
    לכמה מהמסלול בפועל עובר בו. נופל לממוצע רגיל אם המשקל הכולל 0 (לא צפוי
    במציאות — כל המקטעים אורכם חיובי — אבל מגן מפני חלוקה באפס)."""
    if not tci_valid:
        return None
    if sum(seg_lens) <= 0:
        return float(np.mean(tci_valid))
    return float(np.average(tci_valid, weights=seg_lens))


def _summarize_path(G: nx.MultiDiGraph, path: list, tci_by_uv: dict) -> dict:
    """מסכם מסלול (רשימת צמתים) למדדים: מרחק, זמן, TCI פר-מקטע, ממוצע TCI, ומטרי חשיפה גבוהה.

    `high_exposure_m` = סכום אורכי המקטעים שה-TCI שלהם > HIGH_EXPOSURE_TCI — הבסיס
    לתובנת "דקות חשיפה גבוהה שנחסכו". משותף למסלול המוצל ולמסלול-הבסיס (weight=length)
    כדי שההשוואה תהיה עקבית (אותו מקור TCI, אותה שיטת מדידה)."""
    route_latlon = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

    total_dist    = 0.0
    high_exp_m    = 0.0
    tci_full      = []   # N-1 ערכים, None כשאין TCI לקשת
    tci_valid     = []   # לחישוב avg_tci (משוקלל-אורך)
    tci_valid_len = []   # אורך המקטע המקביל לכל ערך ב-tci_valid
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        seg_len = 0.0
        if G.has_edge(u, v):
            best_k  = min(G[u][v], key=lambda k: G[u][v][k].get("length", 0))
            seg_len = G[u][v][best_k].get("length", 0.0)
        total_dist += seg_len
        tci_val = tci_by_uv.get((u, v)) or tci_by_uv.get((v, u))
        if tci_val is not None:
            val = float(tci_val)
            tci_full.append(val)
            tci_valid.append(val)
            tci_valid_len.append(seg_len)
            if val > HIGH_EXPOSURE_TCI:
                high_exp_m += seg_len
        else:
            tci_full.append(None)

    return {
        "route_latlon":    route_latlon,
        "distance_m":      total_dist,
        "duration_min":    total_dist / WALK_SPEED_MPM,
        "avg_tci":         _weighted_avg_tci(tci_valid, tci_valid_len),
        "tci_list":        tci_full,   # תמיד אורך N-1
        "high_exposure_m": high_exp_m,
    }


_SNAP_MAX_DIST_M = 40.0  # אם הקשת הכי קרובה רחוקה יותר — TCI לא ידוע לאותו מקטע


def _point_to_segment_m(p_lat: float, p_lon: float, a_lat: float, a_lon: float,
                         b_lat: float, b_lon: float) -> float:
    """מרחק (מטרים) מנקודה לקטע a-b, בהקרנה מקומית equirectangular סביב p — מספיק
    מדויק לטווח של עשרות-מאות מטרים (לא לחישובי מרחק-מסלול ארוכים)."""
    coslat = math.cos(math.radians(p_lat))
    ax = (a_lon - p_lon) * 111_320.0 * coslat
    ay = (a_lat - p_lat) * 111_320.0
    bx = (b_lon - p_lon) * 111_320.0 * coslat
    by = (b_lat - p_lat) * 111_320.0
    abx, aby = bx - ax, by - ay
    ab_len2 = abx ** 2 + aby ** 2
    if ab_len2 == 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, (-ax * abx - ay * aby) / ab_len2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(cx, cy)


def _snap_tci_to_latlon_path(route_latlon: list, tci_by_uv: dict,
                              G: nx.MultiDiGraph) -> dict:
    """מצמידה TCI-פר-מקטע למסלול חיצוני (למשל OSRM) ע"י מציאת הקשת הקרובה ביותר
    בגרף המקומי לכל מקטע — כדי לאפשר צביעה/השוואה מול המסלול המוצל, גם כש-
    המקור המקורי (OSRM) לא נושא אנוטציית TCI בעצמו. קירוב גיאומטרי: מדויק ברוב
    הרחובות (שם שתי הרשתות חופפות), פחות מדויק היכן ש-OSRM סוטה מרשת ההליכה
    המקומית (למשל כביש ראשי) — סף _SNAP_MAX_DIST_M מסמן מקרים כאלה כ'לא זמין'.

    ox.distance.nearest_edges מחזיר את הקשת המועמדת הקרובה ביותר (בקירוב, ללא
    התחשבות בעיוות קו-הרוחב על גרף לא-מוקרן) — אבל את מרחק-הסף עצמו מודדים כאן
    בנפרד ב-_point_to_segment_m (מטרים אמיתיים), כי ל-return_dist של osmnx על
    גרף לא-מוקרן אין יחידת-מטרים עקבית (מעלות-קו-אורך/רוחב גולמיות)."""
    if len(route_latlon) < 2:
        return {"avg_tci": None, "tci_list": [], "high_exposure_m": 0.0}
    mid_lats = [(route_latlon[i][0] + route_latlon[i + 1][0]) / 2
                for i in range(len(route_latlon) - 1)]
    mid_lons = [(route_latlon[i][1] + route_latlon[i + 1][1]) / 2
                for i in range(len(route_latlon) - 1)]
    edges = ox.distance.nearest_edges(G, X=mid_lons, Y=mid_lats)
    tci_list, tci_valid, tci_valid_len = [], [], []
    high_exp_m = 0.0
    for i, (u, v, _k) in enumerate(edges):
        seg_len = _haversine_latlon(*route_latlon[i], *route_latlon[i + 1])
        mlat, mlon = mid_lats[i], mid_lons[i]
        edge_dist = _point_to_segment_m(
            mlat, mlon, G.nodes[u]["y"], G.nodes[u]["x"], G.nodes[v]["y"], G.nodes[v]["x"])
        tci_val = None if edge_dist > _SNAP_MAX_DIST_M else (tci_by_uv.get((u, v)) or tci_by_uv.get((v, u)))
        if tci_val is not None:
            val = float(tci_val)
            tci_list.append(val)
            tci_valid.append(val)
            tci_valid_len.append(seg_len)
            if val > HIGH_EXPOSURE_TCI:
                high_exp_m += seg_len
        else:
            tci_list.append(None)
    return {
        "avg_tci":         _weighted_avg_tci(tci_valid, tci_valid_len),
        "tci_list":        tci_list,
        "high_exposure_m": high_exp_m,
    }


def compute_length_route(
    origin_latlon: tuple,
    dest_latlon: tuple,
    tci_by_uv: dict,
    G: nx.MultiDiGraph = None,
) -> dict:
    """מסלול הקצר-ביותר על הגרף (A* weight=length) עם אנוטציית TCI פר-מקטע.

    מסלול-הבסיס להשוואה מול המסלול המוצל: יש לו TCI פר-מקטע (מ-tci_by_uv), בניגוד
    ל-OSRM. משמש לחישוב תובנות (compute_route_insights)."""
    if G is None:
        G = load_graph()
    orig_node = ox.nearest_nodes(G, X=origin_latlon[1], Y=origin_latlon[0])
    dest_node = ox.nearest_nodes(G, X=dest_latlon[1], Y=dest_latlon[0])
    if orig_node == dest_node:
        raise ValueError("נקודת המוצא והיעד קרובות מדי — נסה כתובות מרוחקות יותר")
    try:
        # weight="length" (מחרוזת) — networkx לוקח מינימום על קשתות מקבילות ב-MultiDiGraph.
        # חובה מחרוזת ולא lambda(u,v,d): במולטי-גרף ה-d הוא {key:data} של כל הקשתות, לא
        # קשת בודדת, אז d.get("length") היה מחזיר None→ברירת מחדל אחידה→מזעור צמתים במקום מרחק.
        path = nx.astar_path(
            G, orig_node, dest_node,
            heuristic=lambda u, v: _haversine_m(G, u, v),
            weight="length",
        )
    except nx.NetworkXNoPath:
        raise ValueError("לא נמצא מסלול בין הנקודות שהוגדרו")
    return _summarize_path(G, path, tci_by_uv)


def compute_route_insights(shaded: dict, baseline: dict) -> dict:
    """תובנות השוואה בין המסלול המוצל למסלול-הבסיס הקצר. פונקציה טהורה (ברת-בדיקה).

    מחזיר:
      extra_min                — כמה דקות ארוך יותר המסלול המוצל מהקצר.
      tci_saved                — הפחתת ה-TCI הממוצע (בסיס − מוצל; חיובי = יותר מוצל).
      high_exposure_min_saved  — דקות חשיפה גבוהה (TCI>7) שנחסכו (בסיס − מוצל).
    כל הערכים נגזרים ממדדים שחושבו בקוד (תחזיות מודל ה-TCI + A*), לא מה-LLM."""
    extra_min = shaded["duration_min"] - baseline["duration_min"]
    tci_saved = None
    if shaded.get("avg_tci") is not None and baseline.get("avg_tci") is not None:
        tci_saved = baseline["avg_tci"] - shaded["avg_tci"]
    high_saved_min = (baseline.get("high_exposure_m", 0.0)
                      - shaded.get("high_exposure_m", 0.0)) / WALK_SPEED_MPM
    return {
        "extra_min":               round(extra_min, 1),
        "tci_saved":               round(tci_saved, 1) if tci_saved is not None else None,
        "high_exposure_min_saved": round(high_saved_min, 1),
        "avg_tci_shaded":          (round(shaded["avg_tci"], 1)
                                    if shaded.get("avg_tci") is not None else None),
        "avg_tci_baseline":        (round(baseline["avg_tci"], 1)
                                    if baseline.get("avg_tci") is not None else None),
    }


def shortest_walk_distance(
    origin_latlon: tuple,
    dest_latlon: tuple,
    G: nx.MultiDiGraph = None,
) -> float:
    """אורך המסלול הקצר ביותר (מטרים) בין שתי נקודות — בסיס להשוואת עיקוף.

    רץ על אותו גרף כמו המסלול המוצל (weight="length") כדי שההשוואה תהיה עקבית.
    מחזיר inf אם אין מסלול — כך שתקרת העיקוף לא תיכשל אלא פשוט לא תופעל.
    """
    if G is None:
        G = load_graph()
    orig_node = ox.nearest_nodes(G, X=origin_latlon[1], Y=origin_latlon[0])
    dest_node = ox.nearest_nodes(G, X=dest_latlon[1], Y=dest_latlon[0])
    try:
        return float(nx.shortest_path_length(G, orig_node, dest_node, weight="length"))
    except nx.NetworkXNoPath:
        return float("inf")


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
    on_progress=None,
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

    on_progress(msg: str), אם הועבר (ברירת מחדל None — לא משנה שום התנהגות), נקרא
    לפני כל שלב אמיתי בזרימה — כדי שממשק המשתמש יוכל להציג סטטוס חי (st.status)
    במקום ספינר סתום יחיד, מבלי לשבור את חוזה ה-DI/testability הקיים.

    מחזיר dict:
      route_result — תוצאת המסלול (route_latlon, distance_m, duration_min, avg_tci?)
      color        — צבע לציור (ירוק=מוצל, כחול=מהיר)
      mode         — "shaded" / "fast" בפועל (אחרי fallbacks)
      fallback     — None (הצלחה מלאה) / "model_missing" / "weights_missing" /
                     "night" / "overcast" (נפילה למהיר, סיבות זמינות) /
                     "tier_escalated" (mode="shaded" — הורחב ל"הרבה צל" כדי למצוא מסלול) /
                     "best_effort_over_budget" (mode="shaded" — חורג מהתקציב הרגיל,
                     אך קריר מהמהיר בפועל) /
                     "detour_cap_unreachable" (mode="fast" — אין שום מסלול-צל קריר מהמהיר)
    """
    # on_progress מועבר positional (לא keyword) כדי שיתאים גם ל-geocode_fn
    # מוזרק שהפרמטר הפנימי שלו נקרא אחרת (למשל _on_progress, לצורך cache-key
    # ב-Streamlit) וגם ל-geocode_address המקורית. כשאין on_progress בכלל,
    # geocode_fn נקראת בארגומנט יחיד בדיוק כמו קודם — אפס סיכון לפונקציות
    # מוזרקות ישנות/פשוטות שלא מכירות את הפרמטר.
    if on_progress:
        origin_latlon = geocode_fn(origin, on_progress)
        dest_latlon = geocode_fn(dest, on_progress)
    else:
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

    if on_progress:
        on_progress("בודק תנאי מזג אוויר נוכחיים...")
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
        if on_progress:
            on_progress("מחשב חשיפה לשמש לכל הרחובות...")
        wdict, tci_uv = weights_fn(alt_r, az_r, cloud_r, temp_r, hum_r, shade_r)

    if wdict is None:
        result["route_result"] = compute_route(origin_latlon, dest_latlon)
        result["fallback"] = "weights_missing"
        return result

    if on_progress:
        on_progress("מחשב את המסלול המוצל ביותר...")
    route = compute_shaded_route(
        origin_latlon, dest_latlon, wdict, tci_uv, G=graph)

    # תקרת עיקוף פר-רמה: אם המסלול המוצל ארוך מהתקרה האפקטיבית של הרמה שנבחרה
    # (המחמיר מבין תקרת-האחוזים ותקרת-הדקות שלה, ר' _TIER_DETOUR_CAPS), מחפשים
    # את ה-shade_factor הגבוה ביותר (הכי קרוב להעדפה המקורית) שעדיין עומד בתקרה,
    # ע"י ירידה בצעדים עדינים של 0.1 (לא 0.5!) עד רצפה של 0.1 — לא 0.0 ולא 1.0.
    #
    # למה 0.1 ולא 0.5 (תוקן 2026-07-18): בבדיקה אמפירית על מסלולים אמיתיים
    # התברר שהיחס בין shade_factor למרחק/TCI לא ליניארי ולא חלק — לפעמים יש
    # "נקודת מתיקה" זולה (למשל 0.3: רק 7.5% עיקוף אך כבר קריר מהמסלול המהיר),
    # אבל צעד של 0.5 (1.0→0.5→0.0) קופץ *מעליה* לגמרי, ישר מ"אגרסיבי מדי" ל
    # "עיוור-ל-TCI לחלוטין". צעד של 0.1 (עד 10 ניסיונות) מוצא נקודות-מתיקה כאלה
    # כשהן קיימות, בלי לוותר לגמרי על שיקול-TCI.
    #
    # למה רצפה 0.1 ולא 0.0: ב-shade_factor=0.0 המשקל הופך ל-tci^0=1 קבוע לכל
    # קשת — עיוור לגמרי לחום האמיתי, רק מרחק מוזל לפי רחוב-ירוק/מדרכה. זה מה
    # שיצר מסלולי "צל" קצרים יותר אך חמים יותר מהמסלול המהיר. גם ב-0.1 יש עדיין
    # שיקול-TCI אמיתי (חלש, אך לא אפס).
    #
    # מדרג נפילה (2026-07-18, ערב): תקרת-הדקות הקבועה (+5/+30 דק') מתכווצת
    # יחסית ככל שהטיול ארוך יותר (מעל ~50 דק' היא הופכת למגבילה מהיחס%) —
    # אז רמה שנבחרה יכולה להיכשל בתקציב שלה גם כשעדיין יש צל ממשי זמין ברמה
    # הרחבה יותר. במקום לוותר על צל לגמרי בפעם הראשונה שהרמה הנבחרת נכשלת:
    #   1. מנסים את הרמה שנבחרה (כרגיל).
    #   2. אם נכשלה וזו לא כבר "הרבה צל" — מנסים את "הרבה צל" (התקרה הרחבה
    #      ביותר שיש) לפני שמוותרים. הצלחה כאן → "tier_escalated" (מגולה למשתמש).
    #   3. אם גם הרמה הרחבה ביותר נכשלה — בודקים אם הניסיון הכי טוב שנמצא
    #      (מכל אחת מ-2 הרמות) בכל זאת קריר מהמסלול המהיר. אם כן → מציגים אותו
    #      כ"best_effort_over_budget" (חורג מהתקציב הרגיל, אבל אמיתי ומועיל).
    #      רק אם אף ניסיון לא קריר מהמהיר → נופלים בכנות למסלול מהיר רגיל
    #      (fallback="detour_cap_unreachable") — לעולם לא מסלול "מוצל" שקרי.
    base_dist = shortest_walk_distance(origin_latlon, dest_latlon, G=graph)
    if base_dist in (0.0, float("inf")):
        result["route_result"] = route
        result["color"] = "#27ae60"
        result["mode"] = "shaded"
        result["shade_factor_used"] = shade_r
        result["capped"] = False
        result["tci_uv"] = tci_uv
        return result

    base_time_min = base_dist / WALK_SPEED_MPM

    # cache משותף לשתי קריאות _fit_within_cap (רמה-נבחרת + הסלמה): אותו factor על
    # אותו מזג-אוויר/גרף → אותו מסלול בדיוק, אז לא מחשבים אותו פעמיים.
    _route_memo = {}

    def _fit_within_cap(shade_start, start_route=None):
        """מוצא את ה-shade_factor הגבוה ביותר (על רשת 0.1) שהמסלול שלו עדיין נכנס
        לתקרת-העיקוף האפקטיבית של הרמה, בחיפוש בינארי (~5 חישובי-מסלול) במקום סריקה
        לינארית (~30). מסתמך על מונוטוניות distance(shade_factor): יותר משקל-צל → עיקוף
        ארוך/שווה. מחזיר (route, factor_used, fits_within_cap). רצפה 0.1. cache משותף
        (_route_memo) נמנע מחישוב כפול של אותו decile בין הרמות."""
        ratio_cap, time_cap_min = _TIER_DETOUR_CAPS.get(shade_start, _DEFAULT_DETOUR_CAP)
        eff_cap = (min(ratio_cap, 1 + (time_cap_min / base_time_min))
                   if base_time_min > 0 else ratio_cap)
        thresh = eff_cap * base_dist
        hi_d = int(round(shade_start * 10))   # deciles: shade_factor × 10
        lo_d = 1                               # רצפה = 0.1
        if start_route is not None:
            _route_memo.setdefault(hi_d, start_route)

        def _route_at(d):
            if d not in _route_memo:
                _w, _t = weights_fn(alt_r, az_r, cloud_r, temp_r, hum_r, d / 10.0)
                _route_memo[d] = (None if _w is None else
                                  compute_shaded_route(origin_latlon, dest_latlon, _w, _t, G=graph))
            return _route_memo[d]

        def _fits(d):
            # "נכנס" = גם בתוך תקרת-האורך וגם חלק (backtrack ≤ MAX_BACKTRACK_M). שני
            # התנאים מונוטוניים ב-shade_factor (אקספוננט↑ → אורך↑ ו-backtrack↑), אז
            # החיתוך נשאר downward-closed והחיפוש הבינארי תקין.
            rt = _route_at(d)
            return (rt is not None and rt["distance_m"] <= thresh
                    and _route_backtrack(rt, dest_latlon) <= MAX_BACKTRACK_M)

        # הרמה המלאה כבר נכנסת → אין צורך בחיפוש (המקרה השכיח).
        if _fits(hi_d):
            return _route_at(hi_d), hi_d / 10.0, True

        # חיפוש בינארי על [lo_d, hi_d-1] ל-decile הגבוה ביותר שנכנס.
        best_d, lo, hi = None, lo_d, hi_d - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _fits(mid):
                best_d, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best_d is not None:
            return _route_at(best_d), best_d / 10.0, True

        # אף factor לא נכנס — מחזירים את הרצפה (fits=False), כמו הסריקה הישנה שעצרה ברצפה.
        rt_floor = _route_at(lo_d)
        if rt_floor is None:                       # weights_fn שבור לאורך כל הדרך
            return (_route_at(hi_d) or start_route), hi_d / 10.0, False
        return rt_floor, lo_d / 10.0, (rt_floor["distance_m"] <= thresh)

    # שער רווח-צל: מסלול-הבסיס הישיר מחושב עצלנית פעם אחת. אם המסלול המוצל שנבחר
    # *עוקף* (ארוך מהישיר) אך משפר את ה-TCI הממוצע בפחות מ-MIN_TCI_GAIN — העיקוף
    # אינו שווה (למשל ב-10:00 הכל כבר קריר), ומחזירים את הישיר. אם המסלול המוצל כבר
    # מתכנס לישיר (לא עוקף) — השער לא נוגע בו, מוחזר כרגיל.
    _direct_holder = {}

    def _direct_route():
        if "r" not in _direct_holder:
            try:
                _direct_holder["r"] = compute_length_route(
                    origin_latlon, dest_latlon, tci_uv, G=graph)
            except Exception:
                _direct_holder["r"] = None
        return _direct_holder["r"]

    def _finalize(route_sel, factor_sel, capped_sel, fallback_sel):
        direct = _direct_route()
        if (direct is not None and route_sel is not None
                and route_sel.get("distance_m", 0.0) > direct.get("distance_m", 0.0) + 1.0
                and route_sel.get("avg_tci") is not None and direct.get("avg_tci") is not None
                and (direct["avg_tci"] - route_sel["avg_tci"]) < MIN_TCI_GAIN):
            result["route_result"] = direct
            result["color"] = "#27ae60"
            result["mode"] = "shaded"
            result["shade_factor_used"] = 0.0
            result["capped"] = True
            result["fallback"] = "shade_gain_negligible"
            result["tci_uv"] = tci_uv
            return result
        result["route_result"] = route_sel
        result["color"] = "#27ae60"
        result["mode"] = "shaded"
        result["shade_factor_used"] = factor_sel
        result["capped"] = capped_sel
        if fallback_sel is not None:
            result["fallback"] = fallback_sel
        result["tci_uv"] = tci_uv
        return result

    route1, factor1, fits1 = _fit_within_cap(shade_r, start_route=route)
    if fits1:
        return _finalize(route1, factor1, (factor1 != shade_r), None)

    WIDEST_TIER = 2.0
    best_route, best_factor = route1, factor1
    if shade_r < WIDEST_TIER:
        if on_progress:
            on_progress("מרחיב את תקציב העיקוף לרמת 'הרבה צל'...")
        route2, factor2, fits2 = _fit_within_cap(WIDEST_TIER)
        if fits2:
            return _finalize(route2, factor2, True, "tier_escalated")
        if (route2.get("avg_tci") is not None and
                (best_route.get("avg_tci") is None or route2["avg_tci"] < best_route["avg_tci"])):
            best_route, best_factor = route2, factor2

    if on_progress:
        on_progress("בודק אם יש בכל זאת מסלול קריר יותר מהמהיר...")
    fast_route = compute_route(origin_latlon, dest_latlon, tci_by_uv=tci_uv, G=graph)
    if (best_route.get("avg_tci") is not None and fast_route.get("avg_tci") is not None
            and best_route["avg_tci"] < fast_route["avg_tci"]):
        return _finalize(best_route, best_factor, True, "best_effort_over_budget")

    if on_progress:
        on_progress("לא נמצא מסלול מוצל קריר יותר מהמהיר — עובר למסלול מהיר...")
    result["route_result"] = fast_route
    result["fallback"] = "detour_cap_unreachable"
    result["shade_factor_used"] = best_factor
    result["capped"] = True
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
    tci_list: רשימת TCI פר-קשת לצביעת רמזור (ירוק/כתום/אדום) למסלול המוצל.
    fast_result: תוצאת מסלול מהיר להשוואה — קו דק במסגרת כהה (casing), לא
    מקווקו: קווקו לא נראה טוב כשכל מקטע-TCI הוא PolyLine קצר נפרד (הדפוס
    "מתאפס" בכל מקטע). אם יש לו tci_list משלו (הוצמד ע"י
    _snap_tci_to_latlon_path), הקו הפנימי צבוע לפי אותו סרגל TCI; אחרת כחול קבוע.
    """
    m = folium.Map(location=[TA_LAT_PRECISE, TA_LON_PRECISE], zoom_start=14, tiles="CartoDB positron")

    _FAST_CASING_COLOR  = "#1a1a2e"   # מסגרת כהה קבועה — לא תלוי-TCI, נראית רציפה
    _FAST_CASING_WEIGHT = 5
    _FAST_LINE_WEIGHT   = 2.5

    def _tci_color(tci) -> str:
        if tci is None:
            return "#95a5a6"   # אפור — לא ידוע
        if tci < 4:
            return "#27ae60"   # ירוק — מוצל
        if tci < 7:
            return "#f39c12"   # כתום — ביניים
        return "#c0392b"       # אדום — שמש

    pts = route_result["route_latlon"]
    _shaded_has_tci = bool(tci_list and len(tci_list) == len(pts) - 1)
    if _shaded_has_tci:
        for i, tci in enumerate(tci_list):
            folium.PolyLine(
                pts[i:i + 2],
                color=_tci_color(tci),
                weight=6,
                opacity=0.9,
                tooltip=f"TCI: {tci:.1f}" if tci is not None else "TCI לא זמין",
            ).add_to(m)
    else:
        folium.PolyLine(
            pts, color=color, weight=5, opacity=0.85,
            tooltip=f"מסלול: {route_result['distance_m']:.0f} מ' | {route_result['duration_min']:.0f} דקות",
        ).add_to(m)

    _fast_tci_list = fast_result.get("tci_list") if fast_result else None
    _fast_has_tci = bool(fast_result and _fast_tci_list
                          and len(_fast_tci_list) == len(fast_result["route_latlon"]) - 1)
    if fast_result:
        _fpts = fast_result["route_latlon"]
        if _fast_has_tci:
            for i, tci in enumerate(_fast_tci_list):
                seg = _fpts[i:i + 2]
                folium.PolyLine(seg, color=_FAST_CASING_COLOR,
                                 weight=_FAST_CASING_WEIGHT, opacity=0.55).add_to(m)
                folium.PolyLine(
                    seg,
                    color=_tci_color(tci),
                    weight=_FAST_LINE_WEIGHT,
                    opacity=0.95,
                    tooltip=f"מהיר TCI: {tci:.1f}" if tci is not None else "מהיר: TCI לא זמין",
                ).add_to(m)
        else:
            folium.PolyLine(_fpts, color=_FAST_CASING_COLOR,
                             weight=_FAST_CASING_WEIGHT, opacity=0.55).add_to(m)
            folium.PolyLine(
                _fpts,
                color="#2980b9",
                weight=_FAST_LINE_WEIGHT,
                opacity=0.95,
                tooltip=f"מהיר: {fast_result['distance_m']:.0f} מ' | {fast_result['duration_min']:.0f} דקות",
            ).add_to(m)

    if _shaded_has_tci or _fast_has_tci:
        _fast_legend_line = (
            'מסלול מהיר (קו דק במסגרת כהה, צבוע לפי TCI)' if _fast_has_tci
            else 'מסלול מהיר (OSRM)'
        )
        legend_html = f"""
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
          <span style="display:inline-block;width:16px;height:6px;background:#f39c12;
                border:2px solid {_FAST_CASING_COLOR};border-radius:2px;
                margin-left:5px;vertical-align:middle;"></span>{_fast_legend_line}
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    # קווי-חיבור מהסמן לקצה-המסלול: המסלול מצויר מצמתי-הגרף, והגיאוקוד עשוי ליפול
    # עשרות מטרים מהצומת הקרוב (נמדד ~72מ' במוצא). קו מקווקו דק כדי שהמסלול "יגיע"
    # עד הסמנים במקום להיעצר בפער.
    if pts:
        if _haversine_latlon(*origin_latlon, *pts[0]) > 5:
            folium.PolyLine([origin_latlon, pts[0]], color="#7f8c8d", weight=3,
                            opacity=0.75, dash_array="3 7",
                            tooltip="חיבור להתחלת המסלול").add_to(m)
        if _haversine_latlon(*dest_latlon, *pts[-1]) > 5:
            folium.PolyLine([pts[-1], dest_latlon], color="#7f8c8d", weight=3,
                            opacity=0.75, dash_array="3 7",
                            tooltip="חיבור ליעד").add_to(m)

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
