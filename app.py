"""
SHADY – ניווט עירוני מוצל | Streamlit App
"""

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

try:
    import pysolar.solar as _solar
    _PYSOLAR = True
except ImportError:
    _PYSOLAR = False

try:
    from src.weather import get_current_weather as _get_weather
    _WEATHER = True
except ImportError:
    _WEATHER = False

try:
    from src.routing import (
        geocode_address, compute_route,
        compute_tci_weights, compute_shaded_route,
        build_route_map,
    )
    _ROUTING = True
except ImportError:
    _ROUTING = False

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SHADY",
    layout="wide",
    page_icon="🌤️",
    initial_sidebar_state="collapsed",
)

# ── RTL גלובלי לכל האפליקציה ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* כיוון בסיסי לכל גוף האפליקציה */
        html, body, [class*="stApp"], [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"], [data-testid="stMarkdownContainer"] {
            direction: rtl;
            text-align: right;אקלם
            unicode-bidi: plaintext;
        }

        /* כותרות, פסקאות ורשימות */
        h1, h2, h3, h4, h5, h6, p, li, span, label, div {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
        }

        /* טאבים — לסדר את הטאבים מימין לשמאל */
        [data-testid="stTabs"] [role="tablist"] {
            direction: rtl;
        }

        /* רכיבי קלט (slider, multiselect, radio) — להשאיר טבעיים */
        [data-baseweb="select"], [data-baseweb="input"], [data-baseweb="slider"] {
            direction: ltr;
        }

        /* טבלאות — נתונים נשארים LTR */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            direction: ltr;
        }

        /* קוד וגרפים — LTR */
        pre, code, .stPlotlyChart, .stPyplotChart {
            direction: ltr;
            text-align: left;
        }

        /* captions */
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
            direction: rtl;
            text-align: right;
        }

        /* תיבות info/warning/success/error */
        [data-testid="stAlert"] > div,
        [data-testid="stAlertContainer"] > div {
            direction: rtl;
            text-align: right;
        }

        /* metrics */
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            direction: rtl;
            text-align: right;
        }

        /* subheader ו-markdown בתוך עמודות */
        [data-testid="stColumn"] [data-testid="stMarkdownContainer"] p {
            direction: rtl;
            text-align: right;
        }

        /* link buttons */
        [data-testid="stLinkButton"] {
            direction: rtl;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── טעינת נתונים ──────────────────────────────────────────────────────────────
@st.cache_data
def load_buildings() -> pd.DataFrame:
    return pd.read_csv("data/buildings_clean.csv")

@st.cache_data
def load_trees() -> pd.DataFrame:
    df = pd.read_parquet("data/national_canopy_clean.parquet")
    return df[["lat", "lon", "canopy_area_m2", "area_class"]].dropna(subset=["lat", "lon"])

@st.cache_data
def load_trees_full() -> pd.DataFrame:
    # טוען את כל העמודות המספריות + הקטגוריות לטאב ה-EDA
    df = pd.read_parquet("data/national_canopy_clean.parquet")
    return df[["OBJECTID", "canopy_perimeter_m", "canopy_area_m2",
               "lon", "lat", "Geometry_Type", "area_class"]].copy()

buildings = load_buildings()
trees = load_trees()

# מזג אוויר — נשמר ב-cache ל-5 דקות כדי שלא ייקרא HTTP request בכל rerun
@st.cache_data(ttl=300, show_spinner=False)
def _get_weather_cached() -> dict:
    if _WEATHER:
        return _get_weather()
    return {"temperature": 27.0, "humidity": 65.0, "cloud_cover": 30.0, "source": "default"}

if _ROUTING:
    @st.cache_data(show_spinner=False)
    def _geocode_cached(address: str) -> tuple:
        return geocode_address(address)

# ── מדגם שכונת הצפון הישן (למפה בלבד) ────────────────────────────────────────
_ON_LAT_MIN, _ON_LAT_MAX = 32.076, 32.100
_ON_LON_MIN, _ON_LON_MAX = 34.762, 34.783

@st.cache_data
def load_buildings_old_north() -> pd.DataFrame:
    b = load_buildings()
    mask = (
        b["lat"].between(_ON_LAT_MIN, _ON_LAT_MAX)
        & b["lon"].between(_ON_LON_MIN, _ON_LON_MAX)
    )
    return b[mask].reset_index(drop=True)

@st.cache_data
def load_trees_old_north() -> pd.DataFrame:
    t = load_trees()
    mask = (
        t["lat"].between(_ON_LAT_MIN, _ON_LAT_MAX)
        & t["lon"].between(_ON_LON_MIN, _ON_LON_MAX)
    )
    result = t[mask].reset_index(drop=True)
    if len(result) > 3000:
        result = result.sample(n=3000, random_state=42).reset_index(drop=True)
    return result

buildings_old_north = load_buildings_old_north()
trees_old_north = load_trees_old_north()


@st.cache_data
def build_tci_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """בונה N דוגמאות TCI.

    אם edges_features.parquet קיים (נוצר ע"י precompute_features.py) — משתמש
    ב-mean_building_height ו-tree_canopy_ratio האמיתיים לכל קשת.
    אחרת — fallback לדגימה מהתפלגויות גלובליות.
    """
    import json as _j
    from pathlib import Path as _Path
    rng = np.random.default_rng(seed)

    _edges_path = _Path("data/edges_features.parquet")
    if _edges_path.exists():
        # נתונים מחושבים לפי קשת אמיתית (Spatial Join) — דגימת שורה שלמה לשמירת הצמד
        _ef = pd.read_parquet(_edges_path, columns=["mean_building_height", "tree_canopy_ratio", "street_azimuth"])
        _ef = _ef.fillna(0)
        _idx = rng.choice(len(_ef), size=n, replace=True)  # אותו אינדקס = אותו רחוב
        bh        = _ef["mean_building_height"].values[_idx]
        cr        = _ef["tree_canopy_ratio"].values[_idx]
        street_az = _ef["street_azimuth"].values[_idx]  # 0°–180°, כיוון הרחוב
    else:
        # fallback — דגימה מהתפלגות גלובלית (לפני הרצת precompute_features.py)
        bh        = rng.choice(load_buildings()["height"].values, size=n, replace=True)
        cr        = np.clip(
            rng.choice(load_trees_full()["canopy_area_m2"].values, size=n, replace=True) / 500,
            0, 1,
        )
        street_az = rng.uniform(0, 180, n)

    # מיקום שמש — PySolar מחזיר זוגות (altitude, azimuth) לת"א
    if _PYSOLAR:
        _sun_pool = []
        for _mo in range(1, 13):
            for _hr in range(0, 24):
                try:
                    _dt = datetime(2024, _mo, 15, _hr, tzinfo=timezone.utc)
                    _alt = _solar.get_altitude(32.08, 34.77, _dt)
                    if _alt > 10:
                        _az = _solar.get_azimuth(32.08, 34.77, _dt)
                        _sun_pool.append((float(_alt), float(_az)))
                except Exception:
                    pass
        if _sun_pool:
            _chosen = rng.integers(0, len(_sun_pool), size=n)
            sa     = np.array([_sun_pool[i][0] for i in _chosen])
            sun_az = np.array([_sun_pool[i][1] for i in _chosen])
        else:
            sa     = rng.uniform(5, 72, n)
            sun_az = rng.uniform(0, 360, n)
    else:
        _pre_baked = [(18.1,140.0),(31.4,155.0),(44.2,165.0),(55.3,175.0),
                      (64.1,185.0),(70.2,195.0),(72.1,180.0)]
        _chosen = rng.integers(0, len(_pre_baked), size=n)
        sa     = np.array([_pre_baked[i][0] for i in _chosen])
        sun_az = np.array([_pre_baked[i][1] for i in _chosen])

    # מזג אוויר — מ-12 ערכים חודשיים אמיתיים
    try:
        with open("data/climate_fallback.json", encoding="utf-8") as _cf:
            _clim = _j.load(_cf)
        _temps  = np.array([m["temperature"]       for m in _clim])
        _clouds = np.array([m["cloud_cover"]        for m in _clim])
        _humids = np.array([m.get("humidity", 70)   for m in _clim])
        _idx = rng.integers(0, 12, size=n)
        temp  = _temps[_idx]
        cloud = _clouds[_idx]
        humid = _humids[_idx]
    except Exception:
        temp  = rng.uniform(13, 28, n)
        cloud = rng.uniform(0, 50, n)
        humid = rng.uniform(65, 80, n)

    # shadow_angle: זווית בין כיוון השמש לכיוון הרחוב, 0°–90°
    _sun_dir     = sun_az % 180
    _diff        = np.abs(_sun_dir - street_az)
    shadow_angle = np.minimum(_diff, 180 - _diff)

    # חישוב TCI מהנוסחה האנליטית
    _W1, _W2 = 0.6, 0.4
    sa_rad = np.radians(sa)
    shadow_factor = np.sin(np.radians(shadow_angle))
    bf = np.clip(bh / 30, 0, 1) * np.cos(sa_rad) * shadow_factor
    tci = np.clip(
        1 + 9 * (sa / 80) * (1 - cloud / 100) * (1 - _W1 * cr - _W2 * bf),
        1, 10,
    )

    return pd.DataFrame({
        "sun_altitude":    sa,
        "building_height": bh,
        "canopy_ratio":    cr,
        "cloud_cover":     cloud,
        "temperature":     temp,
        "humidity":        humid,
        "shadow_angle":    shadow_angle,
        "TCI":             tci,
    })


@st.cache_resource
def load_tci_model(path: str = "data/tci_model.joblib"):
    """טוען את מודל ה-TCI השמור (joblib). מחזיר bundle dict, או None אם הקובץ חסר."""
    import joblib
    from pathlib import Path as _P
    if not _P(path).exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_model_results(path: str = "data/model_results.json"):
    """טוען את תוצאות השוואת המודלים (נכתב ע"י `python -m src.model`). None אם חסר."""
    import json
    from pathlib import Path as _P
    if not _P(path).exists():
        return None
    return json.loads(_P(path).read_text(encoding="utf-8"))


@st.cache_data
def load_edges_full():
    """טוען את כל פיצ'רי הקשתות (58k שורות) — לניווט מוצל."""
    import geopandas as _gpd
    from pathlib import Path as _PF
    p = _PF("data/edges_features.parquet")
    if not p.exists():
        return None
    return _gpd.read_parquet(p)


@st.cache_resource(show_spinner="טוען גרף רחובות...")
def _load_nav_graph():
    """טוען את גרף הרחובות לזיכרון — נקרא פעם אחת לכל חיי התהליך."""
    from src.routing import load_graph
    return load_graph()


@st.cache_data(show_spinner=False)
def _precompute_nav_weights(
    sun_alt_r: float, sun_az_r: float,
    cloud_r: float, temp_r: float, hum_r: float,
):
    """
    מחשב TCI ל-58k קשתות ומחזיר (weight_dict, tci_by_uv).
    ממוטמן לפי פרמטרי מזג אוויר מעוגלים — אותם תנאים → cache hit (0ms).
    """
    _bundle   = load_tci_model()
    _edges_df = load_edges_full()
    if _bundle is None or _edges_df is None:
        return None, None
    return compute_tci_weights(
        _edges_df, _bundle,
        float(sun_alt_r), float(sun_az_r),
        float(cloud_r), float(temp_r), float(hum_r),
    )


@st.cache_data
def load_rothschild_edges():
    """קורא את edges_features ומסנן לאזור רוטשילד — ממוטמן כדי לא לקרוא 58k שורות בכל ריצה."""
    import geopandas as _gpd
    from pathlib import Path as _PF
    p = _PF("data/edges_features.parquet")
    if not p.exists():
        return None
    ef = _gpd.read_parquet(p)
    s = ef.cx[34.762:34.792, 32.054:32.076].copy()       # רוטשילד–לב העיר
    if len(s) == 0:
        s = ef.cx[34.774:34.800, 32.083:32.102].copy()   # fallback: הצפון הישן
    return s


# ── טעינה מוקדמת של גרף הרחובות ──────────────────────────────────────────────
# נטען כאן (לפני הטאבים) כדי שה-cache יהיה חם עד שהמשתמש יגיע לטאב הניווט.
# בגלל @st.cache_resource ו-module-level cache ב-routing.py, רץ פעם אחת בלבד.
if _ROUTING:
    _load_nav_graph()

# ── טאבים ──────────────────────────────────────────────────────────────────────
tab_problem, tab_lit, tab_market, tab_eda, tab_map, tab_nav, tab_about = st.tabs([
    "🎯 למידת הבעיה", "📚 סקירת ספרות", "🏪 סקר שוק", "📊 ניתוח נתונים",
    "🗺️ מפה", "🚶 ניווט", "ℹ️ אודות"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — PROBLEM DISCOVERY (M2 · רכיב 1 מתוך 4)
# ══════════════════════════════════════════════════════════════════════════════
with tab_problem:
    st.title("🎯 למידת הבעיה — זיהוי החולשה")
    st.caption("רכיב 1 מתוך 4 בדשבורד M2 ")

    # ── הבעיה במשפט אחד ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#fdecea;border-right:6px solid #c0392b;padding:18px 22px;border-radius:8px;margin:10px 0 22px 0;">
        <div style="font-size:14px;color:#7b241c;font-weight:600;margin-bottom:6px;">הבעיה במשפט אחד</div>
        <div style="font-size:20px;line-height:1.5;color:#1c1c1c;">
        בקיץ הישראלי הליכה בעיר הופכת לסיכון בריאותי —
        והולכי הרגל נאלצים לנווט בשמש הקופחת, בלי התחשבות בצל או נוחות
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── מפת סטייקהולדרים ──────────────────────────────────────────────────────
    st.subheader("🗺️ מפת סטייקהולדרים — מי מושפע מהבעיה?")

    sh_col1, sh_col2, sh_col3 = st.columns(3)
    with sh_col1:
        st.markdown(
            """
            **🚶 משתמשי קצה**
            - הולכי רגל בכלל
            - קשישים (65+)
            - הורים עם עגלות
            - נשים בהריון
            - עובדים בדרך 
            - בעלי כלבים 
            - תיירים
            """
        )
    with sh_col2:
        st.markdown(
            """
            **🏛️ גורמים מוסדיים**
            - עיריית ת"א — אגף תחבורה
            - עיריית ת"א — אגף קיימות
            - משרד הבריאות
            - קופות חולים (מד"א)
            - חברות ביטוח בריאות
            - מעסיקים גדולים בעיר
            """
        )
    with sh_col3:
        st.markdown(
            """
            **💼 גורמים עסקיים / מתחרים**
            - Google Maps · Citymapper · Strava
            - חברות תחבורה ציבורית
            - שירותי Mobility-as-a-Service
            - חברות תיירות וטיולים
            - חברות אופניים וקורקינטים שיתופיים
            - אפליקציות כושר (Strava)
            - חברות פרסום חוצות
            """
        )

    st.divider()

    # ── הפרסונה ────────────────────────────────────────────────────────────────
    st.subheader("👤 פרסונה ראשית")

    p_col_img, p_col_txt = st.columns([1, 2])
    with p_col_img:
        st.image(
            "data/screenshots/suffering_noa.png",
            use_container_width=True,
        )
        st.markdown(
            """
            <div style="background:#fdf6e3;border:1px solid #e6d9a7;border-radius:0 0 10px 10px;
                        padding:10px 18px 14px 18px;text-align:center;margin-top:-6px;">
            <div style="font-size:18px;font-weight:700;color:#7d6608;">נועה, 28</div>
            <div style="font-size:13px;color:#7d6608;">עובדת הייטק · לבקנית</div>
            <div style="font-size:12px;color:#7d6608;margin-top:4px;">תחנת השלום → רוטשילד · 20 דק' רגל</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col_txt:
        st.markdown(
            """
            **הקשר:** נועה היא צעירה בת 28, **לבקנית**, גרה במרכז ת"א ללא רכב.
            מגיעה כל בוקר ברכבת לתחנת השלום וצועדת כ-**20 דקות לעבודה בשדרות רוטשילד**.
            אוהבת לשלב הליכה בשגרת היום ולשמור על אורח חיים בריא — ויוקר המחיה
            ממילא לא מאפשר לה לקחת מונית כל יום.

            **כאב יומי בקיץ:** בתור לבקנית, חשיפה ישירה לשמש בקיץ הישראלי **אינה
            אי-נוחות — היא סיכון בריאותי ממשי**: כוויות UV, סיכון מוגבר לסרטן עור,
            ועומס תרמי קיצוני. אמצע אוגוסט בשעה 13:00 היא הסיוט שלה, אבל היא עדיין
            צריכה להגיע לעבודה.

            **מה היא עושה היום בלי פתרון:**
            פותחת את Google Maps — הוא מציע *את אותו מסלול קצר ביותר*,
            ללא קשר לשעת היום, לזווית השמש או לחופת העצים. אין לה דרך לדעת מראש
            איזה צד של הרחוב מוצל או באיזה רחוב יש חופת עצים צפופה.

            וכך נועה נאלצת לבחור בין שתי אפשרויות גרועות: **לצאת מהבית עם מטרייה
            ובגדים ארוכים** דווקא בעונה שכולנו רוצים — ואמורים — להסתובב בה ברחוב
            "בקלילות"; או **לוותר על ההגנה הזו ולפגוע בבריאותה** מול ה-UV והעומס
            התרמי. בשורה התחתונה — **Google Maps הוא כרגע האופציה הטובה ביותר
            שעומדת לרשותה, והוא ממש לא מספיק.**
            """
        )

    st.divider()

    # ── מסע משתמש: לפני / אחרי ────────────────────────────────────────────────
    st.subheader("🛣️ מסע המשתמש — תרחיש: אמצע אוגוסט, 13:00")

    tab_before, tab_after = st.tabs(["😓 לפני · ללא SHADY", "😎 אחרי · עם SHADY"])

    with tab_before:
        st.graphviz_chart(
            """
            digraph BEFORE {
                rankdir=LR;
                bgcolor="transparent";
                node [shape=box, style="rounded,filled", fillcolor="#fadbd8",
                      color="#c0392b", fontname="Arial", fontsize=11];
                edge [color="#c0392b", fontname="Arial", fontsize=9];

                A [label="12:50\\nסיימה סידורי בוקר\\nצריכה להגיע למשרד"];
                B [label="13:00\\nGoogle Maps: 20 דק'\\nמסלול קצר · אין מידע על שמש"];
                C [label="13:00–13:20\\nמדרכות חשופות לחלוטין\\n33°C · UV index 10"];
                D [label="13:20\\nמגיעה מיוזעת ותשושה\\nעור אדום מהשמש"];
                E [label="13:35\\n15 דק' מנוחה במשרד\\nלפני שיכולה לעבוד"];
                F [label="ערב\\nסימני כוויית UV\\nבידיים ובפנים"];
                G [label="לאורך הקיץ\\nסיכון מצטבר\\nלנזק עורי קבוע"];

                A -> B -> C -> D -> E -> F -> G;
            }
            """
        )
        st.error(
            "❌ **שורה תחתונה:** נועה משלמת פעמיים — בפרודוקטיביות (15 דק' התאוששות אחרי כל הליכה) "
            "ובבריאות (חשיפה מצטברת ל-UV היא סיכון רפואי, לא חוסר נוחות). הניווט הקיים מתעלם מקיומה של השמש."
        )

    with tab_after:
        st.graphviz_chart(
            """
            digraph AFTER {
                rankdir=LR;
                bgcolor="transparent";
                node [shape=box, style="rounded,filled", fillcolor="#d5f5e3",
                      color="#1e8449", fontname="Arial", fontsize=11];
                edge [color="#1e8449", fontname="Arial", fontsize=9];

                A [label="12:50\\nפותחת SHADY\\nמזינה תחנת השלום → רוטשילד"];
                B [label="12:51\\nSHADY: 23 דק'\\nעד 80% מהמסלול בצל (CoolWalks)"];
                C [label="13:00–13:23\\nחופת עצים + צל בניינים\\nתוספת של 3 דק' בלבד"];
                D [label="13:23\\nמגיעה במצב סביר\\nמתאוששת ב-2 דק'"];
                E [label="13:25\\nחוזרת לעבודה במלוא הקיבולת\\nבלי התאוששות ממושכת"];
                F [label="ערב\\nעור בסדר\\nיום עבודה מלא ובריא"];
                G [label="לאורך הקיץ\\nעד 80% מהמסלול בצל\\n(Sulzer & Bönisch, 2025)"];

                A -> B -> C -> D -> E -> F -> G;
            }
            """
        )
        st.success(
            "✅ **שורה תחתונה:** תוספת של **3 דקות הליכה בלבד** מניבה מסלול שבו "
            "**עד 80% מהדרך מוצלת** (Sulzer & Bönisch, 2025). "
            "נועה מחזירה לעצמה את היכולת ללכת ברגל בקיץ — בלי לבחור בין בריאות לנוחות כלכלית."
        )

    st.divider()

    # ── הצעת הערך ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#eafaf1;border-right:6px solid #1e8449;padding:18px 22px;border-radius:8px;margin:10px 0;">
        <div style="font-size:14px;color:#196f3d;font-weight:600;margin-bottom:6px;">הצעת ערך · במשפט אחד</div>
        <div style="font-size:20px;line-height:1.5;color:#1c1c1c;">
        <b>SHADY</b> הופכת את הליכת הקיץ בעיר ממסלול מתיש למסלול נעים —
        בתוספת של 2–3 דקות בלבד מגיעים ליעד כש<b>עד 80% מהמסלול מוצל</b> (Sulzer &amp; Bönisch, 2025).
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("💡 איך ניתחנו את הבעיה לפני שכתבנו שורת קוד?"):
        st.markdown(
            """
            - **מיפוי אוכלוסיות פגיעות** — לבקנים, חולי לופוס, מקבלי טיפול תרופתי פוטו-סנסיטיבי,
              קשישים, ילדים והורים עם עגלות — כולם חולקים את אותו הכאב ברמות שונות.
            - **השוואת מתחרים** — המתחרים מתעדפים *זמן ומרחק בלבד* ומתעלמים
              מאי החום העירוני. גם Cool Walks Barcelona ו-Shadowmap קיימים, אך אף אחד מהם
              לא ישים בישראל או לא כולל ניווט בפועל.
            - **מיפוי דאטה זמין** — גילינו ש-3 מאגרי מידע ציבוריים פתוחים (עירייה, מפ"י, OSM)
              מכילים את כל המידע הדרוש — אבל הם מבוזרים ובלתי נגישים למשתמש קצה.
            - **תיקוף ההזדמנות** — אין כיום פתרון שמבצע *Data Fusion* בין שכבות GIS סטטיות
              לבין נתוני שמש ומזג אוויר דינמיים, ברזולוציה של מקטע רחוב.
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LITERATURE REVIEW (M2 · רכיב 2 מתוך 4)
# ══════════════════════════════════════════════════════════════════════════════
with tab_lit:
    st.title("📚 סקירת ספרות — מה כבר נחקר ואיפה הפער")
    st.caption("רכיב 2 מתוך 4 בדשבורד M2")

    # ── שאלת המחקר ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div dir="rtl" style="background:#fef5e7;border-right:6px solid #d68910;padding:18px 22px;border-radius:8px;margin:10px 0 22px 0;">
        <div dir="rtl" style="font-size:14px;color:#7d6608;font-weight:600;margin-bottom:6px;">שאלת המחקר</div>
        <div dir="rtl" style="font-size:19px;line-height:1.5;color:#1c1c1c;">
        האם ניתן לחשב בזמן אמת מסלול הליכה עירוני שממזער חשיפה לשמש —
        תוך שילוב גבהי מבנים, חופת עצים, מיקום השמש ומזג אוויר —
        ברזולוציה של מקטע רחוב, בעיר ים-תיכונית?
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        סקרנו את הספרות האקדמית לאורך ארבעת הצירים של SHADY — רשת רחובות, צל מבנים,
        נוחות תרמית וניווט מוצל. המטרה הייתה לאתר מה כבר הוכח בשטח, ולזהות במדויק
        את הפער שעדיין פתוח. המקורות אותרו ב-**Google Scholar**, **arXiv**
        ו-**Nature Scientific Reports**, וכל DOI אומת ידנית.
        """
    )

    st.divider()

    # ── ארבע כרטיסיות פירוט ─────────────────────────────────────────────────────
    st.subheader("📖 פירוט המקורות — לחיצה לקריאה מורחבת")

    with st.expander("📘 **1. Boeing (2017)** — *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks*"):
        st.markdown(
            """
            **ציטוט APA:**
            Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.
            *Computers, Environment and Urban Systems, 65*, 126–139.
            🔗 [https://doi.org/10.1016/j.compenvurbsys.2017.05.004](https://doi.org/10.1016/j.compenvurbsys.2017.05.004)

            **למה זה חשוב ל-SHADY:**
            המאמר מציג את הספרייה שהפכה לסטנדרט בתעשייה לעבודה עם רשתות רחובות עירוניות.
            הוא מצדיק את הבחירה הטכנית שלנו — להוציא את רשת הרחובות של ת"א מ-OpenStreetMap
            ולא לבנות גרף מאפס.

            **מה אנחנו לוקחים:** את כל שכבת ה-NetworkX + אלגוריתם Dijkstra.
            **מה אנחנו מוסיפים:** משקלי קשתות שמבוססים על נוחות תרמית, לא רק על מרחק.
            """
        )

    with st.expander("📗 **2. Park, Guldmann & Liu (2021)** — *Impacts of tree and building shades on the urban heat island*"):
        st.markdown(
            """
            **ציטוט APA:**
            Park, Y., Guldmann, J.-M., & Liu, D. (2021). Impacts of tree and building shades on the urban heat island:
            Combining remote sensing, 3D digital city and spatial regression approaches.
            *Computers, Environment and Urban Systems, 88*, 101655.
            🔗 [https://doi.org/10.1016/j.compenvurbsys.2021.101655](https://doi.org/10.1016/j.compenvurbsys.2021.101655)

            **למה זה חשוב ל-SHADY:**
            המאמר הראשון שכימת בנפרד את התרומה של צל עצים מול צל מבנים לקירור העירוני.
            זה ה**הצדקה האקדמית** לכך שב-SHADY יש שני פיצ'רים נפרדים: `tree_canopy_ratio` ו-`mean_building_height`,
            ולא פיצ'ר אחד מאוחד.

            **תובנה מפתח:** קיים קשר ישיר בין חופת העצים לבין הנוחות התרמית —
            ולכן בעיר ים-תיכונית כמו ת"א, חופת העצים היא קריטית.
            """
        )

    with st.expander("📙 **3. Lin, Tsai, Hwang & Matzarakis (2012)** — *Quantification of the effect of thermal indices and sky view factor on park attendance*"):
        st.markdown(
            """
            **ציטוט APA:**
            Lin, T.-P., Tsai, K.-T., Hwang, R.-L., & Matzarakis, A. (2012). Quantification of the effect of thermal indices
            and sky view factor on park attendance.
            *Landscape and Urban Planning, 107*(2), 137–146.
            🔗 [https://doi.org/10.1016/j.landurbplan.2012.05.011](https://doi.org/10.1016/j.landurbplan.2012.05.011)

            **למה זה חשוב ל-SHADY:**
            המאמר מוכיח שהולכי רגל אכן מתנהגים אחרת בתגובה לחשיפה לשמש — הם **מצביעים ברגליים**
            ובוחרים אקטיבית באזורים מוצלים. זה תוקף את עצם קיומה של הבעיה שלנו: אם אנשים
            לא היו אכפת להם מהשמש, לא היה צורך באפליקציה.

            **תובנה מפתח:** הקשר בין SVF (Sky View Factor) להתנהגות אינו לינארי —
            יש "סף" סביב SVF=0.5 שמעליו אנשים נמנעים אקטיבית מהמקום בשעות חמות.
            זה מצדיק את בחירת הסקאלה שלנו (1-10) .
            """
        )

    with st.expander("📕 **4. Sulzer & Bönisch (2024/2025)** — *CoolWalks: Assessing the potential of shaded routing for active mobility*  ⭐ המאמר הקרוב ביותר"):
        st.markdown(
            """
            **ציטוט APA (גרסה רשמית):**
            Sulzer, M., & Bönisch, S. (2025). CoolWalks for active mobility in urban street networks.
            *Scientific Reports, 15*.
            🔗 [https://doi.org/10.1038/s41598-025-97200-2](https://doi.org/10.1038/s41598-025-97200-2)

            **גרסת arXiv (2024):**
            [https://arxiv.org/abs/2405.01225](https://arxiv.org/abs/2405.01225)

            **למה זה הכי חשוב ל-SHADY:**
            זה המאמר היחיד שפורסם עד היום שבונה אלגוריתם ניווט מבוסס-צל על רשת רחובות עירונית.
            אבל הוא מותיר **ארבעה פערים מהותיים** שאנחנו ממלאות:

            | # | הפער אצלם | מה SHADY עושה אחרת |
            |---|-----------|---------------------|
            | 1 | רק צל מבנים — מתעלם מעצים | כולל שכבת חופת עצים מ-data.gov.il |
            | 2 | מזג אוויר סטטי | Open-Meteo API חי |
            | 3 | Boston / Barcelona / גריד סינתטי | ת"א — עיר ים-תיכונית עם רחובות צרים |
            | 4 | נוסחה אנליטית בלבד | ML על Scikit-Learn לחיזוי TCI + השלמת גבהים חסרים |

            **השורה התחתונה:** הם הוכיחו שהרעיון ישים — אנחנו מוכיחות שהוא ישים **כאן, עכשיו, עם דאטה אמיתי**.
            """
        )

    st.divider()

    # ── טבלת השוואה ──────────────────────────────────────────────────────────────
    st.subheader("📊 טבלת השוואה — דאטה · מודל · תוצאה")

    st.markdown(
        """
        <div dir="rtl">
        <table style="width:100%;border-collapse:collapse;font-size:14px;direction:rtl;text-align:right;">
          <thead>
            <tr style="background:#f4f6f8;">
              <th style="padding:10px 14px;border-bottom:2px solid #ccc;width:7%;text-align:center;">מקור</th>
              <th style="padding:10px 14px;border-bottom:2px solid #ccc;width:31%;">דאטה</th>
              <th style="padding:10px 14px;border-bottom:2px solid #ccc;width:31%;">מודל</th>
              <th style="padding:10px 14px;border-bottom:2px solid #ccc;width:31%;">תוצאה</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background:#fff;border-bottom:1px solid #eee;">
              <td style="padding:12px 14px;border-right:5px solid #2471a3;text-align:center;vertical-align:top;">
                <span style="font-size:22px;">📘</span><br>
                <span style="font-size:11px;font-weight:bold;color:#2471a3;">1</span>
              </td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">OpenStreetMap — רשתות רחובות של 27,009 ערים בארה&quot;ב (צמתים, קשתות, תכונות)</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">אלגוריתמים גרפיים מבוססי NetworkX: Dijkstra, A*, מדדי מרכזיות</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">ספריית קוד פתוח (OSMnx) שמורידה וממדלת רשתות רחובות מכל מקום בעולם תוך שניות</td>
            </tr>
            <tr style="background:#f9faf9;border-bottom:1px solid #eee;">
              <td style="padding:12px 14px;border-right:5px solid #1e8449;text-align:center;vertical-align:top;">
                <span style="font-size:22px;">📗</span><br>
                <span style="font-size:11px;font-weight:bold;color:#1e8449;">2</span>
              </td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">קולומבוס, אוהיו — Landsat 8 תרמי + מודל LiDAR תלת-ממדי + חופת עצים</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">רגרסיה מרחבית (OLS + Spatial Lag) על עוצמת Urban Heat Island</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">צל עצים מפחית UHI ב-0.7°C, צל מבנים ב-0.3°C — שני האפקטים מובהקים ועצמאיים</td>
            </tr>
            <tr style="background:#fff;border-bottom:1px solid #eee;">
              <td style="padding:12px 14px;border-right:5px solid #d68910;text-align:center;vertical-align:top;">
                <span style="font-size:22px;">📙</span><br>
                <span style="font-size:11px;font-weight:bold;color:#d68910;">3</span>
              </td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">3 פארקים במרכז טייוואן — מדידות SVF + מדד PET + ספירת מבקרים שעה-שעה</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">רגרסיה לוגיסטית בין SVF/PET לבין הסתברות נוכחות הולכי רגל</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">כש-SVF &lt; 0.5 (יותר צל), נוכחות הולכי רגל בשעות חמות עולה ב-40–60%</td>
            </tr>
            <tr style="background:#f9faf9;">
              <td style="padding:12px 14px;border-right:5px solid #c0392b;text-align:center;vertical-align:top;">
                <span style="font-size:22px;">📕</span><br>
                <span style="font-size:11px;font-weight:bold;color:#c0392b;">4 ⭐</span>
              </td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">גריד סינתטי + ערים אמיתיות (Boston, Barcelona) — Building footprints + OSM</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">מדד CoolWalkability + פרמטר sun-avoidance על גרף הרחובות</td>
              <td style="padding:12px 14px;line-height:1.7;vertical-align:top;">עד 80% הליכה בצל בתוספת 5–10% למרחק — תלוי קריטית בגיאומטריית העיר</td>
            </tr>
          </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("💡 צבע השוליים מימין תואם לצבע הספר בפירוט המקורות למעלה · ⭐ = המאמר הרלוונטי ביותר ל-SHADY")

    st.divider()

    # ── הלקח האחד מכל מקור ──────────────────────────────────────────────────────
    st.subheader("🎯 הלקח האחד מכל מקור")

    l1, l2 = st.columns(2)
    with l1:
        st.markdown(
            """
            **📘 מ-Boeing למדנו:**
            הבנו באופן סופי ש-OSM היא תשתית מתאימה ליצירת גרף ניווט. OSMnx הוא הסטנדרט —
            וגם נותן לנו מטא-דאטה עשירה (highway, length, geometry) בחינם.

            **📙 מ-Lin למדנו:**
            הולכי רגל באמת בוחרים מסלולים מוצלים — אז יש למי לבנות את האפליקציה.
            בנוסף, הקשר בין נוחות לחשיפה הוא לא לינארי, ולכן סקאלה 1-10 עדיפה
            על מעלות צלזיוס.
            """
        )
    with l2:
        st.markdown(
            """
            **📗 מ-Park למדנו:**
            צל עצים וצל מבנים תורמים בנפרד לקירור — חובה לייצג כל אחד כפיצ'ר נפרד.
            צל עצים אפקטיבי פי 2.3 יותר, ולכן `tree_canopy_ratio` יקבל משקל גבוה במודל.

            **📕 מ-Sulzer & Bönisch למדנו:**
            למרות ש-OSM מייצגת כל רחוב כ-edge בודד בגרף, עלינו להבין כיצד להתחשב
            ברחובות "עבים" — לעיתים יש משמעות לכך שצד אחד של הרחוב מוצל מספיק.
            """
        )

    st.divider()

    # ── הפער ש-SHADY סוגרת ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#eafaf1;border-right:6px solid #1e8449;padding:18px 22px;border-radius:8px;margin:10px 0;">
        <div dir="rtl" style="font-size:14px;color:#196f3d;font-weight:600;margin-bottom:6px;">הפער שאף אחד עדיין לא סגר</div>
        <div dir="rtl" style="font-size:18px;line-height:1.7;color:#1c1c1c;unicode-bidi:embed;text-align:right;">
        שלושת מקורות הבסיס מוכיחים ש-<b>(א)</b> הטכנולוגיה לניתוח רשתות רחובות בשלה,
        <b>(ב)</b> השפעת הצל על נוחות תרמית מכומתת ומובהקת, ו-<b>(ג)</b> הולכי רגל אכן בוחרים אקטיבית במסלולים מוצלים.
        המחקר העדכני ביותר הוכיח שניתוב-מוצל ישים תיאורטית — אבל <b>רק עבור צל מבנים, רק בערים מערביות, וללא דאטה דינמי</b>.
        <br><br>
        <span dir="rtl"><b>\u200FSHADY היא הראשונה שמשלבת את כל ארבעת הצירים:</b>
        רשת רחובות + צל מבנים + חופת עצים + מזג אוויר חי + ML על דאטה ישראלי —
        ברזולוציה של מקטע רחוב בודד.</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── שורת השפעה על הפרויקט ──────────────────────────────────────────────────
    with st.expander("📌 שורת השפעה על הפרויקט — מה הסקירה שינתה אצלנו?"):
        st.markdown(
            """
            - **OSM היא תשתית מתאימה ליצירת גרף ניווט** (Boeing) — נשתמש ב-OSMnx כתשתית.
            - **שני פיצ'רים נפרדים לצל** (Park) — `tree_canopy_ratio` ו-`mean_building_height` ייכנסו למודל בנפרד, ולא כפיצ'ר מאוחד.
            - **סקאלה 1-10 ** (Lin) — הקשר בין חשיפה לנוחות אינו לינארי, סקאלה אינטואיטיבית מתאימה יותר.
            - **המיקום שלנו במפת המתחרים** (Sulzer & Bönisch) — אנחנו לא חופרות באותה אדמה; אנחנו מוסיפות 4 נדבכים שלא קיימים אצלם.
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET SURVEY (M2 · רכיב 3 מתוך 4)
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.title("🏪 סקר שוק — מי כבר עושה משהו דומה")
    st.caption("רכיב 3 מתוך 4 בדשבורד M2")

    # ── הקדמה ──────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div dir="rtl" style="background:#fdf2e9;border-right:6px solid #d35400;padding:18px 22px;border-radius:8px;margin:10px 0 22px 0;">
        <div style="font-size:14px;color:#a04000;font-weight:600;margin-bottom:6px;">מה כבר קיים בשטח?</div>
        <div style="font-size:18px;line-height:1.5;color:#1c1c1c;">
        בחרנו 5 מתחרים — 2 אפליקציות ניווט מסחריות, אפליקציית כושר אחת,
        ו-2 פתרונות חלקיים בתחום הצל שנמצאים בקצה הטכנולוגי.
        המטרה: לא להמציא את הגלגל, ולזהות במדויק את הפער שבו <b>SHADY</b> ייחודית.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 5 כרטיסיות מתחרים ──────────────────────────────────────────────────────
    st.subheader("👀 5 המתחרים — תיאור + לינק חי")

    competitors = [
        {
            "screenshot": "data/screenshots/google_maps.png",
            "name": "Google Maps",
            "type": "מסחרי · ענק טכנולוגיה",
            "url": "https://maps.google.com",
            "ui": "מסך מפה נקי עם חיפוש כתובת, קיצורי דרך לבית/עבודה והיסטוריית חיפוש. סרגלי קטגוריה עליונים.",
            "strength": "כיסוי גלובלי, עברית מלאה, אינטגרציה עם Gmail/Calendar.",
            "weakness": "מתעלם לחלוטין משמש, צל ומזג אוויר. ממליץ אותו מסלול ב-8:00 ו-13:00.",
            "color": "#4285F4",
        },
        {
            "screenshot": "data/screenshots/citymapper.png",
            "name": "Citymapper",
            "type": "מסחרי · בריטי",
            "url": "https://citymapper.com",
            "ui": "מפה עירונית נקייה עם כפתורי Go לפי מצב (הליכה/אופניים/תחב\"צ) וזמן הגעה בולט.",
            "strength": "ממוקד הולכי רגל בעיר, UX חלק, מידע בזמן אמת על כל אמצעי תחבורה.",
            "weakness": "אינו זמין בת\"א. מתעלם ממזג אוויר, שמש וצל — מסלול קבוע ב-8:00 ו-13:00.",
            "color": "#009688",
        },
        {
            "screenshot": "data/screenshots/strava.png",
            "name": "Strava",
            "type": "מסחרי · כושר",
            "url": "https://strava.com",
            "ui": "פיד פעילויות, Heatmap של שבילים פופולריים, ניתוח קצב/גובה/קלוריות.",
            "strength": "קהילה ענקית של הולכי רגל ורצים, Heatmap מבוסס-קהילה מראה מסלולים אנושיים.",
            "weakness": "מיועד לספורטאים — לא לניווט יומי. אין התחשבות בשמש, צל או נוחות תרמית.",
            "color": "#FC4C02",
        },
        {
            "screenshot": "data/screenshots/shadowmap.png",
            "name": "Shadowmap",
            "type": "פתרון חלקי · אוסטריה",
            "url": "https://shadowmap.org",
            "ui": "מפה תלת-ממדית עם דיסקת מצפן וסליידר זמן (לפי שעה ולפי יום בשנה). מצבי VISUALIZE / ANALYZE.",
            "strength": "ויזואליזציה מרהיבה של צל מבנים, גלובלי, חינמי לצפייה.",
            "weakness": "תצוגה בלבד — אין ניווט! אין חופת עצים. לא ממליץ מסלולים.",
            "color": "#6c3483",
        },
        {
            "screenshot": "data/screenshots/coolwalks.png",
            "name": "CoolWalks",
            "type": "אקדמי · POC",
            "url": "https://arxiv.org/abs/2405.01225",
            "ui": "אין UI ציבורי — איורים אקדמיים בלבד. Figure 1 מציג 5 מסלולים חלופיים לפי פרמטר sun-avoidance α.",
            "strength": "אלגוריתם ניווט-מוצל ראשון שפורסם, מתמטיקה חזקה (Sulzer & Bönisch).",
            "weakness": "לא מוצר. רק צל מבנים. נבדק על ערים מערביות. ללא דאטה דינמי.",
            "color": "#1e8449",
        },
    ]

    import os

    def _card(comp):
        c = comp["color"]
        st.markdown(
            f"""<div style="border-top:5px solid {c};border-radius:6px 6px 0 0;
                            padding:10px 12px 6px 12px;background:#fafafa;text-align:center;">
                <div style="font-size:17px;font-weight:700;color:{c};">{comp['name']}</div>
                <div style="font-size:11px;color:#888;">{comp['type']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if os.path.exists(comp["screenshot"]):
            st.image(comp["screenshot"], use_container_width=True)
        else:
            st.markdown(
                f"""<div style="background:#f8f9fa;border:2px dashed #ccc;padding:22px 10px;
                               text-align:center;color:#888;font-size:12px;line-height:1.6;">
                    📷 צילום מסך<br>
                    <code style="direction:ltr;display:inline-block;background:#fff;
                                 padding:2px 5px;border-radius:3px;font-size:10px;margin-top:3px;">{comp['screenshot'].split('/')[-1]}</code>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div style="border:1px solid #e0e0e0;border-top:none;border-radius:0 0 6px 6px;
                            padding:12px 14px;margin-bottom:6px;background:#fafafa;
                            text-align:right;min-height:110px;">
                <div style="font-size:12px;color:#555;margin-bottom:8px;"><b>ממשק:</b> {comp['ui']}</div>
                <div style="font-size:12px;margin-bottom:6px;color:#196f3d;"><b>✓ חוזק:</b> {comp['strength']}</div>
                <div style="font-size:12px;color:#922b21;"><b>✗ חולשה:</b> {comp['weakness']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.link_button(f"🔗 לאתר {comp['name']}", comp["url"], use_container_width=True)

    st.markdown("##### 📱 פתרונות ניווט מסחריים")
    row1 = st.columns(3)
    for i in range(3):
        with row1[i]:
            _card(competitors[i])

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
    st.markdown("##### 🔬 פתרונות חלקיים / אקדמיים")
    _, col_b1, col_b2, _ = st.columns([0.5, 1, 1, 0.5])
    for i, col in enumerate([col_b1, col_b2]):
        with col:
            _card(competitors[3 + i])

    st.divider()

    # ── טבלת השוואה ────────────────────────────────────────────────────────────
    st.subheader("📊 טבלת השוואה — פיצ'רים · מחיר · קהל · ייחוד")

    market_data = [
        {
            "מתחרה": "Google Maps",
            "פיצ'רים עיקריים": "ניווט רב-מודלי, 3 מסלולים מוצעים, תחבורה ציבורית",
            "מחיר": "חינם (פרסומות)",
            "קהל יעד": "כללי — מיליארדי משתמשים",
            "נוחות תרמית": "❌ אין",
            "דאטה דינמי": "פקקים בלבד",
            "חולשה מרכזית": "מתעלם משמש לחלוטין",
        },
        {
            "מתחרה": "Citymapper",
            "פיצ'רים עיקריים": "ניווט עירוני מולטי-מודלי, כפתורי Go, אינטגרציה עם תח\"צ בזמן אמת",
            "מחיר": "חינם / Premium £2.99/חודש",
            "קהל יעד": "הולכי רגל ורוכבי אופניים בעיר",
            "נוחות תרמית": "❌ אין",
            "דאטה דינמי": "תחבורה ציבורית בזמן אמת",
            "חולשה מרכזית": "לא זמין בת\"א, מתעלם מצל ושמש",
        },
        {
            "מתחרה": "Strava",
            "פיצ'רים עיקריים": "Heatmap מסלולים, ניתוח ביצועים, תכנון מסלולים ספורטיביים",
            "מחיר": "חינם / Premium $11.99/חודש",
            "קהל יעד": "רצים, רוכבי אופניים, הולכי שביל",
            "נוחות תרמית": "❌ אין",
            "דאטה דינמי": "GPS בזמן אמת + Heatmap קהילתי",
            "חולשה מרכזית": "לא מיועד לניווט יומי, מתעלם מנוחות תרמית",
        },
        {
            "מתחרה": "Shadowmap",
            "פיצ'רים עיקריים": "תצוגת צל מבנים תלת-ממדית, סליידר זמן",
            "מחיר": "חינם / Pro $5/חודש",
            "קהל יעד": "אדריכלים, מתכננים, חובבי טכנולוגיה",
            "נוחות תרמית": "⚠️ צל מבנים בלבד",
            "דאטה דינמי": "מיקום שמש בלבד",
            "חולשה מרכזית": "אין ניווט — רק תצוגה",
        },
        {
            "מתחרה": "CoolWalks",
            "פיצ'רים עיקריים": "אלגוריתם ניווט-מוצל אקדמי, קוד פתוח",
            "מחיר": "חינם (לא מוצר)",
            "קהל יעד": "חוקרים בלבד",
            "נוחות תרמית": "✅ חלקי (מבנים)",
            "דאטה דינמי": "❌ סטטי",
            "חולשה מרכזית": "לא נגיש לציבור, ללא עצים",
        },
        {
            "מתחרה": "SHADY (אנחנו)",
            "פיצ'רים עיקריים": "ניווט מוצל מותאם-שעה, מבנים + עצים + מזג אוויר חי",
            "מחיר": "חינם (MVP)",
            "קהל יעד": "הולכי רגל בת\"א — אוכלוסיות רגישות UV",
            "נוחות תרמית": "✅ מלא (TCI 1-10)",
            "דאטה דינמי": "✅ שמש + מזג אוויר + ML",
            "חולשה מרכזית": "כיסוי גיאוגרפי: ת\"א בלבד (MVP)",
        },
    ]
    _comp_colors = {
        "Google Maps": "#4285F4", "Citymapper": "#009688", "Strava": "#FC4C02",
        "Shadowmap": "#6c3483", "CoolWalks": "#1e8449", "SHADY (אנחנו)": "#d35400",
    }
    _headers = ["מתחרה", "פיצ'רים עיקריים", "מחיר", "קהל יעד", "נוחות תרמית", "דאטה דינמי", "חולשה מרכזית"]
    _widths  = ["12%", "24%", "11%", "14%", "8%", "13%", "18%"]
    _hdr = "".join(
        f'<th style="padding:10px 12px;text-align:right;font-weight:600;width:{w};">{h}</th>'
        for h, w in zip(_headers, _widths)
    )
    _rows = ""
    for idx, row in enumerate(market_data):
        name     = row["מתחרה"]
        features = row["פיצ'רים עיקריים"]
        price    = row["מחיר"]
        audience = row["קהל יעד"]
        thermal  = row["נוחות תרמית"]
        dynamic  = row["דאטה דינמי"]
        weakness = row["חולשה מרכזית"]
        is_shady = "SHADY" in name
        bg   = "#eafaf1" if is_shady else ("#ffffff" if idx % 2 == 0 else "#f8f9fa")
        bold = "font-weight:700;" if is_shady else ""
        nc   = _comp_colors.get(name, "#333")
        wc   = "#196f3d" if is_shady else "#922b21"
        _rows += f"""
        <tr style="background:{bg};">
          <td style="padding:10px 12px;border-bottom:1px solid #eee;color:{nc};font-weight:700;">{name}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;{bold}">{features}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;{bold}white-space:nowrap;">{price}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;{bold}">{audience}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;text-align:center;font-size:17px;">{thermal}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;{bold}">{dynamic}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;color:{wc};{bold}">{weakness}</td>
        </tr>"""
    st.markdown(
        f"""<div style="overflow-x:auto;border-radius:8px;border:1px solid #e0e0e0;">
        <table style="width:100%;border-collapse:collapse;font-size:13px;direction:rtl;text-align:right;">
          <thead><tr style="background:#2c3e50;color:white;">{_hdr}</tr></thead>
          <tbody>{_rows}</tbody>
        </table></div>""",
        unsafe_allow_html=True,
    )
    st.caption("💡 השורה הירוקה האחרונה היא SHADY — הפתרון הייחודי שלנו")

    st.divider()

    # ── תרשים מיצוב ────────────────────────────────────────────────────────────
    st.subheader("🎯 תרשים מיצוב — איפה אנחנו על המפה?")

    fig_pos, ax_pos = plt.subplots(figsize=(10, 7))

    positions = [
        ("Google Maps", 3.3, 0.5, "#4285F4", 1200),
        ("Citymapper", 4.1, 0.95, "#009688", 800),
        ("Strava", 1.8, 0.4, "#FC4C02", 700),
        ("Shadowmap", 1.5, 3.8, "#6c3483", 500),
        ("CoolWalks", 1.0, 4.3, "#1e8449", 400),
        ("SHADY", 4.5, 4.7, "#d35400", 1600),
    ]

    for name, x, y, color, size in positions:
        is_us = name == "SHADY"
        ax_pos.scatter(
            x, y, s=size, c=color, alpha=0.88 if is_us else 0.75,
            edgecolors="#2c3e50" if is_us else "white",
            linewidths=3 if is_us else 1.5, zorder=5,
        )
        ax_pos.annotate(
            name, (x, y), xytext=(0, 0),
            textcoords="offset points", ha="center", va="center",
            fontsize=9 if not is_us else 11,
            fontweight="bold" if is_us else "normal",
            color="white",
            zorder=6,
        )

    # מסגרת הרבעים
    ax_pos.axhline(2.5, color="#999", linestyle="--", linewidth=1, alpha=0.5)
    ax_pos.axvline(2.5, color="#999", linestyle="--", linewidth=1, alpha=0.5)

    # תוויות רבעים
    ax_pos.text(0.7, 4.7, "Shade focus,\nstatic data", fontsize=9,
                color="#666", style="italic", ha="left")
    ax_pos.text(4.7, 4.95, "Shade focus,\nreal-time data ✨",
                fontsize=10, color="#d35400", fontweight="bold", ha="right")
    ax_pos.text(0.7, 0.15, "No shade,\nstatic", fontsize=9,
                color="#666", style="italic", ha="left")
    ax_pos.text(4.7, 0.15, "No shade,\nreal-time", fontsize=9,
                color="#666", style="italic", ha="right")

    ax_pos.set_xlim(0, 5.2)
    ax_pos.set_ylim(-0.3, 5.3)
    ax_pos.set_xlabel("Real-time / Dynamic Data Integration  →", fontsize=11)
    ax_pos.set_ylabel("Shade & Thermal Comfort Focus  →", fontsize=11)
    ax_pos.set_title("Competitive Positioning Map — SHADY occupies an empty quadrant",
                     fontsize=12, fontweight="bold", pad=15)
    ax_pos.set_xticks([])
    ax_pos.set_yticks([])
    ax_pos.spines["top"].set_visible(False)
    ax_pos.spines["right"].set_visible(False)
    ax_pos.set_facecolor("#fafafa")

    st.pyplot(fig_pos, use_container_width=True)
    plt.close(fig_pos)

    st.info(
        "💡 **התובנה המרכזית:** הרבע הימני-עליון — שילוב של **דאטה דינמי** עם "
        "**מיקוד בנוחות תרמית** — היה ריק לחלוטין לפני SHADY. "
        "המתחרים המסחריים חזקים בזמן אמת אבל מתעלמים מהשמש; "
        "פתרונות הצל הקיימים סטטיים ולא ניווטיים."
    )

    st.divider()

    # ── תובנות לעיצוב ──────────────────────────────────────────────────────────
    st.subheader("✨ תובנות לעיצוב — מה לאמץ, מה לשנות, מה הייחוד שלנו")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown("#### ✅ מה לאמץ מהמתחרים")
        for _label, _color, _bullets in [
            ("מ-Google Maps", "#4285F4",
             ["ממשק נקי עם מסלול ברור (לא 3 אפשרויות מבלבלות)", "חיפוש כתובת חופשי"]),
            ("מ-Citymapper", "#009688",
             ['כפתור "Go" ישיר ובולט — לחיצה אחת ומתחילים ללכת',
              "הפרדה ויזואלית בין מצבי תנועה (הליכה / אופניים / תח\"צ)"]),
            ("מ-Strava", "#FC4C02",
             ["Heatmap של מסלולים — מראה אילו רחובות אנשים בוחרים בפועל",
              "הצגת נתוני מסלול (זמן, מרחק) כ-overlay על המפה"]),
            ("מ-Shadowmap", "#6c3483",
             ["ויזואליזציית צל כשכבה על המפה", "סליידר זמן לתכנון מראש"]),
        ]:
            _bl = "".join(f"• {b}<br>" for b in _bullets)
            st.markdown(
                f"""<div style="background:#f8f9fa;border-right:4px solid {_color};
                               border-radius:6px;padding:10px 14px;margin-bottom:10px;text-align:right;">
                    <b style="color:{_color};">{_label}:</b><br>
                    <span style="font-size:13px;color:#444;">{_bl}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    with insight_col2:
        st.markdown("#### 🚫 מה לשנות / מה הייחוד שלנו")
        for _title, _body, _color in [
            ("לא לחזור על הטעות של Google Maps",
             "ממליצים אותו מסלול ב-8:00 ו-13:00 — אנחנו נשנה לפי שעת היציאה.",
             "#e74c3c"),
            ("להשלים את החסר ב-Shadowmap",
             "הם מציגים צל אבל לא מנווטים — אנחנו עושים את שני הדברים.",
             "#e74c3c"),
            ("לקחת את CoolWalks למוצר אמיתי",
             "הם נשארו אקדמיים — אנחנו מוציאות לאוויר עם UI נגיש.",
             "#e74c3c"),
            ("🔑 הייחוד היחיד שלנו",
             "חופת עצים + מזג אוויר חי + ML — ביחד. אף אחד מהמתחרים לא משלב את שלושת אלה.",
             "#d35400"),
        ]:
            st.markdown(
                f"""<div style="background:{'#fef9e7' if 'd35400' in _color else '#fdf2f2'};
                               border-right:4px solid {_color};border-radius:6px;
                               padding:10px 14px;margin-bottom:10px;text-align:right;">
                    <b style="color:{_color};">{_title}:</b><br>
                    <span style="font-size:13px;color:#444;">{_body}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── הצהרת מיצוב ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div dir="rtl" style="background:#eafaf1;border-right:6px solid #1e8449;padding:18px 22px;border-radius:8px;margin:10px 0;">
        <div style="font-size:14px;color:#196f3d;font-weight:600;margin-bottom:6px;">הצהרת המיצוב שלנו</div>
        <div dir="rtl" style="font-size:18px;line-height:1.7;color:#1c1c1c;unicode-bidi:embed;text-align:right;">
        <b>\u200FSHADY היא אפליקציית הניווט הראשונה</b> המיועדת ל<b>הולכי רגל בעיר ים-תיכונית</b>,
        שמייצרת מסלול אישי לפי <b>שעת היציאה, גובה המבנים, חופת העצים ומזג האוויר בזמן אמת</b>.
        <br><br>
        בניגוד למתחרים המסחריים — אנחנו לא משלמים בקילומטרים, אנחנו משלמים ב-UV.
        בניגוד לפתרונות הצל הקיימים — אנחנו לא רק מציגים מפה, אנחנו מובילים אותך הביתה.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── תרגיל SCAMPER ──────────────────────────────────────────────────────────
    with st.expander("🧠 תרגיל SCAMPER — איך הגענו לרעיון? (ממפגש 1)"):
        st.markdown(
            """
            SCAMPER הוא תרגיל בריינסטורם שיטתי — שואל 7 שאלות על מוצרים קיימים
            כדי לחשוף הזדמנויות חדשות. ככה הגענו ל-SHADY:

            | אות | משמעות | התובנה שלנו על Google Maps |
            |-----|---------|------------------------------|
            | **S** | מה **נחליף**? | מסלול קצר-ביותר → מסלול **מוצל-ביותר** |
            | **C** | מה **נשלב**? | GIS סטטי (מבנים, עצים) + מזג אוויר חי + ML |
            | **A** | מה **נתאים**? | פילוסופיית Cool Walks Barcelona → לעיר הים-תיכונית של ת"א |
            | **M** | מה **נשנה**? | משקלי קשתות בגרף — לא מרחק, אלא Thermal Comfort Index |
            | **P** | מה **נשתמש אחרת**? | אותה תשתית יכולה לשמש מתכנני ערים לזיהוי "אזורי סיכון תרמי" |
            | **E** | מה **נסלק**? | את הצורך של המשתמש לחשב לבד מתי השמש "מאחור" |
            | **R** | מה **נסדר אחרת**? | במקום "המסלול הקצר ועוקפים שמש" → "המסלול המוצל ועוקפים כביש" |

            **שורה תחתונה:** התרגיל חשף שאין כאן צורך בטכנולוגיה חדשה — אלא ב**שילוב חכם** של
            שכבות שכבר קיימות, סביב משקל-קשת אחר. זה מה שהוביל אותנו להחלטה לבנות על OSMnx
            
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.title("📊 ניתוח נתונים — EDA")
    st.caption("שכבות הנתונים של SHADY: מבנים · עצים · שמש · מזג אוויר")

    st.divider()
    st.subheader("🏢 חלק א׳ — ניתוח נתוני המבנים")
    st.caption("מקור: opendata.tel-aviv.gov.il (עיריית תל אביב) · הורד מאי 2026")
    # ── Metrics ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("מבנים בת\"א", f"{len(buildings):,}")
    m2.metric("גובה ממוצע", f"{buildings['height'].mean():.1f} מ'")
    m3.metric("גובה חציוני", f"{buildings['height'].median():.1f} מ'")
    m4.metric("מבנים > 20 מ'", f"{(buildings['height'] > 20).sum():,}")
    m5.metric("גובה מחושב", f"{(buildings['height_source'] == 'imputed').mean() * 100:.1f}%")

    st.divider()

    # ── גרף 1: היסטוגרמת גבהים + גרף 2: קומות מול גובה ──────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("התפלגות גבהי מבנים")
        fig, ax = plt.subplots(figsize=(6, 4))
        d_rec = buildings.loc[buildings["height_source"] == "recorded", "height"].clip(upper=60)
        d_imp = buildings.loc[buildings["height_source"] == "imputed", "height"].clip(upper=60)
        ax.hist(d_rec, bins=40, color="#2980b9", alpha=0.8, label=f"Recorded ({len(d_rec):,})")
        ax.hist(d_imp, bins=40, color="#e67e22", alpha=0.8, label=f"Imputed ({len(d_imp):,})")
        ax.set_xlabel("Height (m)")
        ax.set_ylabel("Count")
        ax.set_title("Building Height Distribution (clipped at 60m)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.subheader("גבהי מבנים — Boxplot (outliers)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bp = ax2.boxplot(
            buildings["height"].clip(upper=80),
            vert=False, patch_artist=True,
            boxprops=dict(facecolor="#aed6f1"),
            medianprops=dict(color="#1a5276", lw=2),
            flierprops=dict(marker=".", color="#e74c3c", alpha=0.3, markersize=3),
        )
        ax2.set_xlabel("Height (m)")
        ax2.set_title("Building Height Boxplot (clipped at 80m)")
        ax2.axvline(buildings["height"].median(), color="#1a5276", lw=1.5,
                    linestyle="--", label=f"Median {buildings['height'].median():.1f}m")
        ax2.legend(fontsize=8)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── תובנה בולטת ───────────────────────────────────────────────────────────
    le20_pct = (buildings["height"] <= 20).mean() * 100
    median_h = buildings["height"].median()
    st.info(
        f"💡 **תובנה:** למרות הדימוי של ת\"א כ\"עיר הגדולה\" עם קו רקיע גבוה, "
        f"הנתונים מראים שרוב ההשפעה של הצל על המדרכה מגיעה מבניינים בגובה של עד 20 מ': "
        f"הגובה החציוני הוא רק **{median_h:.1f} מ'** (~3 קומות), ו-**{le20_pct:.1f}%** "
        f"מהמבנים נמוכים מ-20 מ'. אלו הרוב המוחלט של המבנים בעיר — והם שמכתיבים את "
        f"ההצללה ברמת הרחוב."
    )

    # ── חלק ב׳: עצים ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🌳 חלק ב׳ — ניתוח נתוני העצים")

    trees_full = load_trees_full()

    st.caption("מקור: data.gov.il (מפ\"י — המרכז למיפוי ישראל) · CC-BY-SA 4.0 · הורד מאי 2026")

    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("פוליגוני עצים", f"{len(trees_full):,}")
    tm2.metric("עמודות", f"{trees_full.shape[1]}")
    tm3.metric("ערכים חסרים", "✅ אין")
    tm4.metric("פורמט", "Parquet")

    st.caption(
        f'סטטיסטיקה תיאורית — שטח חופה (canopy_area_m2): '
        f'חציון {trees_full["canopy_area_m2"].median():.1f} m² · '
        f'ממוצע {trees_full["canopy_area_m2"].mean():.1f} m² · '
        f'מקסימום {trees_full["canopy_area_m2"].max():,.0f} m²'
    )

    tree_col1, tree_col2 = st.columns(2)

    with tree_col1:
        fig_h, ax_h = plt.subplots(figsize=(6, 4))
        hist_bins = np.arange(0, 205, 5)
        ax_h.hist(
            trees_full["canopy_area_m2"], bins=hist_bins,
            color="#27ae60", alpha=0.85, log=True,
            edgecolor="white", linewidth=0.6,
        )
        ax_h.set_xlabel("Canopy Area (m²)")
        ax_h.set_ylabel("Count (log scale)")
        ax_h.set_title("Tree Canopy Area Distribution (log Y)")
        ax_h.set_xlim(0, 200)
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)
        st.info(
            f'💡 **תובנה:** עמודת המטרה היא **שטח החופה** (`canopy_area_m2`) ולא ההיקף. '
            f'ה**הבדל הגדול בין החציון ({trees_full["canopy_area_m2"].median():.1f} m²) '
            f'לממוצע ({trees_full["canopy_area_m2"].mean():.1f} m²)** מעיד על ערכי קיצון '
            f'וטווח רחב מאוד — מערכים קטנים מ-1 ועד ~{trees_full["canopy_area_m2"].max():,.0f} m². '
            f'לכן **נשקול שימוש בסקאלה לוגריתמית** כדי לעבוד טוב יותר עם ההפרשים הגדולים.'
        )

    with tree_col2:
        fig_hex, ax_hex = plt.subplots(figsize=(6, 4))
        hb = ax_hex.hexbin(
            trees_full["lon"], trees_full["lat"],
            C=trees_full["canopy_area_m2"],
            reduce_C_function=np.mean,
            gridsize=50, cmap="Greens", mincnt=5,
            linewidths=0.1, edgecolors="white",
        )
        cb = fig_hex.colorbar(hb, ax=ax_hex, shrink=0.7)
        cb.set_label("Mean canopy area (m²)")
        ax_hex.set_xlabel("Longitude")
        ax_hex.set_ylabel("Latitude")
        ax_hex.set_title("Tel Aviv — Spatial Distribution of Tree Canopy")
        ax_hex.set_aspect(1 / np.cos(np.radians(32.08)))
        st.pyplot(fig_hex, use_container_width=True)
        plt.close(fig_hex)
        st.markdown(
            '<p dir="rtl" style="font-size:0.82rem;color:#888;">'
            '💡 ניתן לראות צפיפות חופה גדולה בעיקר בצפון-מערב העיר — '
            'אזור פארק הירקון והשטחים הירוקים הגדולים.</p>',
            unsafe_allow_html=True,
        )


    # ══════════════════════════════════════════════════════════════════════════
    # חלק ג׳ — ☀️ ניתוח זווית השמש
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("☀️ חלק ג׳ — ניתוח זווית השמש")
    st.markdown(
        '<p style="font-size:13px;color:#888;direction:rtl;text-align:right;margin:0 0 8px 0;">'
        'פיצ\'רים: <code style="direction:ltr;display:inline-block;">sun_altitude, shadow_angle</code>'
        ' &nbsp;·&nbsp; ת"א (32.08°N, 34.77°E) &nbsp;·&nbsp; PySolar</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div dir="rtl" style="background:#fef9e7;border-right:6px solid #f39c12;padding:18px 22px;border-radius:8px;margin:10px 0 22px 0;">
            <div style="font-size:14px;color:#9a7d0a;font-weight:600;margin-bottom:8px;">למה זווית השמש חשובה?</div>
            <div style="font-size:16px;line-height:1.7;color:#1c1c1c;">
            גובה השמש מעל האופק (<b>sun_altitude</b>) הוא הפיצ'ר הישיר ביותר ל-TCI:
            שמש נמוכה = צל ארוך = TCI נמוך (נוח). שמש גבוהה = צל קצר, חשיפה = TCI גבוה.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _hours_sun = list(range(5, 21))
    if _PYSOLAR:
        _summer = datetime(2024, 6, 21, tzinfo=timezone.utc)
        _winter = datetime(2024, 12, 21, tzinfo=timezone.utc)
        _alts_s = [max(0.0, _solar.get_altitude(32.08, 34.77, _summer.replace(hour=h))) for h in _hours_sun]
        _alts_w = [max(0.0, _solar.get_altitude(32.08, 34.77, _winter.replace(hour=h))) for h in _hours_sun]
    else:
        # ערכים pre-baked לת"א (21 יוני / 21 דצמבר)
        _alts_s = [0.0, 5.2, 18.1, 31.4, 44.2, 55.3, 64.1, 70.2, 72.1, 70.1, 64.0, 55.1, 44.0, 31.2, 18.0, 5.1]
        _alts_w = [0.0, 0.0,  7.1, 18.3, 28.2, 35.9, 40.8, 43.1, 40.7, 35.7, 27.9, 18.1,  6.9,  0.0,  0.0, 0.0]

    sun_col1, sun_col2 = st.columns(2)

    with sun_col1:
        st.markdown("**גרף: גובה השמש לאורך היום**")
        fig_sun, ax_sun = plt.subplots(figsize=(6, 4))
        ax_sun.plot(_hours_sun, _alts_s, color="#f39c12", lw=2.5, marker="o", markersize=4, label="Summer (Jun 21)")
        ax_sun.plot(_hours_sun, _alts_w, color="#2980b9", lw=2,   marker="s", markersize=4, linestyle="--", label="Winter (Dec 21)")
        ax_sun.set_xlabel("Hour (UTC)")
        ax_sun.set_ylabel("Sun Altitude (°)")
        ax_sun.set_title("Sun Altitude — Tel Aviv | Summer vs. Winter")
        ax_sun.legend(fontsize=8)
        ax_sun.set_xticks(_hours_sun)
        ax_sun.set_xticklabels([f"{h}:00" for h in _hours_sun], rotation=45, fontsize=7)
        ax_sun.grid(alpha=0.2)
        st.pyplot(fig_sun, use_container_width=True)
        plt.close(fig_sun)

    with sun_col2:
        st.markdown("**טבלה: אורך צל לאורך היום (קיץ)**")
        _H_REF = 10.0  # גובה מבנה ייחוס (מ')
        _sun_rows_html = ""
        for _sh, _sa2 in zip(_hours_sun, _alts_s):
            if _sa2 > 0:
                _slen = _H_REF / np.tan(np.radians(_sa2))
                _slen_txt = f"{min(_slen, 5 * _H_REF):.0f} מ'"
            else:
                _slen_txt = "—"
            _sun_rows_html += (
                f'<tr style="border-bottom:1px solid #eee;">'
                f'<td style="padding:7px 14px;text-align:center;font-family:monospace;font-size:13px;">{_sh}:00</td>'
                f'<td style="padding:7px 14px;text-align:center;font-size:13px;">{_sa2:.1f}°</td>'
                f'<td style="padding:7px 14px;text-align:center;font-size:13px;">{_slen_txt}</td>'
                "</tr>"
            )
        st.markdown(
            f"""<div style="overflow-y:auto;max-height:340px;border:1px solid #ddd;border-radius:6px;">
<table style="width:100%;border-collapse:collapse;">
  <thead><tr style="background:#f4f6f8;position:sticky;top:0;">
    <th style="padding:9px 14px;border-bottom:2px solid #ccc;text-align:center;font-size:13px;">שעה (UTC)</th>
    <th style="padding:9px 14px;border-bottom:2px solid #ccc;text-align:center;font-size:13px;">altitude (°)</th>
    <th style="padding:9px 14px;border-bottom:2px solid #ccc;text-align:center;font-size:13px;">אורך צל (h=10מ')</th>
  </tr></thead>
  <tbody>{_sun_rows_html}</tbody>
</table></div>""",
            unsafe_allow_html=True,
        )
        st.caption("אורך הצל מחושב לפי L = h / tan(altitude) עבור מבנה בגובה 10 מ' (חתוך ל-50 מ').")

    st.info(
        "💡 **תובנה:** בשעות שבהן המודל שלנו \"הכי נחוץ\" (שיא הקיץ, שיא החום ביום), "
        "ההשפעה של המבנים תורגש הרבה פחות בגלל זווית השמש — הצל קצר; "
        "וסביר להניח שההשפעה של העצים ביחס לבניינים תגדל."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # חלק ד׳ — 🌤️ ניתוח מזג האוויר
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🌤️ חלק ד׳ — ניתוח מזג האוויר")
    st.markdown(
        '<p style="font-size:13px;color:#888;direction:rtl;text-align:right;margin:0 0 8px 0;">'
        'פיצ\'רים: <code style="direction:ltr;display:inline-block;">temperature, humidity, cloud_cover</code>'
        ' &nbsp;·&nbsp; Fallback: <code style="direction:ltr;display:inline-block;">data/climate_fallback.json</code>'
        ' &nbsp;·&nbsp; Open-Meteo API</p>',
        unsafe_allow_html=True,
    )

    import json as _json_mod
    try:
        with open("data/climate_fallback.json", encoding="utf-8") as _cf:
            _climate = _json_mod.load(_cf)
    except Exception:
        _climate = []

    if _climate:
        _months_en  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        _temps      = [m["temperature"] for m in _climate]
        _clds       = [m["cloud_cover"] for m in _climate]

        _idx_summer = [4, 5, 6, 7, 8]
        _idx_winter = [11, 0, 1, 2]
        _avg_t_sum  = sum(_temps[i] for i in _idx_summer) / len(_idx_summer)
        _avg_t_win  = sum(_temps[i] for i in _idx_winter) / len(_idx_winter)
        _avg_c_sum  = sum(_clds[i]  for i in _idx_summer) / len(_idx_summer)
        _avg_c_win  = sum(_clds[i]  for i in _idx_winter) / len(_idx_winter)

        wm1, wm2, wm3, wm4 = st.columns(4)
        wm1.metric("🌡️ טמפ' ממוצע קיץ", f"{_avg_t_sum:.1f}°C")
        wm2.metric("🌡️ טמפ' ממוצע חורף", f"{_avg_t_win:.1f}°C")
        wm3.metric("☁️ ענן ממוצע קיץ",   f"{_avg_c_sum:.0f}%")
        wm4.metric("☁️ ענן ממוצע חורף",  f"{_avg_c_win:.0f}%")

        fig_w, axs_w = plt.subplots(1, 2, figsize=(10, 4))

        axs_w[0].bar(range(12), _temps, color=["#e74c3c" if t > 22 else "#2980b9" for t in _temps])
        axs_w[0].set_xticks(range(12))
        axs_w[0].set_xticklabels(_months_en, rotation=45, ha="right", fontsize=8)
        axs_w[0].set_ylabel("°C")
        axs_w[0].set_title("Temperature by Month")
        axs_w[0].axhline(22, color="#e74c3c", lw=1, linestyle="--", alpha=0.5)

        axs_w[1].bar(range(12), _clds,
                     color=["#f1c40f" if c <= 20 else "#95a5a6" for c in _clds])
        axs_w[1].set_xticks(range(12))
        axs_w[1].set_xticklabels(_months_en, rotation=45, ha="right", fontsize=8)
        axs_w[1].set_ylabel("%")
        axs_w[1].set_title("Cloud Cover by Month (≤20% sunny | >20% cloudy)")
        axs_w[1].axhline(20, color="#7f8c8d", lw=1.5, linestyle="--", alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)

        _sunny_count = sum(1 for c in _clds if c <= 20)
        st.info(
            f"💡 **תובנה:** הנתונים מראים בצורה טובה והגיונית את הזמן שבו אופטימלי להשתמש "
            f"ב-SHADY — ת\"א שמשית ב-**{_sunny_count} מתוך 12 חודשים** (cloud_cover ≤ 20%). "
            "חשוב לזכור שהנתונים מציגים כיסוי עננים **ממוצע** לחודש, אך כיסוי עננים גדול מאוד "
            "(למשל ביום מעונן ספציפי) יכול להוריד לגמרי את הצורך בניווט מבוסס צל."
        )
    else:
        st.warning("⚠️ לא ניתן לטעון את data/climate_fallback.json")

    # ══════════════════════════════════════════════════════════════════════════
    # חלק ה׳ — 📐 TCI · פיצ'רים · KPI
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📐 חלק ה׳ — TCI, פיצ'רים ובחירת KPI")

    # ── קופסת הסבר: מה זה TCI ────────────────────────────────────────────────
    st.markdown(
        """
<div dir="rtl" style="background:#eaf2ff;border-right:6px solid #2980b9;padding:18px 22px;border-radius:8px;margin:10px 0 18px 0;">
<div style="font-size:15px;color:#1a5276;font-weight:700;margin-bottom:10px;">🌡️ מה זה TCI — Thermal Comfort Index?</div>
<div style="font-size:15px;line-height:1.8;color:#1c1c1c;">
<b>TCI = ציון 1–10 לכל מקטע רחוב (edge) בגרף OSMnx</b><br>
<span style="color:#27ae60;font-weight:600;">1 = מוצל/נוח</span> &nbsp;|&nbsp; <span style="color:#e74c3c;font-weight:600;">10 = חשיפה מלאה לשמש</span>
<br><br>
<b>נוסחה אנליטית (MVP — לפני ML):</b><br>
<code style="background:#dbeafe;padding:6px 10px;border-radius:4px;font-size:14px;display:inline-block;margin:6px 0;">
TCI = clip( 1 + 9 &times; (sun_altitude/80) &times; (1 &minus; cloud_cover) &times; (1 &minus; 0.6&times;Tree_Factor &minus; 0.4&times;Building_Factor), &nbsp;1, 10 )
</code>
<br>
<code style="background:#dbeafe;padding:4px 10px;border-radius:4px;font-size:13px;display:inline-block;margin:2px 0;">
Tree_Factor = tree_canopy_ratio &nbsp;&nbsp;·&nbsp;&nbsp; Building_Factor = clip(height/30, 0, 1) &times; cos(sun_altitude_rad) &times; sin(shadow_angle_rad)
</code>
<br><br>
<b>פירוש הגורמים:</b>
שמש גבוהה → (alt/80) קרוב ל-1 → TCI עולה (חם) ·
עצים → tree_canopy_ratio גבוה → TCI יורד (w₁=0.6) ·
מבנים → שמש נמוכה → cos גדול → Building_Factor גדול → TCI יורד (w₂=0.4)
<br><br>
<b>בחירת KPI:</b> יעד רציף (1–10) → <b>רגרסיה</b> → מדד: <b>RMSE</b>
(טעות גדולה = הולכי רגל ברחוב שמשי במקום מוצל — עונש גבוה יותר מ-MAE)
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── טבלת פיצ'רים (4 עמודות) ──────────────────────────────────────────────
    _feat_data = [
        ("sun_altitude",         "PySolar",    "0°–75°",  "↑ עולה — שמש גבוהה = חשיפה",   "#e74c3c"),
        ("mean_building_height", "עירייה",     "0–60 מ'", "↓ יורד — מבנה גבוה = צל ארוך", "#27ae60"),
        ('tree_canopy_ratio',    'מפ"י',       "0–1",     "↓ יורד — עצים מסנני קרינה",     "#27ae60"),
        ("temperature",          "Open-Meteo", "13–28°C", "↑ עולה — חום מגביר אי-נוחות",   "#e74c3c"),
        ("cloud_cover",          "Open-Meteo", "0–50%",   "↓ יורד — עננות מסננת קרינה",    "#27ae60"),
        ("humidity",             "Open-Meteo", "65–80%",  "↑ קל — לחות מגבירה תחושת חום",  "#e67e22"),
        ("shadow_angle",          "PySolar+OSM", "0°–90°",  "↓ יורד — שמש ניצבת לרחוב = צל", "#27ae60"),
    ]
    _feat_rows_html = ""
    for _fi, (_fname, _fsrc, _frng, _feff, _fcol) in enumerate(_feat_data):
        _fbg = "#f9fafb" if _fi % 2 else "#ffffff"
        _feat_rows_html += (
            f'<tr style="background:{_fbg};border-bottom:1px solid #eee;">'
            f'<td style="padding:10px 16px;font-family:monospace;font-size:13px;color:#1a5276;white-space:nowrap;">{_fname}</td>'
            f'<td style="padding:10px 12px;font-size:13px;text-align:center;color:#555;white-space:nowrap;">{_fsrc}</td>'
            f'<td style="padding:10px 12px;font-size:13px;text-align:center;font-family:monospace;white-space:nowrap;">{_frng}</td>'
            f'<td dir="rtl" style="padding:10px 16px;font-size:13px;text-align:right;color:{_fcol};font-weight:600;">{_feff}</td>'
            "</tr>"
        )
    st.markdown(
        f"""<div style="border:1px solid #ddd;border-radius:6px;overflow:hidden;margin:8px 0 16px 0;">
<table style="width:100%;border-collapse:collapse;">
  <thead><tr style="background:#1a5276;color:white;">
    <th style="padding:11px 16px;text-align:left;font-size:13px;">פיצ'ר</th>
    <th style="padding:11px 12px;text-align:center;font-size:13px;">מקור</th>
    <th style="padding:11px 12px;text-align:center;font-size:13px;">טווח</th>
    <th style="padding:11px 16px;text-align:right;font-size:13px;direction:rtl;">כש-X עולה, TCI...</th>
  </tr></thead>
  <tbody>{_feat_rows_html}</tbody>
</table></div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── גרף TCI אנליטי + histogram TCI ──────────────────────────────────────
    tci_col1, tci_col2 = st.columns(2)

    # ── נתונים אמיתיים: גבהים מהעירייה, חופת עצים ממפ"י, שמש מ-PySolar, מזג-אוויר מ-Open-Meteo ──
    _tci_df  = build_tci_df()
    _N       = len(_tci_df)
    _cl      = _tci_df["cloud_cover"].values / 100   # fraction לחישוב _cloud_avg בגרף 1
    _tci_syn = _tci_df["TCI"].values
    _W1, _W2 = 0.6, 0.4

    with tci_col1:
        st.markdown('<p dir="rtl" style="font-weight:600;font-size:15px;margin:0 0 2px 0;">גרף 1 — רגישות TCI לזווית השמש לפי תרחיש רחוב</p>', unsafe_allow_html=True)
        st.markdown('<p dir="rtl" style="font-size:0.82rem;color:#888;margin:0 0 8px 0;">כל קו = שילוב עצים + מבנים שונה · עננות ממוצעת שנתית · שעות שמש בלבד (>10°)</p>', unsafe_allow_html=True)
        _sun_range = np.linspace(10, 78, 300)
        _sun_range_rad = np.radians(_sun_range)
        _cloud_avg = float(_cl.mean())
        _scenarios = [
            {"label": "Exposed (0% trees, 0m)",           "tree": 0.0, "height": 0.0,  "color": "#e74c3c", "ls": "-"},
            {"label": "Built, no trees (0% trees, 15m)",  "tree": 0.0, "height": 15.0, "color": "#e67e22", "ls": "--"},
            {"label": "Green blvd (50% trees, 15m)",      "tree": 0.5, "height": 15.0, "color": "#27ae60", "ls": "-"},
        ]
        fig_tci, ax_tci = plt.subplots(figsize=(6, 4))
        for _sc in _scenarios:
            _bf_sc = np.clip(_sc["height"] / 30, 0, 1) * np.cos(_sun_range_rad)
            _tci_line = np.clip(
                1 + 9 * (_sun_range / 80) * (1 - _cloud_avg)
                * (1 - _W1 * _sc["tree"] - _W2 * _bf_sc),
                1, 10,
            )
            ax_tci.plot(_sun_range, _tci_line, label=_sc["label"],
                        color=_sc["color"], ls=_sc["ls"], lw=2)
        ax_tci.fill_between(_sun_range, 7, 10, color="#e74c3c", alpha=0.06)
        ax_tci.axhline(5, color="gray", lw=1, ls=":", alpha=0.5)
        ax_tci.set_xlabel("sun_altitude (°)")
        ax_tci.set_ylabel("TCI (1=cool, 10=hot)")
        ax_tci.set_title("Sensitivity: TCI vs. Sun Altitude — 3 street scenarios")
        ax_tci.legend(fontsize=7, loc="upper left")
        ax_tci.set_xlim(5, 78)
        ax_tci.set_ylim(1, 10)
        ax_tci.grid(alpha=0.2)
        st.pyplot(fig_tci, use_container_width=True)
        plt.close(fig_tci)

    # ── גרף 2: התפלגות TCI — "משתנה היעד" ───────────────────────────────────
    with tci_col2:
        st.markdown('<p dir="rtl" style="font-weight:600;font-size:15px;margin:0 0 2px 0;">גרף 2 — התפלגות TCI (משתנה היעד)</p>', unsafe_allow_html=True)
        st.markdown(f'<p dir="rtl" style="font-size:0.82rem;color:#888;margin:0 0 8px 0;">{_N:,} תצפיות (רחוב × שעה) — מדגם מ-58,360 קשתות OSMnx · שעות שמש בלבד (sun_altitude > 10°)</p>', unsafe_allow_html=True)
        fig_tci_hist, ax_tci_hist = plt.subplots(figsize=(6, 4))
        ax_tci_hist.hist(_tci_syn, bins=36, color="#2980b9", alpha=0.85,
                         edgecolor="white", linewidth=0.5)
        ax_tci_hist.axvline(_tci_syn.mean(), color="#e74c3c", lw=2, linestyle="--",
                            label=f"Mean = {_tci_syn.mean():.2f}")
        ax_tci_hist.axvline(float(np.median(_tci_syn)), color="#8e44ad", lw=2, linestyle=":",
                            label=f"Median = {float(np.median(_tci_syn)):.2f}")
        ax_tci_hist.set_xlabel("TCI  (1 = cool/shaded  ->  10 = fully exposed)", fontsize=10)
        ax_tci_hist.set_ylabel("Count", fontsize=10)
        ax_tci_hist.set_title("TCI Distribution — Tel Aviv Summer", fontsize=11)
        ax_tci_hist.set_xlim(1, 10)
        ax_tci_hist.legend(fontsize=8)
        ax_tci_hist.grid(alpha=0.2)
        st.pyplot(fig_tci_hist, use_container_width=True)
        plt.close(fig_tci_hist)
    st.divider()

    # ── גרף 3: Heatmap קורלציות — כל 7 הפיצ'רים + TCI ──────────────────────
    st.markdown('<p dir="rtl" style="font-weight:600;font-size:15px;margin:0 0 2px 0;">גרף 3 — מטריצת קורלציות: פיצ׳רים מול TCI</p>', unsafe_allow_html=True)
    st.markdown(f'<p dir="rtl" style="font-size:0.82rem;color:#888;margin:0 0 8px 0;">{_N:,} תצפיות (רחוב × שעה) · אדום = קורלציה חיובית (X עולה → TCI עולה) · כחול = שלילית · מבנים + עצים + PySolar + מזג-אוויר</p>', unsafe_allow_html=True)

    _corr_df = _tci_df.rename(columns={
        "sun_altitude": "sun_alt", "building_height": "bldg_h",
        "canopy_ratio": "canopy", "cloud_cover": "cloud",
        "temperature": "temp", "humidity": "humid",
        "azimuth": "azimuth", "TCI": "TCI",
    })
    _corr_syn = _corr_df.corr().round(2)
    _nc = len(_corr_syn)
    fig_corr_syn, ax_corr_syn = plt.subplots(figsize=(9, 7))
    _im_syn = ax_corr_syn.imshow(_corr_syn.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_corr_syn.set_xticks(range(_nc))
    ax_corr_syn.set_yticks(range(_nc))
    ax_corr_syn.set_xticklabels(_corr_syn.columns, rotation=35, ha="right", fontsize=9)
    ax_corr_syn.set_yticklabels(_corr_syn.columns, fontsize=9)
    for _i in range(_nc):
        for _j in range(_nc):
            ax_corr_syn.text(
                _j, _i, f"{_corr_syn.values[_i, _j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(_corr_syn.values[_i, _j]) > 0.5 else "black",
            )
    fig_corr_syn.colorbar(_im_syn, ax=ax_corr_syn, shrink=0.7)
    ax_corr_syn.set_title("Pearson Correlation — features vs. TCI (real data distributions)")
    plt.tight_layout()
    st.pyplot(fig_corr_syn, use_container_width=True)
    plt.close(fig_corr_syn)

    _tci_corrs = _corr_syn["TCI"].drop("TCI").sort_values(key=abs, ascending=False)
    st.markdown(
        '<div dir="rtl" style="background:#e8f4f8;border-right:4px solid #2980b9;padding:12px 16px;'
        'border-radius:6px;font-size:14px;line-height:1.7;">'
        '<b>💡 מטריקה נבחרת: RMSE</b><br>'
        'רגרסיה על יעד רציף (1–10). RMSE מעניש בחומרה על טעויות גדולות — '
        'חיזוי שגוי של רחוב חשוף כמוצל יסכן את המשתמש בפועל.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div dir="rtl" style="background:#fdfefe;border:1px solid #d5d8dc;border-right:5px solid #2980b9;
     padding:20px 24px;border-radius:8px;margin:18px 0 6px 0;line-height:1.9;font-size:14px;color:#1c1c1c;">
<div style="font-size:15px;font-weight:700;color:#1a5276;margin-bottom:12px;">
📋 ניתוח וסיכום — חלק ה'</div>

<b>ניתוח התפלגות וערכי מרכז:</b><br>
 הפיזור הרחב על פני כל הטווח (1–10) מאמת כי הדאטהסט
מייצג שונות גבוהה ומגוון רחב של תתי-מיקרו-אקלים עירוניים ברמת הרחוב.
<br><br>

<b>תובנות הנדסיות מניתוח הקורלציות (Heatmap):</b><br>
<b>1. השפעת זווית השמש (sun_altitude):</b> מתאם חיובי גבוה מאוד בין גובה השמש ל-TCI — זווית קרינה
אנכית ה6יא הגורם הדומיננטי ביותר בהיווצרות עומס תרמי עירוני.<br>
<b>2. אפקט החסימה המבני (building_height):</b> קורלציה שלילית מול משתנה היעד.
ההשפעה גדלה ככל שזווית השמש נמוכה מ-20°, אז מבנים בני 3 קומות ומעלה מצליחים להטיל צל ארוך
האפקטיבי לאורך המדרכה הנגדית — בדיוק כפי שמנוסח ב-Building_Factor = clip(h/30,0,1)×cos(alt_rad).<br>

<b>בחירת KPI ומטריקה למודל:</b><br>
הוחלט לעשות שימוש במטריקת <b>RMSE</b> (Root Mean Squared Error).
הריבוע בנוסחת ה-RMSE מעניש בחומרה טעויות חיזוי גדולות — למשל, מצב בו המודל יחזה בטעות ציון נמוך/בטוח
לרחוב שהוא בפועל חשוף ולוהט. עונש זה מבטיח כי אלגוריתם הניווט (Dijkstra) יקבל משקולות חסינות
ולא יסכן אוכלוסיות רגישות.
</div>
        """,
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # גרף 4 — ייחודי לדומיין: מפת TCI גיאוגרפית על רחובות תל אביב
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🗺️ גרף 4 — ייחודי לדומיין: מפת TCI על שכונת רוטשילד")
    st.markdown(
        '<p dir="rtl" style="font-size:13px;color:#888;margin:0 0 8px 0;">'
        'TCI חושב לכל קשת ברחוב — אזור שדרות רוטשילד ולב העיר (Lev HaIr) · '
        'שעת ייחוס: קיץ 13:00 (sun_altitude=60°, cloud=5%) · '
        'כל הקשתות בתחום גיאוגרפי מוצגות (ללא מדגם) · '
        '<span style="color:#27ae60;font-weight:600;">ירוק TCI≤4</span> · '
        '<span style="color:#f39c12;font-weight:600;">כתום 4–7</span> · '
        '<span style="color:#e74c3c;font-weight:600;">אדום≥7</span></p>',
        unsafe_allow_html=True,
    )

    _ef_s = load_rothschild_edges()
    if _ef_s is not None and len(_ef_s) > 0:
        # ── בקרות: מקור TCI (נוסחה / מודל ML) + בחירת זמן (גובה שמש) ──────────
        _c1, _c2, _c3 = st.columns([1, 1, 1])
        with _c1:
            _tci_source = st.radio(
                "מקור TCI", ["נוסחה אנליטית", "מודל ML"],
                horizontal=True, key="tci_source",
            )
        with _c2:
            _sun_sel = st.number_input(
                "גובה השמש — שעת ייחוס (°)", min_value=10, max_value=78,
                value=60, step=5, key="tci_sun",
            )
        with _c3:
            st.caption("10° = שחר/שקיעה · 45° = בוקר/אחה\"צ · 72° = צהריים קיץ")
        _CLOUD_NOON = 0.05

        _bundle = load_tci_model() if _tci_source == "מודל ML" else None
        if _tci_source == "מודל ML" and _bundle is None:
            st.warning("⚠️ המודל לא נמצא — הרץ `python -m src.model`. מוצגת נוסחה אנליטית.")

        # אזימוט שמש ייחוס — 180° (דרום) = צהריים קיץ אופייני לת"א
        _SUN_AZ_REF = 180.0
        _sun_dir_ref = _SUN_AZ_REF % 180
        _street_az = _ef_s["street_azimuth"].values if "street_azimuth" in _ef_s.columns else np.full(len(_ef_s), 90.0)
        _diff_ref = np.abs(_sun_dir_ref - _street_az)
        _shadow_angle_e = np.minimum(_diff_ref, 180 - _diff_ref)
        _shadow_factor_e = np.sin(np.radians(_shadow_angle_e))

        if _bundle is not None:
            # אפשרות C: חיזוי TCI פר-edge ע"י מודל ה-ML (כל מקטע = שורה)
            _n = len(_ef_s)
            _X = pd.DataFrame({
                "sun_altitude":    np.full(_n, _sun_sel),
                "building_height": _ef_s["mean_building_height"].values,
                "canopy_ratio":    _ef_s["tree_canopy_ratio"].values,
                "cloud_cover":     np.full(_n, _CLOUD_NOON * 100),
                "temperature":     np.full(_n, 30.0),
                "humidity":        np.full(_n, 60.0),
                "shadow_angle":    _shadow_angle_e,
            })[_bundle["features"]]
            _ef_s["tci"] = np.clip(_bundle["model"].predict(_X), 1, 10)
        else:
            # נוסחה אנליטית — גובה שמש מהסליידר, אזימוט ייחוס 180° (דרום)
            _sa_r = np.radians(_sun_sel)
            _bf_e = np.clip(_ef_s["mean_building_height"] / 30, 0, 1) * np.cos(_sa_r) * _shadow_factor_e
            _ef_s["tci"] = np.clip(
                1 + 9 * (_sun_sel / 80) * (1 - _CLOUD_NOON)
                  * (1 - 0.6 * _ef_s["tree_canopy_ratio"] - 0.4 * _bf_e),
                1, 10,
            )

        def _tci_color_map(t):
            if t <= 4:  return "#27ae60"
            if t <= 7:  return "#f39c12"
            return "#e74c3c"

        _m_tci = folium.Map(
            location=[32.065, 34.776], zoom_start=15, tiles="CartoDB positron"
        )

        # GeoJson שכבה אחת במקום ~9800 PolyLine נפרדים — 7x פחות HTML
        _ef_plot = _ef_s[["geometry", "tci", "mean_building_height", "tree_canopy_ratio"]].copy()
        _ef_plot["mean_building_height"] = _ef_plot["mean_building_height"].fillna(0).round(1)
        _ef_plot["tree_canopy_ratio"]    = _ef_plot["tree_canopy_ratio"].fillna(0).round(2)
        _ef_plot["tci"]                  = _ef_plot["tci"].round(1)
        _ef_plot["_color"]               = _ef_plot["tci"].apply(_tci_color_map)

        folium.GeoJson(
            _ef_plot,
            style_function=lambda f: {
                "color":   f["properties"]["_color"],
                "weight":  3,
                "opacity": 0.85,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["tci", "mean_building_height", "tree_canopy_ratio"],
                aliases=["TCI", "גובה מבנה (מ')", "חופת עצים"],
            ),
        ).add_to(_m_tci)

        st_folium(_m_tci, height=540, use_container_width=True, returned_objects=[])
        st.caption(f"מוצגות {len(_ef_s):,} קשתות באזור רוטשילד–לב העיר")
    else:
        st.warning(
            "⚠️ `data/edges_features.parquet` חסר — הרץ `python precompute_features.py` "
            "פעם אחת ליצירת הנתונים המרחביים."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAP (מדגם: הצפון הישן)
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.title("🗺️ שכבות מרחביות – הצפון הישן, תל אביב")
    st.caption(
        f"מדגם שכונתי לצורכי הדגמת הנתונים | "
        f"מבנים: {len(buildings_old_north):,} | "
        f"עצים: {len(trees_old_north):,}"
    )

    # ── מזג אוויר ומיקום שמש ─────────────────────────────────────────────────
    weather = _get_weather_cached()

    if _PYSOLAR:
        now_utc = datetime.now(timezone.utc)
        sun_alt = _solar.get_altitude(32.08, 34.77, now_utc)
        sun_az  = _solar.get_azimuth(32.08, 34.77, now_utc)
        sun_up  = sun_alt > 0
    else:
        sun_alt, sun_az, sun_up = 0.0, 0.0, False

    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("🌡️ טמפרטורה", f"{weather['temperature']:.1f}°C",
               help="Open-Meteo" if weather["source"] == "live" else "ממוצע חודשי (fallback)")
    wc2.metric("💧 לחות יחסית", f"{weather['humidity']:.0f}%")
    wc3.metric("☁️ כיסוי ענן", f"{weather['cloud_cover']:.0f}%")
    if sun_up:
        wc4.metric("☀️ גובה שמש", f"{sun_alt:.1f}°", help=f"אזימוט: {sun_az:.0f}°")
    elif _PYSOLAR:
        wc4.metric("🌙 שמש", "מתחת לאופק")
    else:
        wc4.metric("☀️ שמש", "pysolar לא מותקן")

    st.divider()

    # ── בחירת שכבות ──────────────────────────────────────────────────────────
    col_ctrl, col_map_area = st.columns([1, 5])

    with col_ctrl:
        st.subheader("שכבות")
        show_buildings = st.checkbox("🏢 מבנים (גובה)", value=True)
        show_trees     = st.checkbox("🌳 חופת עצים",   value=True)
        st.divider()
        st.caption("**צבע מבנים לפי גובה:**")
        st.caption("🔵 0–6 מ' | 🔷 6–12 מ'")
        st.caption("🟠 12–20 מ' | 🔴 20–35 מ'")
        st.caption("🟣 35 מ'+")

    def _height_color(h: float) -> str:
        if h < 6:  return "#5dade2"
        if h < 12: return "#2980b9"
        if h < 20: return "#f39c12"
        if h < 35: return "#e74c3c"
        return "#6c3483"

    # cache key: המפה נבנית מחדש רק כשמשנים שכבות או כיוון שמש
    _map_key = f"tab_map_{show_buildings}_{show_trees}_{round(sun_az) if sun_up else 'night'}"
    if _map_key not in st.session_state:
        _m = folium.Map(location=[32.088, 34.772], zoom_start=15, tiles="CartoDB positron")

        if show_buildings:
            b_layer = folium.FeatureGroup(name="מבנים", show=True)
            for _, row in buildings_old_north.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=3,
                    color=_height_color(row["height"]),
                    fill=True,
                    fill_opacity=0.75,
                    weight=0,
                    tooltip=f"{row['height']:.1f} מ'",
                ).add_to(b_layer)
            b_layer.add_to(_m)

        if show_trees:
            t_layer = folium.FeatureGroup(name="חופת עצים", show=True)
            HeatMap(
                trees_old_north[["lat", "lon", "canopy_area_m2"]].values.tolist(),
                radius=8, blur=6, min_opacity=0.3,
            ).add_to(t_layer)
            t_layer.add_to(_m)

        if sun_up:
            sun_layer = folium.FeatureGroup(name="☀️ כיוון שמש", show=True)
            folium.Marker(
                location=[32.088, 34.772],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:26px;'
                        f'transform:rotate({sun_az:.0f}deg);'
                        f'transform-origin:center;">'
                        f"☀️</div>"
                    ),
                    icon_size=(34, 34),
                    icon_anchor=(17, 17),
                ),
                tooltip=f"גובה שמש: {sun_alt:.1f}° | אזימוט: {sun_az:.0f}°",
            ).add_to(sun_layer)
            sun_layer.add_to(_m)

        folium.LayerControl(collapsed=False).add_to(_m)
        st.session_state[_map_key] = _m

    _map = st.session_state[_map_key]

    with col_map_area:
        st_folium(_map, height=620, width=1100)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_nav:
    st.title("🚶 ניווט בתל אביב")

    if not _ROUTING:
        st.error("⚠️ חבילת `osmnx` לא מותקנת — התקן עם `pip install osmnx networkx` והפעל מחדש.")
        st.stop()

    # הגרף כבר נטען לפני הטאבים — הקריאה הזו מחזירה אותו מה-cache (0ms).
    _nav_G = _load_nav_graph()

    route_mode = st.radio(
        "סוג מסלול",
        ["🌿 הכי מוצל (מודל ML)", "⚡ הכי מהיר (OSRM)"],
        horizontal=True,
        index=0,
    )

    c1, c2 = st.columns(2)
    with c1:
        origin_input = st.text_input(
            "📍 נקודת מוצא",
            placeholder="לדוגמה: כיכר רבין, תל אביב",
            key="nav_origin",
        )
    with c2:
        dest_input = st.text_input(
            "🏁 יעד",
            placeholder="לדוגמה: שוק הכרמל, תל אביב",
            key="nav_dest",
        )

    find_btn = st.button("מצא מסלול 🗺️", type="primary")
    st.divider()

    if find_btn:
        if not origin_input.strip() or not dest_input.strip():
            st.warning("⚠️ יש להזין גם נקודת מוצא וגם יעד.")
        else:
            _use_shaded  = route_mode.startswith("🌿")
            _route_color = "#27ae60" if _use_shaded else "#2980b9"
            _spinner_msg = "מחשב מסלול מוצל..." if _use_shaded else "מחשב מסלול..."

            with st.spinner(_spinner_msg):
                error_msg     = None
                route_result  = None
                origin_latlon = None
                dest_latlon   = None
                try:
                    origin_latlon = _geocode_cached(origin_input)
                    dest_latlon   = _geocode_cached(dest_input)

                    if _use_shaded:
                        _bundle   = load_tci_model()
                        _edges_df = load_edges_full()
                        if _bundle is None or _edges_df is None:
                            st.warning("⚠️ מודל ML או קובץ פיצ'רים לא זמין — עובר למסלול מהיר.")
                            route_result = compute_route(origin_latlon, dest_latlon)
                            _route_color = "#2980b9"
                        else:
                            # מזג אוויר ושמש
                            _nav_w = _get_weather_cached()
                            if _PYSOLAR:
                                from datetime import datetime as _dt, timezone as _tz
                                _now = _dt.now(_tz.utc)
                                _nav_sun_alt = float(_solar.get_altitude(32.08, 34.77, _now))
                                _nav_sun_az  = float(_solar.get_azimuth(32.08, 34.77, _now))
                            else:
                                _nav_sun_alt, _nav_sun_az = 45.0, 180.0

                            # עיגול לצורך cache — RF מחושב פעם אחת לכל תנאי מזג אוויר
                            _alt_r   = round(_nav_sun_alt / 5) * 5
                            _az_r    = round(_nav_sun_az  / 10) * 10
                            _cloud_r = round(_nav_w["cloud_cover"] / 10) * 10
                            _temp_r  = round(_nav_w["temperature"] / 5) * 5
                            _hum_r   = round(_nav_w["humidity"]    / 10) * 10

                            _wdict, _tci_uv = _precompute_nav_weights(
                                _alt_r, _az_r, _cloud_r, _temp_r, _hum_r
                            )
                            if _wdict is None:
                                route_result = compute_route(origin_latlon, dest_latlon)
                                _route_color = "#2980b9"
                            else:
                                route_result = compute_shaded_route(
                                    origin_latlon, dest_latlon,
                                    _wdict, _tci_uv,
                                    G=_nav_G,
                                )
                    else:
                        route_result = compute_route(origin_latlon, dest_latlon)

                except ValueError as e:
                    error_msg = f"❌ {e}"
                except Exception as e:
                    error_msg = f"❌ שגיאה: {e}"

            if error_msg:
                st.error(error_msg)
            else:
                _avg_tci = route_result.get("avg_tci")
                _n_cols  = 3 if _avg_tci is not None else 2
                _mc      = st.columns(_n_cols)
                _mc[0].metric("📏 מרחק", f"{route_result['distance_m']:.0f} מ'")
                _mc[1].metric("⏱️ זמן הליכה משוער", f"{route_result['duration_min']:.0f} דקות")
                if _avg_tci is not None:
                    _mc[2].metric("☀️ חשיפה ממוצעת לשמש", f"{_avg_tci:.1f} / 10",
                                  help="TCI: 1 = מוצל לחלוטין · 10 = חשיפה מלאה לשמש")
                route_map = build_route_map(origin_latlon, dest_latlon, route_result,
                                            color=_route_color)
                st_folium(route_map, height=520, width=1100, returned_objects=[])
    else:
        _empty_map = folium.Map(
            location=[32.0853, 34.7818], zoom_start=13, tiles="CartoDB positron"
        )
        st_folium(_empty_map, height=520, width=1100, returned_objects=[])

    st.caption("ניווט מבוסס: OSMnx · Nominatim · NetworkX Dijkstra · RandomForest TCI")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.title("ℹ️ אודות SHADY")

    st.markdown(
        """
        <div style="background:#fdf6e3;border-right:6px solid #d68910;padding:18px 22px;border-radius:8px;margin:10px 0 22px 0;">
        <div style="font-size:14px;color:#7d6608;font-weight:600;margin-bottom:6px;">SHADY — במשפט אחד</div>
        <div style="font-size:18px;line-height:1.5;color:#1c1c1c;">
        <b>SHADY</b> מוצאת מסלולי הליכה מוצלים בתל אביב על בסיס גבהי מבנים, חופת עצים,
        מיקום השמש ומזג אוויר בזמן אמת — ומשתמשת במודל ML לחיזוי מדד הנוחות התרמית לכל קשת ברחוב.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── מקורות נתונים ──────────────────────────────────────────────────────────
    st.subheader("🗄️ מקורות נתונים")

    st.markdown(
        """
        | שכבה | מקור | רשומות |
        |------|------|--------|
        | מבנים | עיריית ת"א (opendata.tel-aviv.gov.il) | 45,783 |
        | חופת עצים | מפ"י (data.gov.il) | 231,234 |
        | רשת רחובות | OSMnx | גרף מלא |
        | מזג אוויר | Open-Meteo API | בזמן אמת |
        """
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # M3 — KPI SELECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🎯 M3 — בחירת KPI: איך עוברים מבעיה למספר אחד?")
    st.caption("בחרנו את מדד ההצלחה של המודל על פי שלושה שלבים מתודיים")

    # ── תיבת ה-KPI הנבחר ───────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#eafaf1;border-right:6px solid #1e8449;padding:20px 24px;border-radius:8px;margin:12px 0 24px 0;">
        <div style="font-size:13px;color:#196f3d;font-weight:600;margin-bottom:8px;">🏁 המסקנה שלנו — תרגיל מהיר</div>
        <div style="font-size:18px;line-height:1.7;color:#1c1c1c;">
        "המודל שלנו הוא <b>מודל רגרסיה</b> ונשתמש ב-<b style="color:#c0392b;">RMSE</b> כי משתנה היעד
        (<em>TCI</em>) הוא מספר רציף (1–10), והמדד מעניש בחומרה טעויות חיזוי גדולות (בריבוע) —
        מה שמבטיח בטיחות להולכי הרגל ומונע שליחתם לרחוב לוהט שנחזה בטעות כמוצל."
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── שלושת השלבים ───────────────────────────────────────────────────────
    st.markdown("#### 🧠 ניתוח לפי 3 שלבים")

    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col3:
        st.markdown(
            """
            <div style="background:#fef9e7;border:2px solid #d4a017;border-radius:10px;padding:18px 16px;min-height:230px;">
            <div style="font-size:22px;font-weight:700;color:#d4a017;margin-bottom:6px;">01</div>
            <div style="font-size:16px;font-weight:700;color:#1c1c1c;margin-bottom:10px;">מה הפלט של המודל?</div>
            <div style="font-size:13px;color:#444;line-height:1.7;">
            <b>→ רגרסיה.</b><br>
            הפלט הוא <em>TCI</em> — מספר רציף בסקאלה 1 עד 10.<br>
            אנחנו רוצות לדעת <em>כמה</em> מוצל הרחוב, לא רק "מוצל / לא מוצל",
            כדי שהמשקולות על גרף Dijkstra יהיו מדויקות ורציפות.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col2:
        st.markdown(
            """
            <div style="background:#fef9e7;border:2px solid #d4a017;border-radius:10px;padding:18px 16px;min-height:230px;">
            <div style="font-size:22px;font-weight:700;color:#d4a017;margin-bottom:6px;">02</div>
            <div style="font-size:16px;font-weight:700;color:#1c1c1c;margin-bottom:10px;">מה עלות של טעות?</div>
            <div style="font-size:13px;color:#444;line-height:1.7;">
            <b>→ טעות גדולה = סיכון בטיחותי.</b><br>
            אם המודל יחזה TCI=4 (נוח) לרחוב שהוא בפועל 9 (לוהט וחשוף),
            אנחנו מסכנות אוכלוסיות פגיעות כמו קשישים ולבקנים.<br>
            <b>RMSE</b> מעלה טעויות בריבוע — מה שנותן עונש ענק לטעויות גדולות ומכריח את המודל להיות שמרן.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col1:
        st.markdown(
            """
            <div style="background:#fef9e7;border:2px solid #d4a017;border-radius:10px;padding:18px 16px;min-height:230px;">
            <div style="font-size:22px;font-weight:700;color:#d4a017;margin-bottom:6px;">03</div>
            <div style="font-size:16px;font-weight:700;color:#1c1c1c;margin-bottom:10px;">איך נראה משתנה היעד?</div>
            <div style="font-size:13px;color:#444;line-height:1.7;">
            <b>→ מתפרש על פני כל הטווח</b><br>
            <b>RMSE</b> נותן אינדיקציה אמיתית על איכות החיזוי בכל הטווח —
            בניגוד למדדי סיווג שהיו מאבדות את הרזולוציה הרציפה.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # ── טבלת מדדים ──────────────────────────────────────────────────────────
    with st.expander("📊 מבט מהיר — טבלת המדדים הנפוצים"):
        st.markdown(
            """
            | סוג | מדד | מתי משתמשים |
            |-----|-----|-------------|
            | רגרסיה | **RMSE** ✅ | טעויות באחידות היעד; מעניש טעויות גדולות (בריבוע) |
            | רגרסיה | MAE | עמיד לחריגים, פשוט להסביר למשתמש |
            | סיווג | Accuracy | רק כשהמחלקות מאוזנות (50/50) |
            | סיווג | F1 | מחלקות לא מאוזנות — מאזן Precision ו-Recall |
            | סיווג | Recall | כש"החמצה" מסוכנת (רפואה, בטיחות) |
            """
        )
        st.info("💡 אנחנו בשורת **רגרסיה + RMSE** — כי TCI הוא מספר רציף ולא קטגוריה.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # M3 — תוצאות: השוואת מודלים ובחירת מנצח (נטען מ-data/model_results.json)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🏆 M3 — תוצאות: השוואת מודלים ובחירת מנצח")
    _res = load_model_results()
    if _res is None:
        st.warning("⚠️ אין תוצאות שמורות — הריצו `python -m src.model` ליצירת `data/model_results.json`.")
    else:
        st.caption(
            f"פיצול: {_res['rows']['train']:,} train / {_res['rows']['test']:,} test · "
            f"KPI: RMSE (נמוך=טוב) · נמדד על test בלבד"
        )
        _tbl = pd.DataFrame(_res["table"]).rename(
            columns={"model": "מודל", "rmse": "RMSE (test)", "r2": "R² (test)"}
        )
        st.dataframe(_tbl, hide_index=True, use_container_width=True)

        _ratio = round(_res["baseline_mean_rmse"] / _res["winner_rmse"], 1)
        st.success(
            f"🏆 **המנצח: {_res['winner']}** — RMSE={_res['winner_rmse']}, R²={_res['winner_r2']} · "
            f"מנצח את רצפת ה-baseline ({_res['baseline_mean_rmse']}) פי {_ratio}."
        )

        st.markdown(
            """
            **למה נבחר Random Forest?**
            - **מנצח את הרצפה** (DummyRegressor שמנבא ממוצע) בפער עצום — מוכיח שהפיצ'רים נושאים מידע.
            - **תופס אי-לינאריות ואינטראקציות** (sun × canopy × building) שמודל לינארי מפספס —
              לכן ה-RMSE שלו נמוך מ-Linear ומ-Decision Tree בודד.
            - **יציב**: ב-5-fold cross-validation קיבל RMSE 0.13 ± 0.01, ומרסן את ה-overfit
              שעץ בודד מפגין (train RMSE=0 אך test 0.21).
            """
        )
        if _res.get("importances"):
            with st.expander("📊 חשיבות הפיצ'רים (RandomForest)"):
                _imp = pd.DataFrame(_res["importances"], columns=["פיצ'ר", "importance"])
                st.dataframe(_imp, hide_index=True, use_container_width=True)
                st.caption(
                    "sun_altitude שולט. temperature שומר חשיבות כי הוא פרוקסי ל-cloud_cover "
                    "(קורלציה ≈ −0.99). shadow_angle = זווית בין כיוון השמש לכיוון הרחוב (0°=מקביל, 90°=ניצב). "
                    "(קורלציה ≠ חשיבות)"
                )

        st.info(
            "🔎 **מהימנות:** ה-R² הגבוה נובע מכך שיעד ה-TCI מחושב כרגע מנוסחה אנליטית "
            "(יעד סינתטי) — המודל משחזר את הנוסחה. זה צעד ביניים מתוכנן; המבחן האמיתי "
            "יגיע עם תוויות אמת מדודות."
        )

    st.divider()

    # ── מפת דרכים ל-M3 ──────────────────────────────────────────────────────
    st.subheader("🗺️ מפת הדרכים שלנו ל-M3")

    rd_col1, rd_col2 = st.columns(2)

    with rd_col2:
        st.markdown("#### 1️⃣ Feature Engineering (שיפור עתידי)")
        st.markdown(
            """
            - **Log-transform** ל-`canopy_area_m2` (Skewness שזוהה ב-EDA)
            - חילוץ **`bearing`** (כיוון רחוב) מ-OSMnx להצלבה עם `sun_azimuth`,
              לחישוב זווית הטלת צל מדויקת — *לא מומש ב-M3; מתוכנן להמשך*
            """
        )

        st.markdown("#### 2️⃣ חלוקת הדאטה (בפועל)")
        st.markdown(
            """
            - **80% train / 20% test**, `random_state=42` (שחזורי)
            - פיצול **לפי שורה** — מתאים לניווט ברשת ידועה. מגבלה גלויה:
              לא בודק הכללה לרחובות חדשים (שיפור עתידי: Spatial Split)
            """
        )

    with rd_col1:
        st.markdown("#### 3️⃣ Baseline (בפועל)")
        st.markdown(
            """
            - **`DummyRegressor(mean)`** — מנבא ממוצע (RMSE≈1.77). כל מודל חייב לנצח אותו
            - **LinearRegression** הוא מודל **מועמד** (לא baseline) — ראו טבלת התוצאות למעלה
            """
        )

        st.markdown("#### 4️⃣ המודל המומלץ")
        st.markdown(
            """
            - **RandomForestRegressor** — קל למימוש, יציב, אידיאלי לסיבוב הראשון
            - מטרה מתקדמת: **XGBoost / LightGBM** — המלכים של דאטה טבלאי, מטפלים
              אוטומטית באינטראקציות לא-ליניאריות (azimuth × bearing × building height)
            """
        )

    st.markdown(
        """
        <div style="background:#eafaf1;border-right:6px solid #1e8449;padding:16px 20px;border-radius:8px;margin:14px 0 0 0;">
        <div style="font-size:13px;color:#196f3d;font-weight:600;margin-bottom:6px;">⚙️ למה לא מודל ליניארי בלבד?</div>
        <div style="font-size:14px;line-height:1.7;color:#1c1c1c;">
        הנתונים הטבלאיים-גיאוגרפיים שלנו כוללים אינטראקציות לא-ליניאריות קיצוניות —
        למשל: "אם השעה 13:00 <em>וגם</em> הרחוב פונה לצפון <em>וגם</em> יש בניין גבוה — אז יש צל".
        מודלים ליניאריים לא יזהו את הצמדים האלה. מודלים מבוססי עצים (Random Forest / XGBoost) כן.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── מדד הנוחות התרמית ──────────────────────────────────────────────────
    st.subheader("🌡️ מדד הנוחות התרמית (TCI)")

    tci_col1, tci_col2 = st.columns([1, 2])
    with tci_col1:
        st.markdown(
            """
            <div style="background:#fdecea;border-right:6px solid #c0392b;padding:16px 18px;border-radius:8px;">
            <div style="font-size:13px;color:#7b241c;font-weight:600;margin-bottom:8px;">ציון 1–10 לכל קשת ברחוב</div>
            <div style="font-size:13px;color:#444;line-height:1.8;">
            <b style="color:#1e8449;">1</b> = מוצל / נוח מאוד<br>
            <b style="color:#d35400;">5</b> = חשיפה חלקית<br>
            <b style="color:#c0392b;">10</b> = חשיפה מלאה לשמש
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tci_col2:
        st.markdown(
            """
            **7 פיצ'רים במודל:**

            | פיצ'ר | תיאור |
            |-------|-------|
            | `sun_altitude` | גובה השמש בשמיים (PySolar) |
            | `building_height` | ממוצע גבהי המבנים הסמוכים |
            | `canopy_ratio` | יחס חופת עצים לאורך הקשת |
            | `cloud_cover` | כיסוי עננים (%) |
            | `temperature` | טמפרטורה (°C) — פרוקסי ל-cloud_cover |
            | `humidity` | לחות יחסית (%) |
            | `shadow_angle` | זווית שמש-רחוב (0°=מקביל→אין צל, 90°=ניצב→צל מקסימלי) |
            """
        )
