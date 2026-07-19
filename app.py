"""
SHADY – ניווט עירוני מוצל | Streamlit App
"""

import random
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import streamlit as st
import streamlit.components.v1 as components
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
        geocode_address, compute_tci_weights, compute_route,
        compute_edge_tci, weights_from_edge_tci,
        build_route_map, plan_route, load_shadow_coverage,
        compute_length_route, compute_route_insights,
        _DID_YOU_KNOW,
    )
    _ROUTING = True
except ImportError:
    _ROUTING = False
    _DID_YOU_KNOW = ("הידעת? SHADY בונה מסלולים מוצלים לפי גובה מבנים, חופת עצים ומיקום השמש.",)
    @st.cache_data
    def load_shadow_coverage():
        from pathlib import Path as _PF
        p = _PF("data/shadow_coverage.parquet")
        if not p.exists():
            return None
        return pd.read_parquet(p)

# M4 — סוכן LLM לחילוץ פרמטרי ניווט מטקסט חופשי (ייבוא groq עצל בתוך הפונקציה)
try:
    from src.agent import extract_route_params, recommend_route_insight
    _AGENT = True
except ImportError:
    _AGENT = False

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
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
            text-align: right;
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

        /* הסתרת מיתוג Streamlit — Deploy, footer בלבד (קוסמטי).
           לא מסתירים את stMainMenu (כפתור "⋮" הקבוע — פתיחתו מציגה Rerun תמיד,
           לא רק אחרי שינוי קובץ) ולא את stToolbar/stStatusWidget (הבאנר הזמני). */
        footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
        [data-testid="stToolbarActions"] {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── שפת עיצוב: אורגני-משחקי + נגיעות עורכניות, אור/חושך ─────────────────────
# טוקנים (--bg/--ink/--shade/--sun/--accent) עם 3 מנגנוני-על עצמאיים:
# (1) prefers-color-scheme אוטומטי לפי מערכת ההפעלה: (2) [data-theme] מפורש —
# מוזרק ע"י המתג הידני בהמשך הקובץ וגובר על (1); (3) config.toml [theme]/[theme.dark]
# מכסה את חלקי Streamlit הילידיים שלא מעוצבים כאן ישירות (למשל ה-checkbox).
st.markdown(
    """
    <style>
        :root {
            --bg:#FFFBF2; --ink:#1F2A22; --muted:#5B6B60;
            --shade:#0E7C66; --shade-soft:#CFEEE3;
            --sun:#F2A93B; --sun-soft:#FDE7C2; --accent:#FF6B4A;
            /* --blob-* נפרדים מ-shade-soft/sun-soft (שנשארים עדינים לגבולות/צ'יפים):
               shade-soft ב-light קרוב מדי בבהירות ל-bg (מנטה חיוורת על קרם) ונבלע —
               לבלובים צריך ניגודיות גבוהה יותר משני הצדדים כדי שהירוק והכתום ייראו שווים. */
            --blob-shade:#9FE0C6; --blob-sun:#F9C97C;
            --font-display:"Segoe UI Rounded","SF Pro Rounded",ui-rounded,"Varela Round",sans-serif;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg:#14231D; --ink:#F2EFE6; --muted:#9FB0A6;
                --shade:#35C9A0; --shade-soft:rgba(53,201,160,.16);
                --sun:#FFB443; --sun-soft:rgba(255,180,67,.16); --accent:#FF7C5C;
                --blob-shade:rgba(53,201,160,.4); --blob-sun:rgba(255,180,67,.4);
            }
        }
        :root[data-theme="dark"] {
            --bg:#14231D; --ink:#F2EFE6; --muted:#9FB0A6;
            --shade:#35C9A0; --shade-soft:rgba(53,201,160,.16);
            --sun:#FFB443; --sun-soft:rgba(255,180,67,.16); --accent:#FF7C5C;
            --blob-shade:rgba(53,201,160,.4); --blob-sun:rgba(255,180,67,.4);
        }
        :root[data-theme="light"] {
            --bg:#FFFBF2; --ink:#1F2A22; --muted:#5B6B60;
            --shade:#0E7C66; --shade-soft:#CFEEE3;
            --sun:#F2A93B; --sun-soft:#FDE7C2; --accent:#FF6B4A;
            --blob-shade:#9FE0C6; --blob-sun:#F9C97C;
        }

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stHeader"] {
            background: var(--bg) !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: var(--muted) !important; opacity: .8 !important;
        }
        .stApp, .stApp p, .stApp label, .stApp span, [data-testid="stCaptionContainer"] {
            color: var(--ink);
        }
        [data-testid="stHeading"] h1 {
            font-family: var(--font-display); color: var(--shade);
        }
        [data-testid="stCaptionContainer"] p {
            font-variant: small-caps; letter-spacing: .06em; color: var(--muted) !important;
        }

        /* כרטיס "הידעת" — נבחר מחדש בכל לחיצה על "מצא מסלול", מוצג בתוך קופסת הטעינה */
        .shady-fact-card {
            position: relative; z-index: 2; width: 100%; box-sizing: border-box;
            margin: 6px 0 18px; padding: 10px 18px; border-radius: 999px;
            background: var(--shade-soft); color: var(--ink); font-size: 16px; line-height: 1.5;
        }

        /* קו-הפרדה דק בגרדיאנט דו-גוני (נגיעה עורכנית) */
        .shady-rule {
            height: 2px; margin: 10px 0 22px; border: none; border-radius: 2px;
            background: linear-gradient(90deg, transparent, var(--shade), var(--sun), transparent);
        }

        /* הירו הראשי — מיתוג SHADY + סיסמה + כותרת עמוד הניווט.
           רכיבי div/span ולא h1 בכוונה: כך העיצוב לא "מדליף" ל-st.title() האחרים
           בנספח המקופל (אודות/למידת הבעיה/סקירת ספרות וכו') שמשתמשים באותו testid. */
        .shady-hero { position: relative; z-index: 2; margin: 4px 0 2px; }
        .shady-hero-brand {
            display: inline-block; /* מצמצם לרוחב הטקסט עצמו כדי שהגרדיאנט יעבור על האותיות ולא ייבלע בקצה הימני של שורה מלאה */
            font-family: var(--font-display); font-size: 3.4rem; font-weight: 800;
            letter-spacing: .03em; line-height: 1.05; margin: 0;
            background: linear-gradient(90deg, var(--shade), var(--sun));
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .shady-hero-tagline {
            font-family: var(--font-display); font-size: 1.15rem; font-weight: 700;
            letter-spacing: .14em; color: var(--accent); margin: 2px 0 16px;
        }
        /* "COOL" בגוון shade (ירוק/טורקיז) ולא accent (אדום-כתום) — אדום מאותת "חם",
           בעוד המשמעות של המילה (וה-shade branding כולו) היא קרירות/צל.
           !important כי `.stApp span { color: var(--ink) }` הגלובלי גובר בספציפיות. */
        .shady-hero-tagline .shady-hero-tagline-cool { color: var(--shade) !important; }
        .shady-hero-subtitle {
            font-family: var(--font-display); font-size: 1.7rem; font-weight: 700;
            color: var(--ink); margin: 0 0 4px;
        }

        /* רקע חי: בלובים צפים + שכבת גרדיאנט "נושמת" על כל העמוד.
           רק background-position/opacity/transform (composited) — עלות ביצועים
           זניחה, לא קשור למנגנוני הטעינה (geocoding/TCI/A*). תנועה מהירה/גדולה
           בכוונה (amplitude ±22px, משכים 4-7s) כדי שתורגש בבירור, לא רק "בעיקרון". */
        @keyframes shady-float {
            0%, 100%  { transform: translate(0, 0) rotate(0deg) scale(1); }
            33%       { transform: translate(-22px, 16px) rotate(-6deg) scale(1.08); }
            66%       { transform: translate(18px, -14px) rotate(5deg) scale(.95); }
        }
        @keyframes shady-breathe {
            0%   { background-position: 0% 0%; }
            100% { background-position: 30% 20%; }
        }
        .stApp::before {
            content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background:
                radial-gradient(55% 45% at 15% 15%, var(--shade-soft), transparent 60%),
                radial-gradient(50% 40% at 85% 10%, var(--sun-soft), transparent 60%);
            background-size: 140% 140%; opacity: .5;
            animation: shady-breathe 28s ease-in-out infinite alternate;
        }

        .shady-blob {
            position: fixed; z-index: 0; pointer-events: none; filter: blur(2px); opacity: .6;
            animation: shady-float 6s ease-in-out infinite;
        }
        .shady-blob.b1 {
            top: 40px; right: 4%; width: 220px; height: 220px; background: var(--blob-shade);
            border-radius: 42% 58% 63% 37% / 41% 44% 56% 59%;
            animation-duration: 6s;
        }
        .shady-blob.b2 {
            top: 110px; left: 6%; width: 160px; height: 160px; background: var(--blob-sun);
            border-radius: 63% 37% 39% 61% / 53% 40% 60% 47%;
            animation-duration: 5s; animation-delay: -2s;
        }
        .shady-blob.b3 {
            top: 240px; right: 14%; width: 120px; height: 120px; background: var(--blob-sun);
            border-radius: 55% 45% 60% 40% / 45% 55% 45% 55%;
            animation-duration: 7s; animation-delay: -4s;
        }
        .shady-blob.b4 {
            top: 0; left: 22%; width: 90px; height: 90px; background: var(--blob-shade);
            border-radius: 60% 40% 45% 55% / 55% 45% 60% 40%;
            animation-duration: 5.5s; animation-delay: -1s;
        }
        .shady-blob.b5 {
            top: 480px; left: 2%; width: 150px; height: 150px; background: var(--blob-shade);
            border-radius: 48% 52% 41% 59% / 58% 42% 55% 45%;
            animation-duration: 6.5s; animation-delay: -3s;
        }
        .shady-blob.b6 {
            top: 560px; right: 8%; width: 110px; height: 110px; background: var(--blob-sun);
            border-radius: 57% 43% 50% 50% / 44% 56% 40% 60%;
            animation-duration: 5s; animation-delay: -2.5s;
        }
        .shady-blob.b7 {
            top: 700px; left: 40%; width: 80px; height: 80px; background: var(--blob-shade);
            border-radius: 50% 50% 45% 55% / 55% 45% 50% 50%;
            animation-duration: 4.5s; animation-delay: -.5s;
        }
        @media (prefers-reduced-motion: reduce) {
            .stApp::before, .shady-blob { animation: none; }
        }

        /* שדות טקסט/תאריך/שעה — pill מעוגל.
           data-baseweb="base-input" צובע את עצמו ב-secondaryBackgroundColor הקבוע
           מ-config.toml (לא עוקב אחרי --data-theme שלנו) — שקוף כדי לחשוף את var(--bg). */
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stDateInput"] [data-baseweb="input"],
        [data-testid="stTimeInput"] [data-baseweb="select"] > div:first-child {
            border-radius: 999px !important; overflow: hidden;
            border: 2px solid var(--shade-soft) !important; background: var(--bg) !important;
        }
        [data-testid="stTextInput"] [data-baseweb="base-input"],
        [data-testid="stDateInput"] [data-baseweb="base-input"] {
            background: transparent !important;
        }
        [data-testid="stTextInput"] input, [data-testid="stDateInput"] input {
            background: transparent !important; color: var(--ink) !important;
        }
        [data-testid="stTimeInputTimeDisplay"] {
            color: var(--ink) !important;
        }

        /* צביעת אינדיקטור ה-"Running..." הגלובלי בפינה לפי הפלטה (במקום אפור ברירת מחדל) */
        [data-testid="stStatusWidgetRunningManIcon"] {
            color: var(--shade) !important;
        }

        /* אייקון-העזרה ("?" ליד תוויות עם help=, כמו בורר העדפת-מסלול ומדדי TCI) —
           לא עוקב אחרי --ink/--data-theme שלנו כברירת מחדל, ובמצב הכהה המותאם
           של האפליקציה כמעט לא נראה על הרקע הכהה. צבוע ב-var(--shade) (ירוק המותג —
           ניגודיות טובה גם בהיר #0E7C66 וגם כהה #35C9A0) לבהירות בשני המצבים.
           הערה (2026-07-18): רק color/fill — בלי width/height! st.button(help=) עוטף
           את הכפתור באותו stTooltipIcon, וכפיית width שם כיווצה את הכפתור "הרץ מיד"
           לרוחב תו-בודד (הטקסט נמרח אנכית). צביעה בלבד לא משפיעה על הפריסה. */
        [data-testid="stTooltipIcon"] {
            color: var(--shade) !important;
            opacity: 1 !important;
        }
        [data-testid="stTooltipIcon"] svg path {
            fill: var(--shade) !important;
        }

        /* בורר העדפת מסלול — pill group קופצני */
        [data-testid="stRadio"] > div[role="radiogroup"] { gap: 8px; }
        [data-testid="stRadio"] label[data-baseweb="radio"] {
            border-radius: 999px !important; padding: 8px 18px !important; margin: 0 !important;
            background: var(--shade-soft); transition: transform .2s cubic-bezier(.34,1.56,.64,1), background .2s ease;
        }
        [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
        [data-testid="stRadio"] label[data-baseweb="radio"] p { color: var(--ink); font-weight: 600; }
        [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background: var(--accent); transform: scale(1.05);
            box-shadow: 0 6px 14px -4px rgba(255,107,74,.5);
        }
        [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color: #fff; }

        /* כפתורים */
        [data-testid="stBaseButton-primary"] {
            border-radius: 999px !important; background: var(--accent) !important; border: none !important;
            box-shadow: 0 10px 22px -8px rgba(255,107,74,.55);
            transition: transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s ease;
        }
        [data-testid="stBaseButton-primary"]:hover { transform: translateY(-2px); }
        [data-testid="stBaseButton-primary"]:active { transform: scale(.95); }
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-secondaryFormSubmit"] {
            border-radius: 999px !important; background: var(--shade-soft) !important;
            border: none !important; color: var(--ink) !important;
        }

        /* כרטיס תוצאות + מדדים */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 22px !important; border-color: var(--shade-soft) !important;
            box-shadow: 0 18px 40px -18px rgba(31,42,34,.18);
        }
        [data-testid="stMetricValue"] {
            font-family: var(--font-display); font-variant-numeric: tabular-nums; color: var(--ink) !important;
        }
        [data-testid="stMetricLabel"] {
            font-variant: small-caps; letter-spacing: .05em; color: var(--muted) !important;
        }
    </style>
    <div class="shady-blob b1"></div>
    <div class="shady-blob b2"></div>
    <div class="shady-blob b3"></div>
    <div class="shady-blob b4"></div>
    <div class="shady-blob b5"></div>
    <div class="shady-blob b6"></div>
    <div class="shady-blob b7"></div>
    """,
    unsafe_allow_html=True,
)

# ── טעינת נתונים ──────────────────────────────────────────────────────────────
@st.cache_data
def load_buildings() -> pd.DataFrame:
    return pd.read_csv("data/buildings_clean.csv")

@st.cache_data
def load_trees_full() -> pd.DataFrame:
    # טוען את כל העמודות המספריות + הקטגוריות לטאב ה-EDA
    df = pd.read_parquet("data/national_canopy_clean.parquet")
    return df[["OBJECTID", "canopy_perimeter_m", "canopy_area_m2",
               "lon", "lat", "Geometry_Type", "area_class"]].copy()

# מזג אוויר — נשמר ב-cache ל-5 דקות כדי שלא ייקרא HTTP request בכל rerun
@st.cache_data(ttl=300, show_spinner=False)
def _get_weather_cached() -> dict:
    if _WEATHER:
        return _get_weather()
    return {"temperature": 27.0, "humidity": 65.0, "cloud_cover": 30.0, "source": "default"}

if _ROUTING:
    from src.agent import normalize_place_name as _normalize_place

    @st.cache_data(show_spinner=False)
    def _geocode_cached(address: str, _on_progress=None) -> tuple:
        # קידומת "_" מוציאה את הפרמטר ממפתח ה-cache (callback לא ניתן ל-hash) —
        # מפתח הקאש נשאר address בלבד, זהה להתנהגות הקודמת.
        return geocode_address(address, on_progress=_on_progress)

    def _geocode_llm(address: str, _on_progress=None) -> tuple:
        """_geocode_cached עם LLM fallback — אם כל שלושת השלבים נכשלים, מנסה לנרמל דרך Groq.

        מקבל _on_progress (positional) כדי להיות תואם-חתימה ל-geocode_fn ש-plan_route
        קורא כ-geocode_fn(addr, on_progress) — כך אותו נתיב-גיאוקוד (כולל ה-LLM fallback)
        משמש גם את התצוגה-המקדימה וגם את חישוב-המסלול, בלי הבדל שגורם ל'זוהה אבל לא נמצא'."""
        try:
            return _geocode_cached(address, _on_progress)
        except ValueError:
            try:
                groq_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                groq_key = ""
            if groq_key:
                normalized = _normalize_place(address, groq_key)
                if normalized:
                    try:
                        return _geocode_cached(normalized, _on_progress)
                    except ValueError:
                        pass
            raise

    def _geocode_and_store(addr_key: str, ok_key: str, err_key: str) -> None:
        """מגאוקד כתובת מ-session_state (עם LLM fallback) ושומר תוצאה/שגיאה שם.

        משותף לשני נתיבי-קריאה: on_change ידני (הקלדה בשדה) ומילוי-הסוכן (תכנותי,
        לא מפעיל on_change) — כדי ששניהם ייהנו מאותו LLM fallback (_geocode_llm),
        לא רק הנתיב הידני.
        """
        addr = st.session_state.get(addr_key, "").strip()
        st.session_state.pop(ok_key, None)
        st.session_state.pop(err_key, None)
        if not addr:
            return
        try:
            st.session_state[ok_key] = _geocode_llm(addr)
        except ValueError as e:
            st.session_state[err_key] = str(e)

    def _validate_address_field(addr_key: str, ok_key: str, err_key: str) -> None:
        """on_change callback — מגאוקד את הכתובת ושומר תוצאה ב-session_state."""
        _geocode_and_store(addr_key, ok_key, err_key)


@st.cache_data
def build_tci_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """בונה N דוגמאות TCI — מאציל ל-src.data (מקור אמת יחיד, בלי כפילות).

    בעבר הייתה כאן לוגיקה משוכפלת מ-data.py; אוחד כדי שהאימון (model.py)
    והתצוגה (app.py) ישתמשו באותו חישוב TCI בדיוק — כולל הצללת shadow_cov.
    הכיוון הזה מותר: ל-src.data אין תלות ב-Streamlit.
    """
    from src.data import build_tci_df as _build_tci_df_clean
    return _build_tci_df_clean(n=n, seed=seed)


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
def _precompute_edge_tci(
    sun_alt_r: float, sun_az_r: float,
    cloud_r: float, temp_r: float, hum_r: float,
):
    """
    מחשב TCI פר-קשת (58k) — **החלק היקר** (model.predict), אינו תלוי ב-shade_factor.
    ממוטמן לפי מזג-אוויר בלבד, כך שבמהלך לולאת ה-backoff (הרבה shade_factor, אותו
    מזג-אוויר) ה-predict רץ פעם אחת ולא עשרות פעמים.
    """
    _bundle   = load_tci_model()
    _edges_df = load_edges_full()
    if _bundle is None or _edges_df is None:
        return None
    return compute_edge_tci(
        _edges_df, _bundle,
        float(sun_alt_r), float(sun_az_r),
        float(cloud_r), float(temp_r), float(hum_r),
    )


@st.cache_data(show_spinner=False)
def _precompute_nav_weights(
    sun_alt_r: float, sun_az_r: float,
    cloud_r: float, temp_r: float, hum_r: float,
    lam: float = 1.0,
):
    """
    מחזיר (weight_dict, tci_by_uv) לניווט. שכבת החֶזְקָה הזולה מעל edge_tci הממוטמן —
    ה-predict היקר כבר בוצע ב-_precompute_edge_tci (cache hit אחרי הקריאה הראשונה).
    ממוטמן לפי מזג אוויר + λ (מחיר-האורך במודל האדיטיבי). plan_route קורא לכאן בערכי-λ
    מגוונים בחיפוש-התקציב (לא ב-shade_factor) — כל λ הוא cache-key נפרד, אך ה-predict
    היקר ב-_precompute_edge_tci ממוטמן במזג-אוויר בלבד ולכן רץ פעם אחת.
    """
    edge_tci = _precompute_edge_tci(sun_alt_r, sun_az_r, cloud_r, temp_r, hum_r)
    if edge_tci is None:
        return None, None
    return weights_from_edge_tci(edge_tci, float(lam))


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



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="shady-hero">
        <div class="shady-hero-brand">SHADY</div>
        <div class="shady-hero-tagline">STAY <span class="shady-hero-tagline-cool">COOL</span> :)</div>
        <div class="shady-hero-subtitle">🚶‍♀️ ניווט חכם בתל אביב</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# מתג אור/חושך — צד-לקוח בלבד (localStorage), בלי st.rerun():
# טוגל Python/session_state היה גורם ל-rerun מלא בכל לחיצה, ומוחק תוצאת מסלול
# שכבר חושבה (find_btn/route_result אינם sticky בין ריצות סקריפט). components.html
# רץ ב-iframe אמיתי (script בו כן מבוצע, בניגוד ל-st.markdown, שמסנן גם <script>
# וגם attributes מסוג onclick=... — DOMPurify חוסם שניהם). לכן מחברים event
# listener מתוך ה-iframe עצמו ל-button שנוצר ע"י st.markdown, עם retry קצר כי
# שני הרכיבים נטענים באופן א-סינכרוני זה מזה וסדר ההופעה ב-DOM לא מובטח.
st.markdown(
    """
    <div style="position:fixed; top:14px; left:20px; z-index:999999;">
        <button id="shady-theme-btn" title="מצב בהיר / כהה"
            style="border:none; background:var(--shade-soft); color:var(--ink); border-radius:999px;
                   width:38px; height:38px; font-size:18px; cursor:pointer;">🌙</button>
    </div>
    <hr class='shady-rule' />
    """,
    unsafe_allow_html=True,
)
components.html(
    """
    <script>
    (function () {
        var d = window.parent.document;
        var stored = null;
        try { stored = window.parent.localStorage.getItem('shady-theme'); } catch (e) {}
        var theme = stored || (window.parent.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        d.documentElement.setAttribute('data-theme', theme);

        function applyIcon(btn, t) { btn.textContent = t === 'dark' ? '☀️' : '🌙'; }

        function wire(tries) {
            var btn = d.getElementById('shady-theme-btn');
            if (!btn) {
                if (tries > 0) { setTimeout(function () { wire(tries - 1); }, 100); }
                return;
            }
            applyIcon(btn, theme);
            if (btn.dataset.shadyWired) { return; }
            btn.dataset.shadyWired = '1';
            btn.addEventListener('click', function () {
                var cur = d.documentElement.getAttribute('data-theme');
                var next = cur === 'dark' ? 'light' : 'dark';
                d.documentElement.setAttribute('data-theme', next);
                try { window.parent.localStorage.setItem('shady-theme', next); } catch (e) {}
                applyIcon(btn, next);
            });
        }
        wire(20);
    })();
    </script>
    """,
    height=0, width=0,
)

if not _ROUTING:
    st.error("⚠️ חבילת `osmnx` לא מותקנת — התקן עם `pip install osmnx networkx` והפעל מחדש.")
    st.stop()

# הגרף כבר נטען לפני הטאבים — הקריאה הזו מחזירה אותו מה-cache (0ms).
_nav_G = _load_nav_graph()


def _clamped_now() -> tuple:
    """שעה נוכחית בשעון ישראל, מעוגלת לחצי-השעה הקרובה ביותר (round-half-up: 15 דק'
    בדיוק מעוגלות למעלה). ללא הצמדה לטווח — מחזירה את השעה האמיתית בכל שעות היממה
    (בערב/לילה השמש שלילית → plan_route נופל אוטומטית למסלול מהיר; ר' טיפול-הלילה למטה).
    השם ההיסטורי "clamped" נשמר לתאימות הקוראים, גם אם ההצמדה עצמה הוסרה.

    שעון ישראל במפורש — `datetime.now()` נאיבי מחזיר UTC על Streamlit Cloud (Linux),
    מה שהזיז את ברירת-המחדל 3 שעות אחורה (19:20 בישראל → 16:20 UTC → 16:00)."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    total = now.hour * 60 + now.minute
    rounded = (((total + 15) // 30) * 30) % 1440   # עיגול לחצי-שעה + גלישת 24:00→00:00
    return rounded // 60, rounded % 60


def _today_il():
    """התאריך של היום בשעון ישראל — `date.today()` נאיבי נותן תאריך UTC על Streamlit
    Cloud, שגוי בחלון חצות–03:00 שעון ישראל (ושובר גם את טבלת התאריכים של הסוכן)."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Jerusalem")).date()

# ── סוכן LLM (M4) ──────────────────────────────────────────────────────────
# מחלץ פרמטרים מטקסט חופשי וממלא את טופס הניווט שלמטה. חייב לרוץ *לפני* יצירת
# ה-widgets, אחרת Streamlit אוסר לכתוב ל-session_state של widget קיים.
st.markdown("#### 💬 דברו עם הסוכן")
with st.form("agent_form", clear_on_submit=False):
    _agent_text = st.text_input(
        "תארו את הבקשה במשפט אחד",
        placeholder='לדוגמה: "מכיכר רבין לשוק הכרמל ב-3 אחה\"צ בדרך הכי מוצלת"',
        key="agent_text",
    )
    _agent_submit = st.form_submit_button("שלח לסוכן 🤖", use_container_width=True)
if _agent_submit:
    st.session_state.pop("_agent_banner", None)           # נקה כרזות ישנות לפני בקשה חדשה
    st.session_state.pop("_agent_recommendation", None)
    st.session_state.pop("_agent_nav_mode", None)
    try:
        _groq_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        _groq_key = None
    if not _AGENT:
        st.warning("⚠️ מודול הסוכן לא זמין (בדקו שהותקן `groq`).")
    elif not _groq_key:
        st.warning("⚠️ מפתח `GROQ_API_KEY` חסר — הוסיפו אותו ל-`.streamlit/secrets.toml`.")
    elif not _agent_text.strip():
        st.warning("⚠️ כתבו בקשה לסוכן.")
    else:
        with st.spinner("הסוכן מנתח את הבקשה..."):
            from datetime import date as _date_cls
            # זווית שמש נוכחית (לפי שעה אמיתית עכשיו, לא nav_time) להמלצת הסוכן
            _agent_sun_alt = None
            if _PYSOLAR:
                try:
                    from zoneinfo import ZoneInfo as _ZI_a
                    import datetime as _dt_a
                    _dtc_a = _dt_a.datetime.now(tz=_ZI_a("Asia/Jerusalem"))
                    _agent_sun_alt = float(_solar.get_altitude(32.08, 34.77, _dtc_a))
                except Exception:
                    pass
            _weather_a = _get_weather_cached()
            _params = extract_route_params(
                _agent_text, _groq_key, today=_today_il(),
                sun_altitude=_agent_sun_alt,
                temperature=_weather_a.get("temperature"),
                cloud_cover=_weather_a.get("cloud_cover"),
            )
        if _params.get("error"):
            st.warning(_params["error"])
        else:
            # ממלאים את טופס הניווט הקיים — ה-widgets נוצרים מיד אחר כך וקוראים את הערכים.
            st.session_state["nav_origin"] = _params["origin"]
            st.session_state["nav_dest"]   = _params["destination"]
            from datetime import time as _time_cls, date as _da, timedelta as _tdd
            if _params["hour"] is not None:
                _ah = int(_params["hour"])
                _am = int(round((_params["hour"] - _ah) * 60))
                st.session_state["nav_time"] = _time_cls(min(_ah, 23), min(_am, 59))
            else:
                # שעה לא צוינה (עכשיו / לא הוזכרה) → איפוס לשעה נוכחית
                _h_r, _m_r = _clamped_now()
                st.session_state["nav_time"] = _time_cls(_h_r, _m_r)
            if _params.get("date"):
                _parsed_d = _da.fromisoformat(_params["date"])
                _today_d  = _da.today()
                if _today_d <= _parsed_d <= _today_d + _tdd(days=7):
                    st.session_state["nav_date"] = _parsed_d
            else:
                # תאריך לא צוין → איפוס להיום
                st.session_state["nav_date"] = _da.today()
            # העדפת צל — עדכן רק אם הסוכן זיהה במפורש
            _sl = _params.get("shade_level")
            _shade_map_agent = {
                "short": "🚶 מעט צל",
                "max":   "🌳 הרבה צל",
            }
            if _sl in _shade_map_agent:
                st.session_state["nav_shade_pref_label"] = _shade_map_agent[_sl]
            st.session_state["_agent_nav_mode"] = _params.get("mode", "shaded")
            # on_change callback לא מופעל על שינוי תכנותי — גיאוקוד ידני, דרך אותה
            # פונקציה משותפת שה-on_change הרגיל משתמש בה (כולל LLM fallback).
            if _ROUTING:
                for _addr_key, _ok_key, _err_key in [
                    ("nav_origin", "_origin_ok", "_origin_err"),
                    ("nav_dest",   "_dest_ok",   "_dest_err"),
                ]:
                    _geocode_and_store(_addr_key, _ok_key, _err_key)
            _h = _params["hour"]
            _ht = (f"{int(_h):02d}:{int(round((_h - int(_h)) * 60)):02d}"
                   if _h is not None else "שעה נוכחית")
            _date_val = _params.get("date")
            if _date_val is None:
                _date_str = "היום"
            else:
                _pd = _da.fromisoformat(_date_val)
                _diff = (_pd - _da.today()).days
                if _diff == 0:
                    _date_str = "היום"
                elif _diff == 1:
                    _date_str = "מחר"
                else:
                    _date_str = f"{_pd.day}/{_pd.month}"
            _shade_label_display = {
                "short": "מעט צל", "max": "הרבה צל",
            }.get(_sl or "", "")
            _shade_str = f" · {_shade_label_display}" if _shade_label_display else ""
            st.session_state["_agent_banner"] = (
                f"✅ הסוכן מילא — מ**{_params['origin']}** "
                f"ל**{_params['destination']}**, {_date_str}, שעה {_ht}{_shade_str}"
            )
            _rec = _params.get("recommendation")
            if _rec:
                st.session_state["_agent_recommendation"] = _rec
            else:
                st.session_state.pop("_agent_recommendation", None)

# כרזה מתמשכת + כפתור הרצה מיידית (נשמרת עד לבקשה הבאה)
_quick_run = False
if "_agent_banner" in st.session_state:
    st.success(st.session_state["_agent_banner"])
    if "_agent_recommendation" in st.session_state:
        st.info(st.session_state["_agent_recommendation"])
    _quick_run = st.button("🗺️ הרץ מיד", key="quick_run_btn",
                           help="הרץ את המסלול עם הפרמטרים שהסוכן מילא")
st.divider()

c1, c2 = st.columns(2)
with c1:
    origin_input = st.text_input(
        "📍 נקודת מוצא",
        placeholder="לדוגמה: כיכר רבין, תל אביב",
        key="nav_origin",
        on_change=_validate_address_field,
        args=("nav_origin", "_origin_ok", "_origin_err"),
    )
    if "_origin_ok" in st.session_state:
        st.success("✅ מיקום זוהה")
    elif "_origin_err" in st.session_state:
        st.error(st.session_state["_origin_err"])
with c2:
    dest_input = st.text_input(
        "🏁 יעד",
        placeholder="לדוגמה: שוק הכרמל, תל אביב",
        key="nav_dest",
        on_change=_validate_address_field,
        args=("nav_dest", "_dest_ok", "_dest_err"),
    )
    if "_dest_ok" in st.session_state:
        st.success("✅ מיקום זוהה")
    elif "_dest_err" in st.session_state:
        st.error(st.session_state["_dest_err"])

# תיקון RTL בלוח שנה — wildcard על כל הצאצאים מבטיח שאף קונטיינר פנימי לא ישאר RTL
st.markdown("""
<style>
div[data-baseweb="calendar"],
div[data-baseweb="calendar"] * {
    direction: ltr !important;
    unicode-bidi: isolate !important;
}
</style>
""", unsafe_allow_html=True)

# בחירת תאריך ושעת יציאה — ברירת מחדל = עכשיו, טווח = 7 ימים קדימה.
from datetime import date as _date, timedelta as _td, time as _time

# טיפול בבקשת איפוס (מהכפתור 🔄) — חייב לקרות *לפני* יצירת הווידג'טים nav_date/nav_time,
# כי Streamlit אוסר לכתוב ל-session_state של ווידג'ט באותה הרצה שבה הוא כבר נוצר.
if st.session_state.pop("_reset_datetime", False):
    _h_r, _m_r = _clamped_now()
    st.session_state["nav_date"] = _today_il()
    st.session_state["nav_time"] = _time(_h_r, _m_r)

_today     = _today_il()
_clamp_h, _clamp_m = _clamped_now()
_default_time = _time(_clamp_h, _clamp_m)

_dcol, _tcol, _rcol = st.columns([3, 3, 1])
with _dcol:
    nav_date = st.date_input(
        "📅 תאריך יציאה",
        value=_today,
        min_value=_today,
        max_value=_today + _td(days=7),
        key="nav_date",
    )
with _tcol:
    nav_time = st.time_input(
        "⏰ שעת יציאה",
        value=_default_time,
        step=1800,
        key="nav_time",
    )
with _rcol:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄", key="nav_reset", help="איפוס לשעה ותאריך נוכחיים"):
        st.session_state["_reset_datetime"] = True
        st.rerun()

_nav_hour = nav_time.hour + nav_time.minute / 60.0
_nav_now  = datetime(nav_date.year, nav_date.month, nav_date.day)
_nh = nav_time.hour; _nm = nav_time.minute
if _PYSOLAR:
    from zoneinfo import ZoneInfo as _ZI
    _dtc = datetime(nav_date.year, nav_date.month, nav_date.day,
                    _nh, _nm, tzinfo=_ZI("Asia/Jerusalem"))
    _ca = float(_solar.get_altitude(32.08, 34.77, _dtc))
    _cz = float(_solar.get_azimuth(32.08, 34.77, _dtc))
    st.caption(f"🕐 {_nh:02d}:{_nm:02d} · שמש: גובה {_ca:.0f}° · אזימוט {_cz:.0f}°")
else:
    _ca = 45.0
    st.caption(f"🕐 {_nh:02d}:{_nm:02d}")

_SHADE_KEY = "nav_shade_pref_label"
_SHADE_LAST_KEY = "_nav_shade_pref_last_choice"

if _ca <= 0:
    # לילה — plan_route מתעלם לגמרי מהעדפת-הצל (routing.py: sun_alt<=0 נופל
    # ל-OSRM לפני שהקוד מגיע ל-shade_factor בכלל), אז לא הגיוני להראות כפתור
    # מסומן. שומרים את הבחירה הקודמת (אם הייתה אמיתית) כדי לשחזר כשחוזרים ליום.
    if st.session_state.get(_SHADE_KEY) is not None:
        st.session_state[_SHADE_LAST_KEY] = st.session_state[_SHADE_KEY]
    st.session_state[_SHADE_KEY] = None
elif st.session_state.get(_SHADE_KEY) is None:
    # חזרנו ליום (או הרצה ראשונה) — משחזרים את הבחירה הקודמת, או "מעט צל" כברירת מחדל
    st.session_state[_SHADE_KEY] = st.session_state.get(_SHADE_LAST_KEY, "🚶 מעט צל")

_shade_opts = {
    "🚶 מעט צל": 1.0,
    "🌳 הרבה צל": 2.0,
}
_shade_label = st.radio(
    "🌿 העדפת מסלול",
    options=list(_shade_opts.keys()),
    index=None,                      # מאפשר "בלי בחירה" (לילה)
    horizontal=True,
    key="nav_shade_pref_label",
    help="מעט צל = ברירת מחדל, מוטה לצל הכי פחות; הרבה צל = צל מקסימלי גם במחיר עיקוף ניכר",
)
shade_pref = _shade_opts.get(_shade_label, 1.0)   # None בלילה — לא משנה, plan_route מתעלם מזה

if "nav_compare" not in st.session_state:
    st.session_state["nav_compare"] = True
compare_fast = st.checkbox("🔀 הצג גם את המסלול המהיר ביותר להשוואה", key="nav_compare")
if _ca <= 0:
    st.info("🌙 חשוך בחוץ — יוצג מסלול מהיר (אין ניגודיות צל בלילה).")

find_btn = st.button("מצא מסלול 🗺️", type="primary")
st.divider()

if find_btn or _quick_run:
    if not origin_input.strip() or not dest_input.strip():
        st.warning("⚠️ יש להזין גם נקודת מוצא וגם יעד.")
    elif "_origin_ok" not in st.session_state or "_dest_ok" not in st.session_state:
        st.warning("⚠️ יש לוודא שהכתובות זוהו (✅) לפני חישוב המסלול.")
    else:
        # "הידעת" נבחר מחדש בכל לחיצה (לא session-level), מוצג כרכיב עצמאי
        # מתחת לכפתור — לא בתוך st.status (שם הוא היה "נבלע" בקופסה המתקפלת
        # עם חץ, ואנשים מפספסים אותו כשהיא נסגרת אוטומטית בסיום).
        st.markdown(
            f'<div class="shady-fact-card">💡 {random.choice(_DID_YOU_KNOW)}</div>',
            unsafe_allow_html=True,
        )
        with st.status("מחשב מסלול מוצל...", expanded=True) as _status:
            def _on_progress(msg: str) -> None:
                _status.update(label=msg)

            error_msg = None
            plan      = None
            _fast_result = None
            _fast_compare_error = False
            try:
                _bundle   = load_tci_model()
                _edges_df = load_edges_full()
                _use_shaded = not (
                    _quick_run and
                    st.session_state.get("_agent_nav_mode") == "fast"
                )
                plan = plan_route(
                    origin_input, dest_input,
                    use_shaded=_use_shaded,
                    nav_hour=_nav_hour,
                    geocode_fn=_geocode_llm,   # אותו נתיב (כולל LLM fallback) כמו התצוגה-המקדימה
                    weather_fn=_get_weather_cached,
                    weights_fn=_precompute_nav_weights,
                    has_model=(_bundle is not None and _edges_df is not None),
                    graph=_nav_G,
                    now=_nav_now,
                    shade_factor=float(shade_pref),
                    on_progress=_on_progress,
                )

                # מסלול מהיר להשוואה — בתוך אותו st.status, לפני "המסלול מוכן!", כדי
                # שזה יראה כרצף טעינה אחד ולא כשתי טעינות נפרדות (הראשונה "מסתיימת"
                # ומיד אחריה ספינר נוסף).
                # משתמשים ב-compute_length_route (הקצר-ביותר *על הגרף*, TCI מדויק) ולא ב-OSRM:
                # זה אותו בסיס שהשער ב-plan_route משווה אליו, כך שני המסלולים בני-השוואה
                # ומובטח שהמוצל לעולם לא חם/ארוך יותר מהמהיר (אחרת ערבוב גרף↔OSRM ומדידה
                # מדויקת↔מוצמדת יצר "מוצל" חם מ"מהיר"). OSRM נשאר למצבי-מהיר האמיתיים.
                if compare_fast and plan["mode"] == "shaded":
                    _status.update(label="מחשב מסלול מהיר להשוואה...")
                    try:
                        _fast_result = compute_length_route(
                            plan["origin_latlon"], plan["dest_latlon"],
                            plan.get("tci_uv"), G=_nav_G,
                        )
                    except Exception:
                        _fast_compare_error = True
            except ValueError as e:
                error_msg = f"❌ {e}"
            except Exception as e:
                error_msg = f"❌ שגיאה: {e}"

            # st.status מסמן "complete" (ירוק) אוטומטית רק אם exception בורח מה-with —
            # מכיוון שהיא נתפסת למעלה, צריך לעדכן ידנית גם למקרה שגיאה וגם להצלחה.
            if error_msg:
                _status.update(label="שגיאה בחישוב המסלול", state="error", expanded=True)
            else:
                _status.update(label="המסלול מוכן!", state="complete", expanded=False)

        _FALLBACK_MSGS = {
            "model_missing":   "⚠️ מודל ML לא זמין — מציג מסלול מהיר.",
            "weights_missing": "⚠️ משקלי TCI לא זמינים — מציג מסלול מהיר.",
            "night":           "🌙 חשוך בחוץ — מציג מסלול מהיר (אין צל בלילה).",
            "overcast":        "☁️ מעונן מאוד (>80%) — מציג מסלול מהיר (הצל פחות רלוונטי).",
            "little_shade_below_band": "העיקוף המוצל קטן מ־10% — מציג את המסלול המהיר.",
        }
        if plan is not None and plan["fallback"] in _FALLBACK_MSGS:
            st.info(_FALLBACK_MSGS[plan["fallback"]])
        if _fast_compare_error:
            st.warning("⚠️ לא הצלחתי לחשב מסלול מהיר להשוואה.")

        if error_msg:
            st.error(error_msg)
        elif plan is not None:
            route_result  = plan["route_result"]
            _route_color  = plan["color"]
            origin_latlon = plan["origin_latlon"]
            dest_latlon   = plan["dest_latlon"]
            _avg_tci      = route_result.get("avg_tci")

            # תצוגת מדדים: עמודות כפולות כשיש השוואה, רגיל אחרת
            with st.container(border=True):
                if _fast_result:
                    _mc_s, _mc_f = st.columns(2)
                    with _mc_s:
                        st.markdown("**🌿 מסלול מוצל**")
                        st.metric("📏 מרחק", f"{route_result['distance_m']:.0f} מ'")
                        st.metric("⏱️ זמן הליכה", f"{route_result['duration_min']:.0f} דקות")
                        if _avg_tci is not None:
                            st.metric("☀️ TCI ממוצע", f"{_avg_tci:.1f} / 10",
                                      help="🟢 TCI 1 = מוצל לחלוטין  \n🔴 TCI 10 = חשיפה מלאה")
                    with _mc_f:
                        st.markdown("**⚡ מסלול מהיר**")
                        st.metric("📏 מרחק", f"{_fast_result['distance_m']:.0f} מ'")
                        st.metric("⏱️ זמן הליכה", f"{_fast_result['duration_min']:.0f} דקות")
                        if _fast_result.get("avg_tci") is not None:
                            st.metric("☀️ TCI ממוצע", f"{_fast_result['avg_tci']:.1f} / 10",
                                      help="🟢 TCI 1 = מוצל לחלוטין  \n🔴 TCI 10 = חשיפה מלאה")
                else:
                    _n_cols = 3 if _avg_tci is not None else 2
                    _mc = st.columns(_n_cols)
                    _mc[0].metric("📏 מרחק", f"{route_result['distance_m']:.0f} מ'")
                    _mc[1].metric("⏱️ זמן הליכה משוער", f"{route_result['duration_min']:.0f} דקות")
                    if _avg_tci is not None:
                        _mc[2].metric("☀️ חשיפה ממוצעת לשמש", f"{_avg_tci:.1f} / 10",
                                      help="🟢 TCI 1 = מוצל לחלוטין  \n🔴 TCI 10 = חשיפה מלאה לשמש")

            route_map = build_route_map(
                origin_latlon, dest_latlon, route_result,
                color=_route_color,
                tci_list=route_result.get("tci_list"),
                fast_result=_fast_result,
            )
            st_folium(route_map, height=520, use_container_width=True, returned_objects=[])

            # תובנת מסלול (M4 tool use): הסוכן קורא לכלי evaluate_route → הקוד מריץ
            # מסלול-בסיס + השוואה (metrics_fn) → הסוכן מנסח משפט מעוגן במספרים.
            if _AGENT and plan["mode"] == "shaded" and plan.get("tci_uv"):
                _tci_uv = plan["tci_uv"]

                def _metrics_fn():
                    if _fast_result and _fast_result.get("avg_tci") is not None:
                        # אותו מסלול-מהיר בדיוק שמוצג בכרטיס ובמפה — לא מסלול-בסיס נסתר.
                        _baseline = _fast_result
                    else:
                        _baseline = compute_length_route(
                            origin_latlon, dest_latlon, _tci_uv, G=_nav_G)
                    return compute_route_insights(route_result, _baseline)

                try:
                    _ins_key = st.secrets.get("GROQ_API_KEY", "")
                except Exception:
                    _ins_key = ""
                with st.spinner("הסוכן מנתח את תועלת המסלול..."):
                    try:
                        _insight = recommend_route_insight(
                            f"הסבר את התועלת של המסלול המוצל מ{origin_input} ל{dest_input}.",
                            _ins_key, _metrics_fn,
                        )
                    except Exception:
                        _insight = ""
                if _insight:
                    st.success(_insight)
else:
    _preview = folium.Map(location=[32.0853, 34.7818], zoom_start=13,
                          tiles="CartoDB positron")
    _pts = []
    if "_origin_ok" in st.session_state:
        _p = st.session_state["_origin_ok"]
        folium.Marker(_p, icon=folium.Icon(color="green", icon="play", prefix="fa"),
                      tooltip="נקודת מוצא").add_to(_preview)
        _pts.append(_p)
    if "_dest_ok" in st.session_state:
        _p = st.session_state["_dest_ok"]
        folium.Marker(_p, icon=folium.Icon(color="red", icon="flag", prefix="fa"),
                      tooltip="יעד").add_to(_preview)
        _pts.append(_p)
    if len(_pts) == 2:
        _lats = [p[0] for p in _pts]
        _lons = [p[1] for p in _pts]
        _preview.fit_bounds([[min(_lats) - 0.005, min(_lons) - 0.005],
                              [max(_lats) + 0.005, max(_lons) + 0.005]])
    st_folium(_preview, height=520, use_container_width=True, returned_objects=[])

st.caption("ניווט מבוסס: OSMnx · Nominatim · NetworkX Dijkstra · RandomForest TCI")


st.divider()
with st.expander("📎 חומרי רקע ותיעוד (למטרות הערכה בקורס)", expanded=False):
    st.caption("חלק זה אינו חלק מחוויית הניווט המרכזית — מסמך דרישות הקורס.")

    with st.expander("ℹ️ אודות SHADY"):
        st.title("ℹ️ אודות SHADY")

        st.markdown(
            """
        <div style="background:#fdf6e3;border-right:6px solid #d68910;padding:18px 22px;border-radius:8px;margin:10px 0 22px 0;direction:rtl;text-align:right;">
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

        st.subheader("👩‍💻 מי אנחנו")
        st.markdown(
            """
        אנחנו **אליסה ורוני** — סטודנטיות בטכניון שאוהבות ללכת ברגל, אבל לא בכל מחיר.
        מהקיץ הישראלי והתסכול בחוץ נולד הרעיון של SHADY — חיפוש מסלול שמתעדף נוחות על פני מהירות.
        כך אנחנו שמות דגש על החוויה של ההליכה ונמנעות מלהזיע או לסבול ברחוב בשעות היום.
        """
        )

        st.divider()

        st.subheader("🧭 איך משתמשים?")
        st.markdown(
            """
        💬 **סוכן חכם** — הקלידי בקשה חופשית כמו *"מכיכר רבין לשוק הכרמל ב-3 אחה״צ"*
        והסוכן ימלא את הפרטים אוטומטית. לחצי **"שלח לסוכן 🤖"**.

        📍 **ידנית** — הקלידי כתובת מוצא ויעד, בחרי תאריך ושעת יציאה (בהפרשים של 30 דקות),
        ובחרי את רמת העדפת הצל:

        | אפשרות | משמעות |
        |--------|---------|
        | 🚶 מעט צל | ברירת מחדל, מינימום התחשבות בצל |
        | 🌳 הרבה צל | הצל מנצח הכל, גם במחיר מסלול ארוך יותר |

        סמני ✅ **"הצג גם את המסלול המהיר ביותר להשוואה"** כדי לראות את שני המסלולים זה לצד זה,
        לחצי **"מצא מסלול 🗺️"** — וסיימתן!

        *\* מנוסח בלשון נקבה אך פונה לכולם*
        """
        )

        st.divider()

        with st.expander("🗄️ מקורות נתונים"):
            st.markdown(
                """
            | שכבה | מקור |
            |------|------|
            | מבנים | עיריית ת"א (opendata.tel-aviv.gov.il) |
            | חופת עצים | מפ"י (data.gov.il) |
            | רשת רחובות | OSMnx |
            | מזג אוויר | Open-Meteo API |
            """
            )

        st.divider()

        with st.expander("🎯 M3 — בחירת KPI: למה RMSE?"):
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

        with st.expander("🏆 M3 — תוצאות: השוואת מודלים ובחירת מנצח"):
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
                            "shadow_cov שולט (מאחד צל-מבנים וצל-עצים לאות פיזי אחד, תלוי-שעה). "
                            "temperature שומר חשיבות כי הוא פרוקסי ל-cloud_cover (קורלציה ≈ −0.99). "
                            "canopy_ratio/building_height הם decoys — shadow_cov המאוחד בלע את תפקידם. "
                            "(קורלציה ≠ חשיבות)"
                        )

                st.info(
                    "🔎 **מהימנות:** ה-R² הגבוה נובע מכך שיעד ה-TCI מחושב כרגע מנוסחה אנליטית "
                    "(יעד סינתטי) — המודל משחזר את הנוסחה. זה צעד ביניים מתוכנן; המבחן האמיתי "
                    "יגיע עם תוויות אמת מדודות."
                )

        st.divider()

        with st.expander("🌡️ מדד הנוחות התרמית (TCI)"):
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
                | `shadow_cov` | כיסוי-צל מבנים [0–1] — אחוז הקשת המכוסה בצל (מחושב מראש) |
                """
                )

    with st.expander("🎯 למידת הבעיה"):
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


    with st.expander("📚 סקירת ספרות"):
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



    with st.expander("🏪 סקר שוק"):
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



    with st.expander("📊 ניתוח נתונים"):
        buildings = load_buildings()
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
            'פיצ\'רים: <code style="direction:ltr;display:inline-block;">sun_altitude, shadow_cov</code>'
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
<b>נוסחה אנליטית:</b><br>
<code style="background:#dbeafe;padding:6px 10px;border-radius:4px;font-size:14px;display:inline-block;margin:6px 0;">
TCI = clip( 1 + 9 &times; (sun_altitude/80) &times; (1 &minus; cloud_cover/100) &times; (1 &minus; shadow_cov), &nbsp;1, 10 )
</code>
<br><br>
<b>פירוש הגורמים:</b>
שמש גבוהה → (alt/80) קרוב ל-1 → TCI עולה (חם) ·
צל (מבנים+עצים) → shadow_cov גבוה → TCI יורד — shadow_cov מאוחד: מצולעי-צל אמיתיים
של מבנים ועצים כאחד, מחושב מראש לפי שעה (ר' precompute_shadow.py). <code>canopy_ratio</code>
נשאר כפיצ'ר-decoy (כמו temperature/humidity) — ההצללה הישירה של עצים כלולה כעת ב-shadow_cov.
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
            ('tree_canopy_ratio',    'מפ"י',       "0–1",     "— decoy: הצללת עצים כלולה כעת ב-shadow_cov המאוחד", "#7f8c8d"),
            ("temperature",          "Open-Meteo", "13–28°C", "↑ עולה — חום מגביר אי-נוחות",   "#e74c3c"),
            ("cloud_cover",          "Open-Meteo", "0–50%",   "↓ יורד — עננות מסננת קרינה",    "#27ae60"),
            ("humidity",             "Open-Meteo", "65–80%",  "↑ קל — לחות מגבירה תחושת חום",  "#e67e22"),
            ("shadow_cov",            "precompute_shadow.py", "0–1", "↓ יורד — כיסוי-צל מאוחד (מבנים+עצים) גדול = TCI נמוך", "#27ae60"),
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
            'בחירת שעה לאורך היום (קיץ, 6:00→19:00) למטה · הצללת מבנים מחושבת מראש (כיסוי-צל) · עננות קבועה 5% · '
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
                _hour_sel = st.slider(
                    "שעה לאורך היום (קיץ, 6:00→19:00)",
                    min_value=6.0, max_value=19.0, value=13.0, step=0.5, key="tci_hour",
                )
            # מיקום שמש אמיתי לתאריך-ייחוס (מיובא מ-src.data — מקור אמת יחיד, לא כפול); IDT=UTC+3
            if _PYSOLAR:
                from src.data import REF_DATE as _REF_DATE
                _h = int(_hour_sel); _mn = int(round((_hour_sel - _h) * 60))
                _dt_ref = datetime(*_REF_DATE, _h - 3, _mn, tzinfo=timezone.utc)
                _sun_sel = float(_solar.get_altitude(32.08, 34.77, _dt_ref))
                _SUN_AZ_REF = float(_solar.get_azimuth(32.08, 34.77, _dt_ref))
            else:
                _sun_sel, _SUN_AZ_REF = 60.0, 180.0
            with _c3:
                # כיוון השמש: בוקר=מזרח, צהריים=דרום, אחה"צ=מערב. המבנים מצילים מהצד שממנו השמש.
                if _SUN_AZ_REF < 135:
                    _sun_side = "ממזרח (בוקר) → צל מצד מזרח"
                elif _SUN_AZ_REF > 225:
                    _sun_side = "ממערב (אחה\"צ) → צל מצד מערב"
                else:
                    _sun_side = "מדרום (צהריים) → צללים קצרים"
                st.caption(f"השמש בשעה זו: גובה {_sun_sel:.0f}° · אזימוט {_SUN_AZ_REF:.0f}° — {_sun_side}.")
            _CLOUD_NOON = 0.05

            _bundle = load_tci_model() if _tci_source == "מודל ML" else None
            if _tci_source == "מודל ML" and _bundle is None:
                st.warning("⚠️ המודל לא נמצא — הרץ `python -m src.model`. מוצגת נוסחה אנליטית.")

            # אזימוט שמש מחושב מהשעה שנבחרה (PySolar) — לא קבוע 180° עוד
            _sun_dir_ref = _SUN_AZ_REF % 180
            _street_az = _ef_s["street_azimuth"].values if "street_azimuth" in _ef_s.columns else np.full(len(_ef_s), 90.0)
            _diff_ref = np.abs(_sun_dir_ref - _street_az)
            _shadow_angle_e = np.minimum(_diff_ref, 180 - _diff_ref)
            _shadow_factor_e = np.sin(np.radians(_shadow_angle_e))

            # כיסוי-צל מבנים מחושב מראש (precompute_shadow.py) — lookup לפי שעה, מיידי.
            # כל מבנה=ריבוע 20מ' בגובהו → מצולע צל אמיתי → אחוז הקשת שנמצא בצל.
            # הכיוון נקבע ע"י השמש, אז רחוב מקביל לשמש יוצא מואר וניצב יוצא מוצל.
            _ef_s = _ef_s.copy()
            _cov_tbl = load_shadow_coverage()
            _hcol = f"{float(_hour_sel):.1f}"
            _has_cov = _cov_tbl is not None and _hcol in _cov_tbl.columns
            if _has_cov:
                _ci = _cov_tbl.set_index(["u", "v", "key"])[_hcol]
                _ef_s["shadow_cov"] = _ci.reindex(
                    list(zip(_ef_s["u"], _ef_s["v"], _ef_s["key"]))).to_numpy()
            else:
                _ef_s["shadow_cov"] = np.nan
            _ef_s["shadow_cov"] = pd.to_numeric(_ef_s["shadow_cov"], errors="coerce").fillna(0.0)

            if _bundle is not None:
                # אפשרות C: חיזוי TCI פר-edge ע"י מודל ה-ML (כל מקטע = שורה)
                _n = len(_ef_s)
                _X = pd.DataFrame({
                    "sun_altitude":    np.full(_n, max(_sun_sel, 0.0)),
                    "building_height": _ef_s["mean_building_height"].values,
                    "canopy_ratio":    _ef_s["tree_canopy_ratio"].values,
                    "cloud_cover":     np.full(_n, _CLOUD_NOON * 100),
                    "temperature":     np.full(_n, 30.0),
                    "humidity":        np.full(_n, 60.0),
                    "shadow_cov":      _ef_s["shadow_cov"].values,  # אותו אות צל כמו במצב האנליטי
                })[_bundle["features"]]
                _ef_s["tci"] = np.clip(_bundle["model"].predict(_X), 1, 10)
            else:
                # נוסחה אנליטית — הצללת מבנים = כיסוי-צל מראש (מצולעי צל אמיתיים)
                _sa_r = np.radians(_sun_sel)
                if _has_cov:
                    _bf_e = _ef_s["shadow_cov"].values
                else:
                    # fallback אם טבלת ה-precompute חסרה
                    _bf_e = np.clip(_ef_s["mean_building_height"] / 30, 0, 1) * np.cos(_sa_r) * _shadow_factor_e
                # שלב 2 (2026-07-10): shadow_cov מאוחד (מבנים+עצים) — tree_canopy_ratio
                # הוסר מהנוסחה (היה מודד את אותה הצללה בצורה סטטית/כפולה), נשאר decoy בלבד.
                _ef_s["tci"] = np.clip(
                    1 + 9 * (_sun_sel / 80) * (1 - _CLOUD_NOON) * (1 - _bf_e),
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
            _ef_plot = _ef_s[["geometry", "tci", "shadow_cov", "tree_canopy_ratio"]].copy()
            _ef_plot["shadow_cov"]           = (_ef_plot["shadow_cov"].fillna(0) * 100).round(0)
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
                    fields=["tci", "shadow_cov", "tree_canopy_ratio"],
                    aliases=["TCI", "כיסוי צל מבנים (%)", "חופת עצים"],
                ),
            ).add_to(_m_tci)

            st_folium(_m_tci, height=540, use_container_width=True, returned_objects=[])
            st.caption(f"מוצגות {len(_ef_s):,} קשתות באזור רוטשילד–לב העיר")
        else:
            st.warning(
                "⚠️ `data/edges_features.parquet` חסר — הרץ `python precompute_features.py` "
                "פעם אחת ליצירת הנתונים המרחביים."
            )


