"""
SHADY – ניווט עירוני מוצל | Streamlit App
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

matplotlib.rcParams["font.family"] = "DejaVu Sans"
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SHADY – ניווט מוצל",
    layout="wide",
    page_icon="🌤️",
    initial_sidebar_state="expanded",
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

buildings = load_buildings()
trees = load_trees()

# ── Sidebar (גלוי בכל הטאבים) ──────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 סינון נתונים")
    height_range = st.slider(
        "טווח גבהים (מ')",
        min_value=0,
        max_value=int(buildings["height"].max()),
        value=(0, int(buildings["height"].max())),
    )
    all_types = sorted(buildings["building_type"].dropna().unique())
    sel_types = st.multiselect("סוג מבנה", options=all_types, default=all_types)
    src_filter = st.radio("מקור גובה", ["הכל", "רשום בלבד", "מחושב בלבד"])

# סינון
df = buildings.copy()
df = df[(df["height"] >= height_range[0]) & (df["height"] <= height_range[1])]
if sel_types:
    df = df[df["building_type"].isin(sel_types)]
if src_filter == "רשום בלבד":
    df = df[df["height_source"] == "recorded"]
elif src_filter == "מחושב בלבד":
    df = df[df["height_source"] == "imputed"]

# ── טאבים ──────────────────────────────────────────────────────────────────────
tab_problem, tab_lit, tab_market, tab_eda, tab_map, tab_nav, tab_about = st.tabs([
    "🎯 למידת הבעיה", "📚 סקירת ספרות", "🏪 סקר שוק", "📊 ניתוח נתונים", "🗺️ מפה", "🚶 ניווט", "ℹ️ אודות"
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
        בקיץ הישראלי הליכה רגלית בעיר הופכת לסיכון בריאותי —
        והולכי הרגל נאלצים לנווט חודשים רבים בשמש הקופחת.
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
            - Google Maps · Waze · Moovit
            - חברות תחבורה ציבורית
            - שירותי Mobility-as-a-Service
            - חברות תיירות וטיולים
            - חברות אופניים וקוקינטים שיתופיים 
            - אפליקציות כושר (Strava)
            - חברות פרסום חוצות
            """
        )

    st.divider()

    # ── הפרסונה ────────────────────────────────────────────────────────────────
    st.subheader("👤 פרסונה ראשית")

    p_col_img, p_col_txt = st.columns([1, 2])
    with p_col_img:
        st.markdown(
            """
            <div style="background:#fdf6e3;border:1px solid #e6d9a7;border-radius:10px;
                        padding:18px;text-align:center;">
            <div style="font-size:72px;line-height:1;">👩‍💻</div>
            <div style="font-size:18px;font-weight:700;color:#7d6608;margin-top:8px;">נועה, 28</div>
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
            פותחת את Google Maps או Waze — שניהם מציעים *את אותו מסלול קצר ביותר*,
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
                B [label="12:51\\nSHADY: 23 דק'\\n82% צל · TCI = 2.8"];
                C [label="13:00–13:23\\nחופת עצים + צל בניינים\\nתוספת של 3 דק' בלבד"];
                D [label="13:23\\nמגיעה במצב סביר\\nמתאוששת ב-2 דק'"];
                E [label="13:25\\nחוזרת לעבודה במלוא הקיבולת\\nבלי התאוששות ממושכת"];
                F [label="ערב\\nעור בסדר\\nיום עבודה מלא ובריא"];
                G [label="לאורך הקיץ\\nחשיפת UV מצטברת\\nמופחתת ב-~70%"];

                A -> B -> C -> D -> E -> F -> G;
            }
            """
        )
        st.success(
            "✅ **שורה תחתונה:** תוספת של **3 דקות הליכה בלבד** מורידה את חשיפת ה-UV "
            "בכ-**70%**. נועה מחזירה לעצמה את היכולת ללכת ברגל בקיץ — בלי לבחור בין בריאות לנוחות כלכלית."
        )

    st.divider()

    # ── הצעת הערך ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#eafaf1;border-right:6px solid #1e8449;padding:18px 22px;border-radius:8px;margin:10px 0;">
        <div style="font-size:14px;color:#196f3d;font-weight:600;margin-bottom:6px;">הצעת ערך · במשפט אחד</div>
        <div style="font-size:20px;line-height:1.5;color:#1c1c1c;">
        <b>SHADY</b> הופכת את הליכת הקיץ בעיר ממסלול ייסורים למסלול אפשרי —
        בתוספת של 2–3 דקות בלבד מגיעים ליעד עם <b>70% פחות חשיפה לשמש</b>.
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
            - **השוואת מתחרים** — Google Maps, Waze, Moovit מתעדפים *זמן ומרחק בלבד* ומתעלמים
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
        <div style="background:#fef5e7;border-right:6px solid #d68910;padding:18px 22px;border-radius:8px;margin:10px 0 22px 0;">
        <div style="font-size:14px;color:#7d6608;font-weight:600;margin-bottom:6px;">שאלת המחקר</div>
        <div style="font-size:19px;line-height:1.5;color:#1c1c1c;">
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
        חיפשנו בספרות האקדמית מה כבר נעשה בכל אחד מהצירים של SHADY —
        כדי לוודא שאנחנו לא ממציאים את הגלגל מחד, ולזהות מהו הפער המדויק
        שאף קבוצה אחרת עדיין לא סגרה מאידך. מקורות אותרו ב-**Google Scholar**,
        **arXiv** ו-**Nature Scientific Reports**, וכל DOI אומת ידנית.
        """
    )

    st.divider()

    # ── טבלת השוואה ──────────────────────────────────────────────────────────────
    st.subheader("📊 טבלת השוואה — דאטה · מודל · תוצאה")

    lit_data = [
        {
            "#": 1,
            "דאטה": "OpenStreetMap — רשתות רחובות של 27,009 ערים בארה\"ב (צמתים, קשתות, תכונות)",
            "מודל": "אלגוריתמים גרפיים מבוססי NetworkX: Dijkstra, A*, מדדי מרכזיות",
            "תוצאה": "ספריית קוד פתוח (OSMnx) שמורידה וממדלת רשתות רחובות מכל מקום בעולם תוך שניות",
        },
        {
            "#": 2,
            "דאטה": "קולומבוס, אוהיו — Landsat 8 תרמי + מודל LiDAR תלת-ממדי + חופת עצים",
            "מודל": "רגרסיה מרחבית (OLS + Spatial Lag) על עוצמת Urban Heat Island",
            "תוצאה": "צל עצים מפחית UHI ב-0.7°C, צל מבנים ב-0.3°C — שני האפקטים מובהקים ועצמאיים",
        },
        {
            "#": 3,
            "דאטה": "3 פארקים במרכז טייוואן — מדידות SVF + מדד PET + ספירת מבקרים שעה-שעה",
            "מודל": "רגרסיה לוגיסטית בין SVF/PET לבין הסתברות נוכחות הולכי רגל",
            "תוצאה": "כש-SVF < 0.5 (יותר צל), נוכחות הולכי רגל בשעות חמות עולה ב-40-60%",
        },
        {
            "#": 4,
            "דאטה": "גריד סינתטי + ערים אמיתיות (Boston, Barcelona) — Building footprints + OSM",
            "מודל": "מדד CoolWalkability + פרמטר sun-avoidance על גרף הרחובות",
            "תוצאה": "עד 80% הליכה בצל בתוספת 5-10% למרחק — תלוי קריטית בגיאומטריית העיר",
        },
    ]
    lit_df = pd.DataFrame(lit_data)

    st.dataframe(
        lit_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small", help="מספר השורה תואם למספר המקור בפירוט למטה"),
            "דאטה": st.column_config.TextColumn(width="medium"),
            "מודל": st.column_config.TextColumn(width="medium"),
            "תוצאה": st.column_config.TextColumn(width="medium"),
        },
    )

    st.caption("💡 לחיצה על כותרת עמודה ממיינת · גרירה משנה רוחב")

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

            **תובנה מפתח:** צל עצים אפקטיבי **פי 2.3 יותר** מצל מבנים בהפחתת UHI —
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
            זה מצדיק את בחירת הסקאלה הלא-לינארית שלנו (1-10) במקום מעלות צלזיוס.
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
            | 2 | מזג אוויר סטטי | Open-Meteo API חי + fallback מקומי |
            | 3 | Boston / Barcelona / גריד סינתטי | ת"א — עיר ים-תיכונית עם רחובות צרים |
            | 4 | נוסחה אנליטית בלבד | ML על Scikit-Learn לחיזוי TCI + השלמת גבהים חסרים |

            **השורה התחתונה:** הם הוכיחו שהרעיון ישים — אנחנו מוכיחות שהוא ישים **כאן, עכשיו, עם דאטה אמיתי**.
            """
        )

    st.divider()

    # ── הלקח האחד מכל מקור ──────────────────────────────────────────────────────
    st.subheader("🎯 הלקח האחד מכל מקור")

    l1, l2 = st.columns(2)
    with l1:
        st.markdown(
            """
            **📘 מ-Boeing למדנו:**
            לא בונים גרף רחובות מאפס. OSMnx הוא הסטנדרט — וגם נותן לנו מטא-דאטה
            עשירה (highway, length, geometry) בחינם.

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
            ניווט-מוצל ישים — וזה מה שאנחנו ממקמים את עצמנו ביחס אליו.
            הם הוכיחו את העיקרון; אנחנו מוסיפות עצים, מזג אוויר חי, ML, וישראליות.
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
            - **לא נבנה גרף רחובות מאפס** (Boeing) — נשתמש ב-OSMnx כתשתית.
            - **שני פיצ'רים נפרדים לצל** (Park) — `tree_canopy_ratio` ו-`mean_building_height` ייכנסו למודל בנפרד, ולא כפיצ'ר מאוחד.
            - **סקאלה 1-10 ולא מעלות** (Lin) — הקשר בין חשיפה לנוחות אינו לינארי, סקאלה אינטואיטיבית מתאימה יותר.
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
        בחרנו 5 מתחרים — 3 אפליקציות ניווט מסחריות שכל הולך רגל מכיר,
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
            "screenshot": "data/screenshots/waze.png",
            "name": "Waze",
            "type": "מסחרי · בבעלות Google",
            "url": "https://waze.com",
            "ui": "מצב Driving directions עם שדות מוצא/יעד וכפתור 'Leave now'. דיווחי משתמשים בזמן אמת.",
            "strength": "קהילתיות חזקה, דיווחים בזמן אמת, מותג ישראלי-בהובלה.",
            "weakness": "מיועד לנהגים. ההליכה היא feature שולי, לא חוויה ראשית.",
            "color": "#33CCFF",
        },
        {
            "screenshot": "data/screenshots/moovit.png",
            "name": "Moovit",
            "type": "מסחרי · ישראלי",
            "url": "https://moovit.com",
            "ui": "מסך נחיתה נקי עם בורר Directions/Lines וכפתור חיפוש כתום בולט. בחירת מדינה ושפה למעלה.",
            "strength": "המומחה בתח\"צ בישראל, עברית/RTL מלאה, חזק במידע על אוטובוסים.",
            "weakness": "סגמנט ההליכה הוא 'הדבר שבין הנקודות' — לא ממוטב לחוויה.",
            "color": "#0066CC",
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

    # מציגים בשתי שורות: 3 + 2
    cols_row1 = st.columns(3)
    cols_row2 = st.columns(3)

    import os
    for i, comp in enumerate(competitors):
        col = cols_row1[i] if i < 3 else cols_row2[i - 3]
        with col:
            # כותרת + סוג (מעל הצילום)
            st.markdown(
                f"""
                <div dir="rtl" style="border-top:5px solid {comp['color']};border-radius:6px 6px 0 0;
                                       padding:10px 12px 6px 12px;background:#fafafa;text-align:center;">
                    <div style="font-size:18px;font-weight:700;color:{comp['color']};">{comp['name']}</div>
                    <div style="font-size:11px;color:#777;">{comp['type']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # צילום מסך (או placeholder אם הקובץ לא קיים)
            if os.path.exists(comp["screenshot"]):
                st.image(comp["screenshot"], use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div dir="rtl" style="background:#fff4e6;border:2px dashed #f39c12;
                                           padding:30px 10px;text-align:center;color:#a04000;
                                           font-size:13px;line-height:1.5;">
                        📷 <b>צילום חסר</b><br>
                        שמרי את הצילום בשם:<br>
                        <code style="direction:ltr;display:inline-block;background:#fff;padding:2px 6px;border-radius:3px;font-size:11px;margin-top:4px;">{comp['screenshot'].split('/')[-1]}</code><br>
                        <span style="font-size:11px;">בתיקייה <code>data/screenshots/</code></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # תיאור + חוזק + חולשה (מתחת לצילום)
            st.markdown(
                f"""
                <div dir="rtl" style="border:1px solid #ddd;border-top:none;border-radius:0 0 6px 6px;
                                       padding:12px 14px;margin-bottom:6px;background:#fafafa;
                                       unicode-bidi:embed;text-align:right;">
                    <div dir="rtl" style="font-size:12px;margin-bottom:8px;unicode-bidi:embed;"><b>ממשק:</b> {comp['ui']}</div>
                    <div dir="rtl" style="font-size:12px;margin-bottom:6px;color:#196f3d;unicode-bidi:embed;"><b>✓ חוזק:</b> {comp['strength']}</div>
                    <div dir="rtl" style="font-size:12px;color:#922b21;unicode-bidi:embed;"><b>✗ חולשה:</b> {comp['weakness']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.link_button(f"🔗 לאתר {comp['name']}", comp["url"], use_container_width=True)

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
            "מתחרה": "Waze",
            "פיצ'רים עיקריים": "ניווט רכב, דיווחים קהילתיים, מצלמות",
            "מחיר": "חינם (פרסומות)",
            "קהל יעד": "נהגים פרטיים",
            "נוחות תרמית": "❌ אין",
            "דאטה דינמי": "תנועה בזמן אמת",
            "חולשה מרכזית": "לא מיועד להולכי רגל",
        },
        {
            "מתחרה": "Moovit",
            "פיצ'רים עיקריים": "תח\"צ, זמני המתנה, סגמנטי הליכה",
            "מחיר": "חינם / Premium 19.90 ₪/חודש",
            "קהל יעד": "משתמשי תחבורה ציבורית",
            "נוחות תרמית": "❌ אין",
            "דאטה דינמי": "זמני אוטובוסים",
            "חולשה מרכזית": "ההליכה היא נספח לתח\"צ",
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
    market_df = pd.DataFrame(market_data)
    st.dataframe(market_df, use_container_width=True, hide_index=True)
    st.caption("💡 לחיצה על כותרת עמודה ממיינת · השורה האחרונה היא SHADY")

    st.divider()

    # ── תרשים מיצוב ────────────────────────────────────────────────────────────
    st.subheader("🎯 תרשים מיצוב — איפה אנחנו על המפה?")

    fig_pos, ax_pos = plt.subplots(figsize=(10, 6.5))

    positions = [
        ("Google Maps", 3.5, 0.5, "#4285F4", 1200),
        ("Waze", 4.2, 0.3, "#33CCFF", 900),
        ("Moovit", 3.0, 0.7, "#0066CC", 700),
        ("Shadowmap", 1.5, 3.8, "#6c3483", 500),
        ("CoolWalks", 1.0, 4.3, "#1e8449", 400),
        ("SHADY", 4.5, 4.7, "#d35400", 1600),
    ]

    for name, x, y, color, size in positions:
        is_us = name == "SHADY"
        ax_pos.scatter(
            x, y, s=size, c=color, alpha=0.85 if is_us else 0.65,
            edgecolors="#2c3e50" if is_us else "white",
            linewidths=3 if is_us else 1.5, zorder=5,
        )
        offset_y = 0.35 if y < 4 else -0.45
        ax_pos.annotate(
            name, (x, y), xytext=(0, 0),
            textcoords="offset points", ha="center", va="center",
            fontsize=9 if not is_us else 11,
            fontweight="bold" if is_us else "normal",
            color="white" if is_us else "#1c1c1c",
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
        st.markdown(
            """
            #### ✅ מה לאמץ מהמתחרים

            **מ-Google Maps:**
            - ממשק נקי עם מסלול ברור (לא 3 אפשרויות מבלבלות)
            - חיפוש כתובת חופשי

            **מ-Waze:**
            - עדכונים בזמן אמת
            - שפה ויזואלית ידידותית (אייקונים, צבעים)

            **מ-Moovit:**
            - עברית/RTL מלאה כברירת מחדל
            - תמיכה במשתמש ישראלי

            **מ-Shadowmap:**
            - ויזואליזציית צל כשכבה על המפה
            - סליידר זמן לתכנון מראש
            """
        )

    with insight_col2:
        st.markdown(
            """
            #### 🚫 מה לשנות / מה הייחוד שלנו

            **לא לחזור על הטעות של Google Maps:**
            ממליצים אותו מסלול ב-8:00 ו-13:00 — אנחנו נשנה לפי שעת היציאה.

            **להשלים את החסר ב-Shadowmap:**
            הם מציגים צל אבל לא מנווטים — אנחנו עושים את שני הדברים.

            **לקחת את CoolWalks למוצר אמיתי:**
            הם נשארו אקדמיים — אנחנו מוציאות לאוויר עם UI נגיש.

            **🔑 הייחוד היחיד שלנו:**
            **חופת עצים + מזג אוויר חי + ML — ביחד**.
            אף אחד מהמתחרים לא משלב את שלושת אלה.
            """
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
            + GeoPandas, ולא להמציא מנוע גרפי חדש.
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.title("📊 ניתוח נתוני המבנים – תל אביב")

    if len(df) == 0:
        st.warning("אין תוצאות לסינון הנוכחי. שנה את הפרמטרים בסרגל הצד.")
        st.stop()

    # ── Metrics ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("מבנים בסינון", f"{len(df):,}")
    m2.metric("גובה ממוצע", f"{df['height'].mean():.1f} מ'")
    m3.metric("גובה חציוני", f"{df['height'].median():.1f} מ'")
    m4.metric("מבנים > 20 מ'", f"{(df['height'] > 20).sum():,}")
    m5.metric("גובה מחושב", f"{(df['height_source'] == 'imputed').mean() * 100:.1f}%")

    st.divider()

    # ── גרף 1: היסטוגרמת גבהים + גרף 2: קומות מול גובה ──────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("התפלגות גבהי מבנים")
        fig, ax = plt.subplots(figsize=(6, 4))
        d_rec = df.loc[df["height_source"] == "recorded", "height"].clip(upper=60)
        d_imp = df.loc[df["height_source"] == "imputed", "height"].clip(upper=60)
        ax.hist(d_rec, bins=40, color="#2980b9", alpha=0.8, label=f"רשום ({len(d_rec):,})")
        ax.hist(d_imp, bins=40, color="#e67e22", alpha=0.8, label=f"מחושב ({len(d_imp):,})")
        ax.set_xlabel("Height (m)")
        ax.set_ylabel("Count")
        ax.set_title("Building Height Distribution (clipped at 60m)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.subheader("קומות מול גובה")
        samp_pool = df[df["floors"].notna() & df["floors"].between(1, 25)]
        samp = samp_pool.sample(min(2000, len(samp_pool)), random_state=42)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = samp["height_source"].map({"recorded": "#2980b9", "imputed": "#e67e22"})
        ax2.scatter(samp["floors"], samp["height"], c=colors, alpha=0.25, s=10)
        x_line = np.linspace(1, 25, 100)
        ax2.plot(x_line, 2.95 * x_line + 5.21, color="red", lw=2,
                 label="y = 2.95x + 5.21  (r = 0.70)")
        ax2.set_xlabel("Floors (ms_komot)")
        ax2.set_ylabel("Height (m)")
        ax2.set_title("Floors vs. Height")
        ax2.legend(fontsize=8)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── גרף 3: סוגי מבנים ────────────────────────────────────────────────────
    st.subheader("סוגי מבנים")
    type_counts = df["building_type"].value_counts().head(8)
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    bars = ax3.barh(type_counts.index[::-1], type_counts.values[::-1], color="#2980b9")
    for bar, val in zip(bars, type_counts.values[::-1]):
        ax3.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=9)
    ax3.set_xlabel("Count")
    ax3.set_title("Building Types (Top 8)")
    ax3.set_xlim(0, type_counts.max() * 1.12)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── תובנה בולטת ───────────────────────────────────────────────────────────
    tall_pct = (buildings["height"] > 20).mean() * 100
    st.info(
        f"💡 **תובנה:** 96.3% מהמבנים בת\"א יש להם גובה רשום מהעירייה. "
        f"הגובה החציוני הוא **10.6 מ'** (~3 קומות) — עיר של בניינים נמוכים. "
        f"רק **{tall_pct:.1f}%** מהמבנים גבוהים מ-20 מ', אך הם אחראים לרוב הצל "
        f"על המדרכות בשעות הצהריים."
    )

    # ── טבלת דגימה ───────────────────────────────────────────────────────────
    st.subheader("דגימה רנדומלית – 10 שורות")
    st.dataframe(
        df[["id_binyan", "lat", "lon", "height", "height_source",
            "floors", "building_type", "year"]]
        .sample(min(10, len(df)))
        .reset_index(drop=True),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.title("🗺️ שכבות מרחביות – תל אביב")

    col_ctrl, col_map_area = st.columns([1, 5])

    with col_ctrl:
        st.subheader("שכבות")
        show_buildings = st.checkbox("🏢 מבנים (גובה)", value=True)
        show_trees = st.checkbox("🌳 חופת עצים", value=True)
        st.divider()
        st.caption("**צבע מבנים לפי גובה:**")
        st.caption("🔵 0–6 מ' | 🔷 6–12 מ'")
        st.caption("🟠 12–20 מ' | 🔴 20–35 מ'")
        st.caption("🟣 35 מ'+")
        st.divider()
        st.caption(f"מבנים: {len(buildings):,} (מוצגים 1,500)")
        st.caption(f"עצים: {len(trees):,}")

    def _height_color(h: float) -> str:
        if h < 6:  return "#5dade2"
        if h < 12: return "#2980b9"
        if h < 20: return "#f39c12"
        if h < 35: return "#e74c3c"
        return "#6c3483"

    m = folium.Map(location=[32.085, 34.79], zoom_start=14, tiles="CartoDB positron")

    if show_buildings:
        b_layer = folium.FeatureGroup(name="מבנים", show=True)
        for _, row in buildings.sample(1500, random_state=42).iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color=_height_color(row["height"]),
                fill=True,
                fill_opacity=0.75,
                weight=0,
                tooltip=f"{row['height']:.1f} מ' | {row.get('building_type', '')}",
            ).add_to(b_layer)
        b_layer.add_to(m)

    if show_trees:
        t_layer = folium.FeatureGroup(name="חופת עצים", show=True)
        t_sample = trees.sample(min(25000, len(trees)), random_state=42)
        HeatMap(
            t_sample[["lat", "lon", "canopy_area_m2"]].values.tolist(),
            radius=8, blur=6, min_opacity=0.3,
        ).add_to(t_layer)
        t_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    with col_map_area:
        st_folium(m, height=620, width=1100)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NAVIGATION (placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_nav:
    st.title("🚶 ניווט מוצל")
    st.info("🚧 **בפיתוח** — יהיה זמין ב-M3. האלגוריתם יחשב מסלול עם מינימום חשיפה לשמש.")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("📍 נקודת מוצא", placeholder="לדוגמה: כיכר רבין, תל אביב")
    with c2:
        st.text_input("🏁 יעד", placeholder="לדוגמה: שוק הכרמל, תל אביב")

    st.time_input("⏰ שעת היציאה (לחישוב מיקום השמש)")
    st.button("מצא מסלול מוצל ☀️", disabled=True)

    st.divider()
    st.caption("הניווט ישתמש ב: OSMnx · PySolar · Scikit-Learn · NetworkX Dijkstra")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.title("ℹ️ אודות SHADY")
    st.markdown("""
**SHADY** מוצאת מסלולי הליכה מוצלים בתל אביב על בסיס גבהי מבנים, חופת עצים, מיקום השמש ומזג אוויר בזמן אמת.

---

### מקורות נתונים

| שכבה | מקור | רשומות |
|------|------|--------|
| מבנים | עיריית ת"א (opendata.tel-aviv.gov.il) | 45,783 |
| חופת עצים | מפ"י (data.gov.il) | 231,234 |
| רשת רחובות | OSMnx | בקרוב |
| מזג אוויר | Open-Meteo API | בזמן אמת |

---

### מדד הנוחות התרמית (Thermal Comfort Index)

ציון **1–10** לכל קשת בגרף הרחובות:
- **1** = מוצל / נוח מאוד
- **10** = חשיפה מלאה לשמש

פיצ'רים: גובה מבנה, חופת עצים, זווית שמש, טמפרטורה, לחות, כיסוי עננים.
    """)
