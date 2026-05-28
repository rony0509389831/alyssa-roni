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
tab_problem, tab_eda, tab_map, tab_nav, tab_about = st.tabs([
    "🎯 למידת הבעיה", "📊 ניתוח נתונים", "🗺️ מפה", "🚶 ניווט", "ℹ️ אודות"
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
# TAB 1 — EDA
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
