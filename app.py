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
tab_eda, tab_map, tab_nav, tab_about = st.tabs([
    "📊 ניתוח נתונים", "🗺️ מפה", "🚶 ניווט", "ℹ️ אודות"
])

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
        samp = df[df["floors"].notna() & df["floors"].between(1, 25)].sample(
            min(2000, len(df)), random_state=42
        )
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
        with st.spinner("טוען מפה..."):
            st_folium(m, height=620, use_container_width=True)


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
