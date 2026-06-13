# CLAUDE.md

## Project Purpose
SHADY — מערכת ניווט עירוני להולכי רגל המוצאת מסלולים מוצלים על בסיס גבהי מבנים, חופת עצים, מיקום השמש ומזג האוויר בזמן אמת.

---

## Data Sources

| מקור | פורמט | תוכן עיקרי |
|------|--------|------------|
| עיריית ת"א (opendata.tel-aviv.gov.il) | GeoJSON | ~45,900 פוליגוני מבנים עם `gova_simplex_2019` (גובה), `ms_komot` (קומות) — קובץ: `tel_aviv_buildings.geojson` בשורש |
| מפ"י (data.gov.il) | Parquet | ~231,000 פוליגוני חופת עצים באזור ת"א — קובץ: `data/national_canopy_clean.parquet` |
| OSMnx | GraphML | רשת רחובות ת"א — cache ב-`data/tel_aviv_walk.graphml` |
| Open-Meteo API | JSON (שעתי) | `temperature_2m`, `relative_humidity_2m`, `cloud_cover` |
| Precomputed (spatial.py) | Parquet | פיצ'רים לכל קשת רחוב — `data/edges_features.parquet` |

שימוש גם בספריות פייתון הבאות עבור ממשק משתמש, זווית שמש וכו':
pandas, numpy, streamlit, requests, matplotlib,
geopandas, osmnx, folium, streamlit-folium, pyarrow, pyogrio,
pysolar, scikit-learn, networkx, pytest

**חוסרים ידועים:** ~4% מגבהי המבנים הם NaN — מטופל ע"י נוסחת imputation: `height = 2.95 * floors + 5.21` (R²≈0.49).

---

## ML Model

- **משימה:** רגרסיה — חיזוי Thermal Comfort Index (ציון 1–10) לכל קשת בגרף הרחובות.
- **פיצ'רים (7, מקור אמת = `build_tci_df` ב-`app.py`/`src/data.py`):** `sun_altitude`, `building_height`, `canopy_ratio`, `cloud_cover`, `temperature`, `humidity`, `azimuth`.
  - בטבלת האימון השמות הם `building_height`/`canopy_ratio`; בקובץ `edges_features.parquet` הם `mean_building_height`/`tree_canopy_ratio`. אין `sun_azimuth`.
  - `temperature`/`humidity`/`azimuth` אינם נכנסים לנוסחת ה-TCI האנליטית (decoys) — המודל אמור ללמוד להוריד להם משקל.
- **חישוב Sun Position:** PySolar (משולב ישירות ב-`app.py` ו-`notebooks/01_eda.ipynb`)
- **נוסחה analytic נוכחית (MVP):**
  ```
  TCI = 1 + 9 * (sun_altitude/80) * (1 - cloud_cover/100) * (1 - 0.6*canopy_ratio - 0.4*building_factor)
  ```
  מחושבת ב-`build_tci_df()` ב-`app.py`; טווח מוצמד ל-[1, 10].
- **`src/data.py`** — `build_tci_df()`: בונה טבלת אימון (7 פיצ'רים + TCI) מ-`edges_features.parquet`, ללא Streamlit (גרסה נקייה של הפונקציה שב-`app.py`).
- **`src/model.py`** — מממש M3 שלבים 3–8: `load_train_test`, `evaluate` (RMSE+R²), `run_baselines` (DummyRegressor), `build_models` (3 Pipelines), `train_and_evaluate`, `select_winner`, `save_model`. מנצח: **RandomForest (RMSE≈0.12 על test)** → נשמר ל-`data/tci_model.joblib` ונטען ב-app.py (גרף 4, מפת ML פר-edge). הרצה: `python -m src.model`.
- **Loss:** MSE | **מטריקה (KPI):** RMSE (ראשי) + R² (בונוס)
- **חלוקה:** כרגע (M3) 80/20 train/test, `random_state=42`, פיצול לפי שורה (לא לפי רחוב). תכנון עתידי: Train 70% / Val 15% / Test 15%.
- **Baseline (שלב 5):** `DummyRegressor` — mean (רצפה ראשית, RMSE≈1.77 על test) + median (רצפה משנית; התפלגות TCI מוטה ימינה, skew≈0.78). Linear Regression / Random Forest הם **מודלים מועמדים** לשלב 6, לא baseline. ניווט: Dijkstra גיאומטרי.
- **ספריה:** Scikit-Learn

---

## Architecture & Stack

```
tel_aviv_buildings.geojson + data/national_canopy_clean.parquet
        ↓
precompute_features.py → src/spatial.py (Spatial Join, buffer=5m)
        ↓
data/edges_features.parquet  +  Open-Meteo + PySolar
        ↓
Feature Vector → TCI (analytic / ML model)
        ↓
Edge Weights → OSRM API (routing) → Optimal Route
        ↓
Route → Folium Map → Streamlit UI (7 tabs)
```

**Fallback:** אם Open-Meteo לא זמין — טעינת נתוני אקלים ממוצעים מ-`data/climate_fallback.json`.

---

## App Structure (7 Tabs)

| Tab | תפקיד |
|-----|--------|
| 🎯 למידת הבעיה (M2 Component 1) | פרסונה (נועה, 28), מסע משתמש לפני/אחרי |
| 📚 סקירת ספרות (M2 Component 2) | 4 מאמרים מרכזיים (Boeing 2017, Park 2021, Lin 2012, Sulzer 2025) |
| 🏪 סקר שוק (M2 Component 3) | 5 מתחרים: Google Maps, Citymapper, Strava, Shadowmap, CoolWalks |
| 📊 ניתוח נתונים (M2 Component 4) | EDA מבנים, עצים, מזג אוויר, זוויות שמש |
| 🗺️ מפה | Folium אינטראקטיבי — מבנים + עצים + שכונת הצפון הישן |
| 🚶 ניווט | Geocoding (OSMnx/Nominatim) + OSRM routing + מפת מסלול |
| ℹ️ אודות | מידע כללי על הפרויקט |

---

## File Structure (בפועל)

```
alyssa-roni/
├── app.py                      # Streamlit entry point (7 tabs, 2094 שורות)
├── tel_aviv_buildings.geojson  # נתוני מבנים גולמיים מהעירייה (47.9 MB)
├── precompute_features.py      # הרצה חד-פעמית: יוצר edges_features.parquet
├── download_buildings.py       # הורדת buildings מ-API עירוני
├── check_data.py               # בדיקת זמינות נתונים
├── data/
│   ├── buildings_clean.csv         # 45,783 מבנים לאחר ניקוי + imputation
│   ├── national_canopy_clean.parquet # 231,234 עצים
│   ├── edges_features.parquet      # פיצ'רים לקשתות (mean_building_height, tree_canopy_ratio)
│   ├── tci_model.joblib            # מודל TCI מאומן (RandomForest) — נטען ב-app.py
│   ├── model_results.json          # תוצאות השוואת המודלים — מוצג בטאב אודות בסטרימליט
│   ├── tel_aviv_walk.graphml       # רשת רחובות (cache מ-OSMnx)
│   ├── climate_fallback.json       # ממוצעים חודשיים (T, humidity, cloud_cover)
│   └── screenshots/                # PNG לממשק
├── src/
│   ├── data.py             # build_tci_df() — טבלת אימון TCI מ-edges_features.parquet (ללא Streamlit)
│   ├── clean_buildings.py  # ניקוי שכבת המבנים
│   ├── eda_buildings.py    # EDA וגרפים לשכבת המבנים
│   ├── spatial.py          # compute_edge_features() — Spatial Join, buffer=5m
│   ├── model.py            # M3 שלבים 3-8: train/test, baseline, 3 מודלים, בחירת מנצח, שמירה (joblib)
│   ├── routing.py          # load_graph(), geocode_address(), compute_route() (OSRM)
│   └── weather.py          # get_current_weather() — Open-Meteo + fallback
├── notebooks/
│   └── 01_eda.ipynb        # EDA: מבנים, עצים, זוויות שמש, מזג אוויר
├── outputs_M2/             # פלטים ממשימה 2
├── tests/                  # בדיקות
├── requirements.txt
├── WORKLOG.md              # יומן עבודה לפי מושבים (תאריך, מה נעשה, בעיות ופתרונן)
└── CLAUDE.md
```

---

## Coding Conventions

- **שפת הערות בקוד:** עברית (הסברים מהותיים בלבד, לא תיאורי)
- **שמות משתנים:** אנגלית, snake_case
- **שמות גיאו-משתנים:** להשתמש בשמות עקביים עם שדות ה-GeoJSON המקוריים (`gova_simplex_2019`, `ms_komot`)
- **CRS:** EPSG:2039 (Israel TM Grid) לפעולות מרחביות; WGS84 לפלט Folium
- **Thermal Comfort Index:** תמיד בטווח [1, 10]; 1=נוח/מוצל, 10=חשיפה מלאה לשמש
- **MVP approach:** קודם לבנות ניווט עובד עם נוסחה אנליטית, אחר כך להחליף במודל ML

---

## Key Decisions (לא להסביר מחדש)

- Spatial Join מבוצע עם `buffer=5m` סביב כל קשת רחוב כדי להתגבר על Spatial Mismatch בין OSM לשכבות GIS
- גבהי מבנים חסרים מחושבים ע"י נוסחה (לא ML): `height = 2.95 * floors + 5.21`
- קובץ המבנים הראשי (`tel_aviv_buildings.geojson`) נמצא בשורש הפרויקט, לא ב-`data/`
- פיצ'רי הקשתות מחושבים מראש ונשמרים ב-`data/edges_features.parquet` (דרך `precompute_features.py`)
- רשת הרחובות נשמרת ב-cache ב-`data/tel_aviv_walk.graphml` כדי להימנע מהורדה חוזרת
- ניווט נוכחי: OSRM API (demo server, foot routing) — Dijkstra עם TCI weights הוא שלב עתידי
- חישובי צל מבוססים על גיאומטריה 2D (לא סימולציית 3D מלאה)
- לוגיקת בניית טבלת האימון (`build_tci_df`) הוצאה ל-`src/data.py` כי ייבוא מ-`app.py` מריץ את כל אפליקציית Streamlit; קיימת כרגע כפילות מכוונת בין השניים
- ה-baseline של M3 הוא `DummyRegressor(strategy="mean")` (לא Linear Regression) — הממוצע ממזער RMSE בהגדרה, ולכן הוא הרצפה ההוגנת תחת ה-KPI; median נבדק כרצפה משנית בגלל הטיה ימנית
- היעד הסינתטי: ה-TCI מחושב מהנוסחה האנליטית, אז מודל ה-ML בשלב 6 בעצם לומד לשחזר אותה — צעד ביניים מתוכנן עד שיהיו תוויות אמת מדודות
- נוסחת ה-TCI מתעלמת מכיוון השמש (תופסת רק אורך צל דרך `cos(sun_altitude)`). שיפור עתידי: אינטראקציה זווית-יחסית שמש↔רחוב (`|sun_azimuth − street_azimuth|` × גובה מבנה) — רק אז `sun_azimuth` הופך פיצ'ר משמעותי. כרגע decoy, הושמט בצדק
