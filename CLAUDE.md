# CLAUDE.md

## Project Purpose
SHADY — מערכת ניווט עירוני להולכי רגל המוצאת מסלולים מוצלים על בסיס גבהי מבנים, חופת עצים, מיקום השמש ומזג האוויר בזמן אמת.

---

## Data Sources

| מקור | פורמט | תוכן עיקרי |
|------|--------|------------|
| עיריית ת"א (opendata.tel-aviv.gov.il) | CSV | **צנטרואידים** של ~45,783 מבנים (lat/lon + גובה + קומות) — `data/buildings_clean.csv`. ⚠️ קובץ הפוליגונים המקורי (`tel_aviv_buildings.geojson`) **אינו בפרויקט** — יש רק צנטרואידים |
| מפ"י (data.gov.il) | Parquet | ~231,000 פוליגוני חופת עצים באזור ת"א — `data/national_canopy_clean.parquet` |
| OSMnx | GraphML | רשת רחובות ת"א — cache ב-`data/tel_aviv_walk.graphml` |
| Open-Meteo API | JSON (שעתי) | `temperature_2m`, `relative_humidity_2m`, `cloud_cover` |
| Precomputed (spatial.py) | Parquet | פיצ'רים סטטיים לכל קשת — `data/edges_features.parquet` (`mean_building_height`, `tree_canopy_ratio`, `street_azimuth`) |
| Precomputed (precompute_shadow.py) | Parquet | כיסוי-צל מבנים פר קשת × שעה (6:00–19:00, יום קיץ) — `data/shadow_coverage.parquet` |

שימוש גם בספריות פייתון הבאות עבור ממשק משתמש, זווית שמש וכו':
pandas, numpy, streamlit, requests, matplotlib,
geopandas, osmnx, folium, streamlit-folium, pyarrow, pyogrio,
pysolar, scikit-learn, networkx, pytest

**חוסרים ידועים:** ~4% מגבהי המבנים הם NaN — מטופל ע"י נוסחת imputation: `height = 2.95 * floors + 5.21` (R²≈0.49).

---

## ML Model

- **משימה:** רגרסיה — חיזוי Thermal Comfort Index (ציון 1–10) לכל קשת בגרף הרחובות.
- **פיצ'רים (7, מקור אמת = `FEATURE_COLS` ב-`src/data.py`):** `sun_altitude`, `building_height`, `canopy_ratio`, `cloud_cover`, `temperature`, `humidity`, `shadow_cov`.
  - **`shadow_cov` ∈ [0,1] = כיסוי-צל מבנים, מחושב מראש** ב-`precompute_shadow.py` (פר קשת × שעה) ונטען מ-`data/shadow_coverage.parquet`. הוא **החליף את `shadow_angle` הישן** (ר' Key Decisions).
  - בקובץ `edges_features.parquet` השמות הסטטיים הם `mean_building_height`/`tree_canopy_ratio`/`street_azimuth`. `building_height` כיום פיצ'ר עזר (חשיבות ~0.001 — תפקידו נבלע ע"י `shadow_cov`).
  - `temperature`/`humidity` = decoys (לא בנוסחת ה-TCI) — המודל אמור להוריד להם משקל.
- **חישוב Sun Position:** PySolar.
- **נוסחת ה-TCI האנליטית (מקור התוויות):**
  ```
  TCI = clip( 1 + 9 * (sun_altitude/80) * (1 - cloud_cover/100) * (1 - 0.6*canopy_ratio - 0.4*shadow_cov), 1, 10 )
  ```
  `shadow_cov` = אחוז הקשת המכוסה בצל מבנים, ממצולעי footprint אמיתיים: כל מבנה = ריבוע ~20מ' בגובהו, צל נגרר `h/tan(alt)` בכיוון ההפוך לשמש, חיתוך עם הקשת. החליף את `building_height × cos(alt) × sin(shadow_angle)` הישן שתפס רק זווית (וזקף הצללת-שווא ברחובות מזרח-מערב).
- **אות הצל אחיד בכל המערכת:** אותו `shadow_cov` מזין את טבלת האימון, המודל, גרף 4 (אנליטי + ML), והניווט.
- **`src/data.py`** — `build_tci_df()`: דוגם קשת × שעת-ייחוס, מחשב יעד עם `shadow_cov`. `app.py` **מאציל** לכאן (אין יותר כפילות).
- **`src/model.py`** — M3 שלבים 3–8. מנצח: **RandomForest (RMSE≈0.136, R²≈0.996 על test)** → `data/tci_model.joblib`. הרצה: `python -m src.model`.
- **Loss:** MSE | **KPI:** RMSE (ראשי) + R² (בונוס).
- **חלוקה:** 70/15/15 (n=5000 → 3500/750/750), `random_state=42`, פיצול לפי שורה. בחירת מנצח על **val**, דיווח על **test**.
- **Baseline:** `DummyRegressor(mean)` (RMSE≈2.14 על test) + median (משני; TCI מוטה ימינה). Linear/Tree/Forest = מודלים מועמדים.
- **ניווט:** `src/routing.py` — מצב "מוצל" = Dijkstra על גרף OSMnx עם משקל `TCI × אורך` (TCI מהמודל, `shadow_cov` לפי השמש הנוכחית); מצב "מהיר" = OSRM.
- **מגבלה מרכזית (לא לשכוח):** היעד סינתטי — המודל משחזר את הנוסחה (R² מנופח, מעגליות). `shadow_cov` קירב את הנוסחה למציאות אך לא שבר את המעגליות; תווית אמת (LST) היא הצעד הבא.
- **ספריה:** Scikit-Learn

---

## Architecture & Stack

```
buildings_clean.csv (צנטרואידים) + data/national_canopy_clean.parquet
        ↓
precompute_features.py → src/spatial.py
   (Spatial Join: מבנים buffer 25מ' מצנטרואידים · עצים חיתוך-פוליגונים buffer 10מ')
        ↓
data/edges_features.parquet (פיצ'רים סטטיים)
        +
precompute_shadow.py → data/shadow_coverage.parquet (כיסוי-צל פר קשת × שעה)
        ↓
Feature Vector (כולל shadow_cov) + Open-Meteo + PySolar → TCI (analytic / ML model)
        ↓
Edge Weights → ניווט: Dijkstra+TCI ("מוצל") / OSRM ("מהיר")
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
| 🚶 ניווט | Geocoding (Nominatim) + שני מצבים: "מוצל" (Dijkstra+TCI, לפי שמש נוכחית) / "מהיר" (OSRM) + מפת מסלול |
| ℹ️ אודות | מידע כללי על הפרויקט |

---

## File Structure (בפועל)

```
alyssa-roni/
├── app.py                      # Streamlit entry point (7 tabs)
├── precompute_features.py      # הרצה חד-פעמית: יוצר edges_features.parquet
├── precompute_shadow.py        # הרצה חד-פעמית: יוצר shadow_coverage.parquet (כיסוי-צל פר קשת×שעה)
│   # הערה: tel_aviv_buildings.geojson, download_buildings.py, check_data.py — אינם בפרויקט
├── data/
│   ├── buildings_clean.csv         # 45,783 צנטרואידי מבנים (lat/lon+גובה) — מקור הצל
│   ├── national_canopy_clean.parquet # ~231k פוליגוני עצים
│   ├── edges_features.parquet      # פיצ'רים סטטיים לקשתות (mean_building_height, tree_canopy_ratio, street_azimuth)
│   ├── shadow_coverage.parquet     # כיסוי-צל: u,v,key + עמודה לכל שעה (59,086 × 27)
│   ├── tci_model.joblib            # מודל TCI מאומן (RandomForest, פיצ'ר shadow_cov)
│   ├── model_results.json          # תוצאות השוואת המודלים — מוצג בטאב אודות
│   ├── tel_aviv_walk.graphml       # רשת רחובות (cache מ-OSMnx)
│   ├── climate_fallback.json       # ממוצעים חודשיים (T, humidity, cloud_cover)
│   └── screenshots/                # PNG לממשק
├── src/
│   ├── data.py             # build_tci_df() — טבלת אימון TCI (יעד+פיצ'ר shadow_cov), ללא Streamlit
│   ├── clean_buildings.py  # ניקוי שכבת המבנים
│   ├── eda_buildings.py    # EDA וגרפים לשכבת המבנים
│   ├── spatial.py          # compute_edge_features() — מבנים buffer 25מ' (צנטרואידים), עצים buffer 10מ' (חיתוך פוליגונים)
│   ├── model.py            # M3 שלבים 3-8: train/val/test, baseline, 3 מודלים, בחירת מנצח, שמירה
│   ├── routing.py          # load_graph, geocode, compute_route (OSRM), compute_tci_weights + compute_shaded_route (Dijkstra+TCI, lookup shadow_cov, העדפת רחובות ירוקים)
│   └── weather.py          # get_current_weather() — Open-Meteo + fallback
├── notebooks/01_eda.ipynb  # EDA
├── outputs_M2/ · outputs_M3/ · tests/
├── requirements.txt
├── WORKLOG.md              # יומן עבודה לפי מושבים
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

- **הצללת מבנים = `shadow_cov`** (לא `shadow_angle` יותר). כל מבנה=ריבוע 20מ' בגובהו → מצולע צל אמיתי (נגרר `h/tan(alt)` נגד השמש) → אחוז הקשת בצל. מחושב מראש ב-`precompute_shadow.py`. מתקן הצללת-שווא ברחובות מזרח-מערב (שינקין) שהנוסחה הישנה (`sin(shadow_angle)`) יצרה. **אומת ידנית מול Shadowmap** (כלי אימות, לא מקור נתונים).
- **`shadow_cov` הוא אות הצל היחיד** — אותו חישוב באימון, במודל, בגרף 4, ובניווט (אחידות מלאה לאחר הגירה).
- **ייחוס יום-קיץ (27 נקודות זמן, כל חצי שעה 6:00–19:00):** קיץ = השמש הכי גבוהה → צללים קצרים → הכי פחות צל → המבחן המחמיר ביותר לראוטר. **פערים ידועים:** (א) עונתיות השמש (הניווט נצמד למצב-השמש הקרוב) — *הבניינים הם שכבה קבועה, הפער בשמש לא בבניינים*; (ב) ריבוע 20מ' מצנטרואיד ולא קו-מתאר אמיתי.
- **`build_tci_df` אוחדה:** `app.py` מאציל ל-`src/data.py` — אין יותר כפילות (החוב הטכני נסגר).
- Spatial Join: **מבנים buffer 25מ' מצנטרואידי `buildings_clean.csv`** (הפוליגונים המקוריים חסרים); **עצים buffer 10מ' בחיתוך פוליגונים אמיתי** מ-`national_canopy_clean.parquet`.
- גבהי מבנים חסרים מחושבים ע"י נוסחה (לא ML): `height = 2.95 * floors + 5.21`.
- פיצ'רים סטטיים מראש ב-`edges_features.parquet`; כיסוי-צל מראש ב-`shadow_coverage.parquet`; גרף רחובות ב-cache `tel_aviv_walk.graphml`.
- **ניווט מיושם:** מצב "מוצל" = Dijkstra עם משקל `TCI × אורך × העדפת-ירוק`; מצב "מהיר" = OSRM. בטאב הניווט יש **בורר שעה** (6:00–19:00): השמש אמיתית לתאריך של היום בשעה הנבחרת, מזג האוויר חי, ורק `shadow_cov` מיוחס ליום קיץ (snap למצב-השמש הקרוב). בשעות שקיעה/זריחה/זנית אין ניגודיות צל → המסלול המוצל מתכנס למהיר (תקין).
- **העדפת רחובות ירוקים בניווט** (`_preferred_edges` ב-`routing.py`): קשת ברחוב "ירוק" מקבלת מקדם `BOULEVARD_WEIGHT_FACTOR=0.5` במשקל. רחוב ירוק = ממוצע כיסוי-עצים > 35% (`CANOPY_STREET_THRESHOLD`), **או** שדרה עם כיסוי ≥ 24% (`BOULEVARD_CANOPY_FLOOR`, רף רוטשילד — תופס בולווארדים שעצי הטיילת שלהם תת-נספרים בקשתות הכביש: רוטשילד, ח"ן, בן גוריון). כיכר/שביל/סמטה מסוננים. ההחלטה **ברמת הרחוב (ממוצע), לא פר-קשת** — כדי שקשת בודדת עם canopy נמוך לא "תשבור" רחוב ירוק שלם (הראוטר מעדיף את הרחוב כיחידה). **ה-TCI לא מושפע** — זו העדפת ניווט בלבד, מעבר לנוחות הגולמית.
- חישובי צל = גיאומטריה 2D (לא 3D ray-casting מלא כמו Shadowmap).
- baseline = `DummyRegressor(mean)` (RMSE≈2.14) — הממוצע ממזער RMSE בהגדרה; median משני בגלל הטיה ימנית.
- **מגבלה מרכזית — יעד סינתטי:** ה-TCI הוא נוסחה שכתבנו, אז המודל "משחזר נוסחה" (R²≈0.996 מנופח). `shadow_cov` קירב למציאות אך לא שבר את המעגליות. השלב שישבור אותה = תוויות אמת מדודות (LST מלוויין דרך Earth Engine).
- `temperature`/`humidity` = decoys (לא בנוסחה) — נשמרו לבדיקת בחירת פיצ'רים.
