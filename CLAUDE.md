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
pysolar, scikit-learn, networkx, pytest, groq

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
- **ניווט:** `src/routing.py` — מצב "מוצל" = A* על גרף OSMnx עם משקל `TCI^shade_factor × אורך` (TCI מהמודל, `shadow_cov` לפי השמש הנוכחית); מצב "מהיר" = OSRM.
- **מגבלה מרכזית (לא לשכוח):** היעד סינתטי — המודל משחזר את הנוסחה (R² מנופח, מעגליות). `shadow_cov` קירב את הנוסחה למציאות אך לא שבר את המעגליות; תווית אמת (LST) היא הצעד הבא.
- **ספריה:** Scikit-Learn

---

## LLM Agent (M4)

- **קובץ:** `src/agent.py` — `extract_route_params(text, **weather_ctx)`
- **מודל:** Groq API — `llama-3.3-70b-versatile`
- **תפקיד:** חילוץ פרמטרים בלבד מטקסט חופשי עברית/אנגלית — הLLM לא מחשב, רק מחלץ
- **שדות פלט (7):** `origin`, `destination`, `hour` (6.0–19.0 decimal; null=אין העדפה), `date` (YYYY-MM-DD; null=היום), `mode` ("shaded"/"fast"), `shade_level` (null/"short"/"balanced"/"shaded"/"max"), `recommendation` (הסבר אוטומטי ≤100 תווים)
- **לוגיקת מזג-אוויר:** אם מועברים `sun_altitude`/`cloud_cover` — LLM ממליץ אוטומטית "fast"/"short" בלילה/עננות כבדה
- **Hebrew normalization:** קידומות דקדוקיות (מ/ל/ב/ה) מנוקות משמות מקומות לפני geocoding
- **Fallback:** כל כשלון (auth, network, JSON parse, שדות חסרים) → `{"error": "הודעה ידידותית בעברית"}`
- **תלות:** `groq` (optional — מחזיר שגיאה אם הספריה חסרה; lazy import)
- **בדיקות:** `tests/test_agent.py` — validation, coercion, fallback ללא קריאות רשת

---

## Architecture & Stack

```
User Text (Hebrew/English)
        ↓
src/agent.py (Groq LLM) → route params (origin, dest, hour, mode, shade_level)
        ↓
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
Edge Weights → ניווט: A*+TCI ("מוצל") / OSRM ("מהיר")
        ↓
Route → Folium Map (TCI color gradient) → Streamlit UI (7 tabs)
```

**Fallback:** אם Open-Meteo לא זמין — טעינת נתוני אקלים ממוצעים מ-`data/climate_fallback.json`.

---

## App Structure (7 Tabs)

| Tab | תפקיד |
|-----|--------|
| 🚶 ניווט (**ראשון**) | קלט LLM agent (טקסט חופשי) + Geocoding 3-שלבי (Nominatim→Overpass→Photon) + A*+TCI ("מוצל") / OSRM ("מהיר") + מפת מסלול עם gradient צבע |
| ℹ️ אודות | מידע כללי + תוצאות השוואת מודלים |
| 🎯 למידת הבעיה (M2 Component 1) | פרסונה (נועה, 28), מסע משתמש לפני/אחרי |
| 📚 סקירת ספרות (M2 Component 2) | 4 מאמרים מרכזיים (Boeing 2017, Park 2021, Lin 2012, Sulzer 2025) |
| 🏪 סקר שוק (M2 Component 3) | 5 מתחרים: Google Maps, Citymapper, Strava, Shadowmap, CoolWalks |
| 📊 ניתוח נתונים (M2 Component 4) | EDA מבנים, עצים, מזג אוויר, זוויות שמש |
| 🗺️ מפה | Folium אינטראקטיבי — מבנים + עצים + שכונת הצפון הישן |

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
│   ├── agent.py            # M4: extract_route_params() — Groq LLM, coercion, validation, Hebrew normalization
│   ├── data.py             # build_tci_df() — טבלת אימון TCI (יעד+פיצ'ר shadow_cov), ללא Streamlit
│   ├── clean_buildings.py  # ניקוי שכבת המבנים
│   ├── eda_buildings.py    # EDA וגרפים לשכבת המבנים
│   ├── spatial.py          # compute_edge_features() — מבנים buffer 25מ' (צנטרואידים), עצים buffer 10מ' (חיתוך פוליגונים)
│   ├── model.py            # M3 שלבים 3-8: train/val/test, baseline, 3 מודלים, בחירת מנצח, שמירה
│   ├── routing.py          # plan_route (DI orchestrator), geocode_address (3-stage), compute_tci_weights, compute_shaded_route (A*+TCI), build_route_map
│   └── weather.py          # get_current_weather() — Open-Meteo + fallback
├── notebooks/01_eda.ipynb  # EDA
├── outputs_M2/ · outputs_M3/
├── tests/
│   └── test_agent.py       # בדיקות validation, coercion ו-fallback ל-agent.py (ללא קריאות רשת)
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
- **ניווט מיושם:** מצב "מוצל" = **A\*** (Haversine heuristic ×0.5, admissible) עם משקל `TCI^shade_factor × אורך × העדפת-ירוק`; מצב "מהיר" = OSRM. בטאב הניווט יש **בורר שעה** (6:00–19:00): השמש אמיתית לתאריך של היום בשעה הנבחרת, מזג האוויר חי, ורק `shadow_cov` מיוחס ליום קיץ (snap למצב-השמש הקרוב). בשעות שקיעה/זריחה/זנית אין ניגודיות צל → המסלול המוצל מתכנס למהיר (תקין).
- **Geocoding מדורג ב-3 שלבים** (`geocode_address()` ב-`routing.py`): (1) Nominatim — 3 וריאנטי שאילתה; (2) Overpass API — שמות POI ללא כתובת; (3) Photon — fuzzy geocoder שסובל שגיאות כתיב וחיפוש שם חנות.
- **`shade_level` + `shade_factor` exponent:** AGENT מחלץ `shade_level` (null/"short"/"balanced"/"shaded"/"max") → ממופה ל-`shade_factor` מספרי → `TCI^shade_factor` מגביר את ניגודיות המשקלות (shade_factor=1.5: TCI 2→2.8, TCI 8→22.6).
- **Pedestrian bonus:** קשתות מסוג footway/pedestrian מקבלות מקדם 0.8× במשקל — עידוד מדרכות וכבישים לא-ראשיים.
- **Fallback modes ב-`plan_route()`:** `model_missing` (אין tci_model.joblib), `night` (sun_altitude≤0°), `overcast` (cloud_cover≥80%), `weights_missing` — בכולם חוזרים ל-OSRM אוטומטית עם הודעה.
- **`plan_route()` — dependency injection:** מקבל `geocode_fn`, `weather_fn`, `weights_fn` — ניתן לבדיקה ללא Streamlit (testable, stateless).
- **העדפת רחובות ירוקים בניווט** (`_preferred_edges` ב-`routing.py`): קשת ברחוב "ירוק" מקבלת מקדם `BOULEVARD_WEIGHT_FACTOR=0.5` במשקל. רחוב ירוק = ממוצע כיסוי-עצים > 35% (`CANOPY_STREET_THRESHOLD`), **או** שדרה עם כיסוי ≥ 24% (`BOULEVARD_CANOPY_FLOOR`, רף רוטשילד — תופס בולווארדים שעצי הטיילת שלהם תת-נספרים בקשתות הכביש: רוטשילד, ח"ן, בן גוריון). כיכר/שביל/סמטה מסוננים. ההחלטה **ברמת הרחוב (ממוצע), לא פר-קשת** — כדי שקשת בודדת עם canopy נמוך לא "תשבור" רחוב ירוק שלם (הראוטר מעדיף את הרחוב כיחידה). **ה-TCI לא מושפע** — זו העדפת ניווט בלבד, מעבר לנוחות הגולמית.
- חישובי צל = גיאומטריה 2D (לא 3D ray-casting מלא כמו Shadowmap).
- baseline = `DummyRegressor(mean)` (RMSE≈2.14) — הממוצע ממזער RMSE בהגדרה; median משני בגלל הטיה ימנית.
- **מגבלה מרכזית — יעד סינתטי:** ה-TCI הוא נוסחה שכתבנו, אז המודל "משחזר נוסחה" (R²≈0.996 מנופח). `shadow_cov` קירב למציאות אך לא שבר את המעגליות. השלב שישבור אותה = תוויות אמת מדודות (LST מלוויין דרך Earth Engine).
- `temperature`/`humidity` = decoys (לא בנוסחה) — נשמרו לבדיקת בחירת פיצ'רים.
