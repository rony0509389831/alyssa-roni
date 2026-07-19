# CLAUDE.md

## Project Purpose
SHADY — מערכת ניווט עירוני להולכי רגל המוצאת מסלולים מוצלים על בסיס גבהי מבנים, חופת עצים, מיקום השמש ומזג האוויר בזמן אמת.

---

## Data Sources

| מקור | פורמט | תוכן עיקרי |
|------|--------|------------|
| עיריית ת"א (opendata.tel-aviv.gov.il) | CSV | **צנטרואידים** של ~45,783 מבנים (lat/lon + גובה + קומות) — `data/buildings_clean.csv`. ⚠️ קובץ הפוליגונים המקורי (`tel_aviv_buildings.geojson`) **אינו בפרויקט** — יש רק צנטרואידים |
| מפ"י (data.gov.il) | Parquet | ~231,000 פוליגוני חופת עצים באזור ת"א — `data/national_canopy_clean.parquet` |
| OSMnx | GraphML | רשת רחובות ת"א — `data/tel_aviv_walk.graphml`, **מועלה ל-repo** (נדרש ל-deploy, ר' Key Decisions) |
| Open-Meteo API | JSON (שעתי) | `temperature_2m`, `relative_humidity_2m`, `cloud_cover` |
| Precomputed (spatial.py) | Parquet | פיצ'רים סטטיים לכל קשת — `data/edges_features.parquet` (`mean_building_height`, `tree_canopy_ratio`, `street_azimuth`) |
| Precomputed (precompute_shadow.py) | Parquet | כיסוי-צל **מאוחד** (מבנים+עצים) פר קשת × שעה (6:00–19:00, יום קיץ) — `data/shadow_coverage.parquet` |

שימוש גם בספריות פייתון הבאות עבור ממשק משתמש, זווית שמש וכו':
pandas, numpy, streamlit, requests, matplotlib,
geopandas, osmnx, folium, streamlit-folium, pyarrow, pyogrio,
pysolar, scikit-learn, networkx, pytest, groq, shapely, scipy

**חוסרים ידועים:**
- ~4% מגבהי המבנים הם NaN — מטופל ע"י נוסחת imputation: `height = 2.95 * floors + 5.21` (R²≈0.49).
- `national_canopy_clean.parquet` אין בו שדה גובה-עץ — משתמשים בטבלת buckets לפי `area_class` (3.0-14.0מ', ר' `precompute_shadow.py::_TREE_HEIGHT_BY_CLASS`). **הנחת-מודל לא-נמדדת**, מאותה משפחה כמו imputation גובה המבנים — לא מוסתרת כעובדה.

---

## ML Model

- **משימה:** רגרסיה — חיזוי Thermal Comfort Index (ציון 1–10) לכל קשת בגרף הרחובות.
- **פיצ'רים (7, מקור אמת = `FEATURE_COLS` ב-`src/data.py`):** `sun_altitude`, `building_height`, `canopy_ratio`, `cloud_cover`, `temperature`, `humidity`, `shadow_cov`.
  - **`shadow_cov` ∈ [0,1] = כיסוי-צל מאוחד (מבנים+עצים), מחושב מראש** ב-`precompute_shadow.py` (פר קשת × שעה) ונטען מ-`data/shadow_coverage.parquet`. הוא **החליף את `shadow_angle` הישן** (ר' Key Decisions), ומאז 2026-07-10 **מאחד גם צל-מבנים וגם צל-עצים** לאות אחד (ר' "שיפור אלגוריתם הצל" למטה).
  - בקובץ `edges_features.parquet` השמות הסטטיים הם `mean_building_height`/`tree_canopy_ratio`/`street_azimuth`. **`building_height` ו-`canopy_ratio` הם כעת decoys** (חשיבות 0.000 בשניהם) — `shadow_cov` המאוחד בלע את שני התפקידים הפיזיים שלהם.
  - `temperature`/`humidity`/`canopy_ratio` = decoys (לא בנוסחת ה-TCI) — המודל אמור להוריד להם משקל.
- **חישוב Sun Position:** PySolar.
- **נוסחת ה-TCI האנליטית (מקור התוויות, עודכנה 2026-07-10):**
  ```
  TCI = clip( 1 + 9 * (sun_altitude/80) * (1 - cloud_cover/100) * (1 - shadow_cov), 1, 10 )
  ```
  `canopy_ratio` **הוסר מהנוסחה** (היה מודד את אותה הצללה בצורה סטטית/כפולה) — נשאר ב-`FEATURE_COLS` כ-decoy מכוון בלבד. `shadow_cov` = אחוז הקשת המכוסה בצל **מבנים ועצים כאחד**: כל מבנה = ריבוע שגודלו נגזר מהצפיפות המקומית וכיוונו מיושר לרחוב הקרוב (לא ריבוע קבוע 20מ' יותר), כל עץ = מצולע חופה אמיתי עם גובה משוער לפי `area_class`; שניהם נגררים `h/tan(alt)` בכיוון ההפוך לשמש ומאוחדים ל-STRtree אחד לפני חיתוך עם הקשת. החליף את `building_height × cos(alt) × sin(shadow_angle)` הישן שתפס רק זווית (וזקף הצללת-שווא ברחובות מזרח-מערב).
- **אות הצל אחיד בכל המערכת:** אותו `shadow_cov` (המאוחד) מזין את טבלת האימון, המודל, גרף 4 (אנליטי + ML), והניווט.
- **`src/data.py`** — `build_tci_df()`: דוגם קשת × שעת-ייחוס, מחשב יעד עם `shadow_cov`. `app.py` **מאציל** לכאן (אין יותר כפילות).
- **`src/model.py`** — M3 שלבים 3–8. מנצח: **RandomForest (RMSE≈0.090, R²≈0.9985 על test)** → `data/tci_model.joblib`. הרצה: `python -m src.model`. חשיבות פיצ'רים לאחר האיחוד: `shadow_cov`=0.764, `sun_altitude`=0.168, `cloud_cover`=0.035, `temperature`=0.031 (decoy), `humidity`=0.003 (decoy), `canopy_ratio`=0.000 (decoy), `building_height`=0.000.
- **Loss:** MSE | **KPI:** RMSE (ראשי) + R² (בונוס).
- **חלוקה:** 70/15/15 (n=5000 → 3500/750/750), `random_state=42`, פיצול לפי שורה. בחירת מנצח על **val**, דיווח על **test**.
- **Baseline:** `DummyRegressor(mean)` (RMSE≈2.35 על test) + median (משני; TCI מוטה ימינה). Linear/Tree/Forest = מודלים מועמדים.
- **ניווט:** `src/routing.py` — מצב "מוצל" = A* על גרף OSMnx עם משקל אדיטיבי `(TCI-1)×אורך×factor + λ×אורך` (TCI מהמודל, `shadow_cov` לפי השמש הנוכחית; λ נבחר בחיפוש בינארי לפי תקציב-עיקוף הרמה — ר' Key Decisions); מצב "מהיר" = OSRM.
- **מגבלה מרכזית (לא לשכוח):** היעד סינתטי — המודל משחזר את הנוסחה (R² מנופח, מעגליות; R² עוד עלה אחרי איחוד-הצל, כי הפישוט לגורם יחיד הקל עוד יותר על המודל "לשחזר נוסחה"). `shadow_cov` המאוחד קירב את הנוסחה למציאות אך לא שבר את המעגליות; תווית אמת (LST) היא הצעד הבא.
- **ספריה:** Scikit-Learn

---

## LLM Agent (M4)

- **קובץ:** `src/agent.py` — `extract_route_params(text, **weather_ctx)`
- **מודל:** Groq API — `llama-3.3-70b-versatile`
- **תפקיד:** חילוץ פרמטרים בלבד מטקסט חופשי עברית/אנגלית — הLLM לא מחשב, רק מחלץ
- **שדות פלט (7):** `origin`, `destination`, `hour` (0.0–23.99 decimal; null=לא צוינה שעה כלל), `date` (YYYY-MM-DD; null=היום), `mode` ("shaded"/"fast"), `shade_level` (null/"short"/"max" — **רק 2 רמות מ-2026-07-18**, "balanced" הוסרה: לא הצדיקה בפועל רמה נפרדת, ר' "מודל-משקל אדיטיבי" למטה), `recommendation` (הסבר אוטומטי ≤100 תווים)
- **`hour` בשעות ערב/לילה (מחוץ ל-6-19, טווח הניווט המוצל):** נשמר כערך אמיתי (למשל 22.0 ל"10 בלילה") ולא מאופס ל-null — `mode='fast'`/`shade_level='short'`/`recommendation='night'` מסמנים את מצב-הלילה בנפרד. תיקון מ-2026-07-10: קודם השעה אופסה ל-null בכל שעת חושך, מה שגרם ל-`app.py` לאבד את השעה שהמשתמש ביקש ולהציג שעה נוכחית אקראית (הבאג שנצפה כ"18:30 מסתורי"). `routing.sun_position`/`plan_route` תומכים בשעון מלא ונופלים אוטומטית ל"מהיר" כששעת השמש שלילית — אין מגבלה טכנית על 6-19 מעבר לחישוב הצל.
- **לוגיקת מזג-אוויר:** אם מועברים `sun_altitude`/`cloud_cover` — LLM ממליץ אוטומטית "fast"/"short" בלילה/עננות כבדה (כשהמשתמש לא ציין העדפה)
- **כלל לילה דטרמיניסטי (2026-07-11):** ב-`_validate` — אם השעה מחוץ ל-[6,19] (או שמש מתחת לאופק) → כופה `mode=fast`/`shade_level=short`/`recommendation=night`, **גובר על כל בקשת צל מפורשת** בטקסט. הוראת פרומפט לבדה הסתברותית; הכפייה בקוד מבטיחה 100%.
- **זיהוי תאריכים עבריים (2026-07-11):** `_weekday_reference(today)` מזריק לפרומפט טבלת 15 ימים (ISO + שם-יום עברי + סימון "השבוע"/"שבוע הבא", שבוע ישראלי מתחיל ראשון). ה-LLM **בוחר** מהטבלה במקום לחשב תאריך — פותר "ביום שני בשבוע הבא" (LLMs לא אמינים בחשבון תאריכים).
- **Hebrew normalization:** קידומות דקדוקיות (מ/ל/ב/ה) מנוקות משמות מקומות לפני geocoding
- **Fallback:** כל כשלון (auth, network, JSON parse, שדות חסרים) → `{"error": "הודעה ידידותית בעברית"}`
- **תלות:** `groq` (optional — מחזיר שגיאה אם הספריה חסרה; lazy import)
- **תובנת מסלול — tool use אמיתי (M4, 2026-07-11):** `recommend_route_insight(user_text, api_key, metrics_fn)` — רושם כלי `evaluate_route`; **Turn 1** ה-LLM קורא לכלי → הקוד מריץ `metrics_fn()` (= `compute_length_route` + `compute_route_insights` ב-routing.py — "המודל רץ") → **Turn 2** ה-LLM מנסח משפט עברית מעוגן ("האריך את ההליכה ב-X דק', הוריד TCI ב-Y, חסך Z דק' חשיפה גבוהה"). **המספרים תמיד מהקוד, ה-LLM רק מנסח.** Fallback: `format_insight_fallback(insights)` (משפט מחושב) כשאין מפתח/Groq — עובד מקומית בלי מפתח, לעולם לא שובר. הכלי **מריץ בפועל** ולא מהדהד מספרים מוכנים (קריטי לעמידה ב-M4).
- **אופק תאריכים (2026-07-13):** `WEATHER_HORIZON_DAYS=7` + `_date_out_of_range`. ב-`_validate` תאריך מחוץ ל-`[today, today+7]` (כולל עבר) → `{"error": _DATE_RANGE_ERR}` ("מזג האוויר זמין רק ל-7 הימים הקרובים"). לפני כן `app.py` נפל בשקט להיום; עכשיו שגיאה מפורשת (app.py:686 מציג `{error}` כ-`st.warning`).
- **אדישות-לצל = מהיר (2026-07-13):** "לא אכפת לי מהצל"/"לא משנה לי צל" → `mode=fast`/`shade_level=short` (הוגדר מפורשות בפרומפט + דוגמה). **החלטה מוצרית** — שני המודלים (הקטן וגם llama-3.3-70b) פירשו את זה כברירת-מחדל shaded, ונקבע שאדישות משמעה מהירות.
- **בדיקות (2026-07-13, ממופה לשקופיות "green != real"):** דטרמיניסטי בלי מפתח = `tests/test_agent.py` + `tests/test_agent_scenarios.py` (ולידציה/כלל-לילה/תאריך-רחוק/קלט-קצה) + `tests/test_insights.py` (תובנות + תיקון-רחוב) + `tests/test_model_contract.py` (**מונוטוניות + אדישות-decoy** — מודל "תמיד-5" נכשל) + `tests/test_route_flow.py` (גרף סינתטי: הצל משנה מסלול, משקלים נכנסים ל-A*, אינvariantים) + `tests/test_smoke.py` (predict אמיתי). חי-ומגודר-מפתח = `tests/test_agent_llm.py` (`skipif(not GROQ_API_KEY)`, מודל קטן דרך `SHADY_TEST_MODEL`; rate-limit→skip, לא fail). **62 דטרמיניסטיים + 14 חיים (עוברים כשיש GROQ_API_KEY, אחרת skip) — עודכן 2026-07-18 ערב מאוחר (כולל `tests/test_geocode_house_number.py`).**

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
Route → Folium Map (TCI color gradient) → Streamlit UI (עמוד ניווט יחיד + נספח מקופל)
```

**Fallback:** אם Open-Meteo לא זמין — טעינת נתוני אקלים ממוצעים מ-`data/climate_fallback.json`.

---

## App Structure (עמוד יחיד + נספח מקופל — מ-2026-07-08)

**החלטת UX:** המרצה קבע שדרישות התיעוד של הקורס (M2) צריכות להיות שוליות ולא חלק מהחוויה המרכזית; המשתמשת ביקשה עיצוב ממוקד-משתמש לקראת הצגה תחרותית. לכן `app.py` הוא כעת **עמוד יחיד** — חוויית הניווט רצה מלמעלה עד סוף בלי שורת טאבים, וכל תוכן דרישות הקורס עבר ל-`st.expander` מקופל אחד בתחתית העמוד (לא נמחק — עדיין מלא ונגיש להערכה). טאב "🗺️ הדגמת נתונים" (מבנים+עצים+שכונת הצפון הישן, נפרד ממפת תוצאת המסלול) **נמחק לחלוטין** — היה לביקורת עצמית, לא רלוונטי יותר.

| אזור | תפקיד |
|-----|--------|
| 🚶 ניווט (**עמוד ראשי, לא טאב**) | קלט LLM agent (טקסט חופשי) + Geocoding 3-שלבי (Nominatim→Overpass→Photon) + A*+TCI ("מוצל") / OSRM ("מהיר") + מפת מסלול עם gradient צבע + `st.status()` עם הודעות התקדמות אמיתיות בזמן חישוב |
| 📎 חומרי רקע ותיעוד (**expander מקופל, בתחתית**) | מכיל 5 תת-expanders: ℹ️ אודות, 🎯 למידת הבעיה (M2 Component 1), 📚 סקירת ספרות (M2 Component 2), 🏪 סקר שוק (M2 Component 3), 📊 ניתוח נתונים (M2 Component 4) — תוכן מקורי בלי שינוי, רק הוזז |

---

## File Structure (בפועל)

```
alyssa-roni/
├── app.py                      # Streamlit entry point (עמוד יחיד: ניווט + expander מקופל לחומרי הקורס)
├── .streamlit/config.toml      # ערכת עיצוב (כחול-ירוק=צל, כתום-צהוב=שמש, מיושר מול _tci_color())
├── .gitattributes              # מסמן data/tel_aviv_walk.graphml כבינארי (-text) — בלי המרת EOL בין Win/Linux
├── precompute_features.py      # הרצה חד-פעמית: יוצר edges_features.parquet
├── precompute_shadow.py        # הרצה חד-פעמית: יוצר shadow_coverage.parquet (כיסוי-צל מאוחד פר קשת×שעה)
│   # הערה: tel_aviv_buildings.geojson, download_buildings.py, check_data.py,
│   # src/clean_buildings.py, src/eda_buildings.py, outputs_M2/ — כולם ב-.gitignore, אינם בעץ העבודה בפועל
├── data/
│   ├── buildings_clean.csv         # 45,783 צנטרואידי מבנים (lat/lon+גובה) — מקור הצל
│   ├── national_canopy_clean.parquet # ~231k פוליגוני עצים
│   ├── edges_features.parquet      # פיצ'רים סטטיים לקשתות (mean_building_height, tree_canopy_ratio, street_azimuth)
│   ├── shadow_coverage.parquet     # כיסוי-צל: u,v,key + עמודה לכל שעה (59,086 × 27)
│   ├── tci_model.joblib            # מודל TCI מאומן (RandomForest, פיצ'ר shadow_cov)
│   ├── model_results.json          # תוצאות השוואת המודלים — מוצג בטאב אודות
│   ├── tel_aviv_walk.graphml       # רשת רחובות מ-OSMnx — **מועלה ל-repo** (נדרש ל-deploy; בלעדיו הענן מוריד גרף לא-תואם)
│   ├── climate_fallback.json       # ממוצעים חודשיים (T, humidity, cloud_cover)
│   └── screenshots/                # PNG לממשק
├── src/
│   ├── agent.py            # M4: extract_route_params() (חילוץ+כלל-לילה+טבלת-תאריכים) · recommend_route_insight() (tool use) · format_insight_fallback()
│   ├── data.py             # build_tci_df() — טבלת אימון TCI (יעד+פיצ'ר shadow_cov מאוחד), ללא Streamlit
│   ├── spatial.py          # compute_edge_features() — מבנים buffer 25מ' (צנטרואידים), עצים buffer 10מ' (חיתוך פוליגונים); _load_trees() גם לשימוש precompute_shadow.py
│   ├── model.py            # M3 שלבים 3-8: train/val/test, baseline, 3 מודלים, בחירת מנצח, שמירה
│   ├── routing.py          # plan_route (DI orchestrator + תקציב-עיקוף אדיטיבי), geocode_address (3-stage), compute_tci_weights, compute_shaded_route (A*+TCI), compute_length_route/compute_route_insights (תובנות), shortest_walk_distance, build_route_map
│   └── weather.py          # get_current_weather() — Open-Meteo + fallback
├── notebooks/01_eda.ipynb  # EDA
├── outputs_M3/             # דוח model_checks.html (מקור; docs/ הוא ההעתק המפורסם ל-GitHub Pages)
├── tests/                  # 62 דטרמיניסטיים (בלי מפתח) + 14 חיים (test_agent_llm, מגודר-מפתח)
│   ├── test_agent.py       # validation, coercion, night-override, weekday-table, insight fallback (ללא רשת)
│   ├── test_agent_scenarios.py # Unit דטרמיניסטי: דוחה-זבל, תאריך-רחוק, גבולות כלל-לילה, קלט-קצה
│   ├── test_model_contract.py  # Integration: מונוטוניות TCI + אדישות-decoy (דורש model+parquet)
│   ├── test_route_flow.py  # Integration: גרף סינתטי — הצל משנה מסלול, משקלים→A*, אינvariantים
│   ├── test_agent_llm.py   # Integration חי: הבנת LLM (skipif GROQ_API_KEY; מודל קטן SHADY_TEST_MODEL)
│   ├── test_insights.py    # compute_route_insights + multigraph ל-compute_length_route + תיקון-רחוב
│   ├── test_geocode_house_number.py  # ולידציית house_number (Nominatim+Photon) + תיקון lang שבור ב-Photon
│   └── test_smoke.py       # import src + טעינת מודל + predict אמיתי (isfinite, סדר-עמודות)
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

- **הצללת מבנים = `shadow_cov`** (לא `shadow_angle` יותר). מצולע צל אמיתי (נגרר `h/tan(alt)` נגד השמש) → אחוז הקשת בצל. מחושב מראש ב-`precompute_shadow.py`. מתקן הצללת-שווא ברחובות מזרח-מערב (שינקין) שהנוסחה הישנה (`sin(shadow_angle)`) יצרה. **אומת ידנית מול Shadowmap** (כלי אימות, לא מקור נתונים).
- **`shadow_cov` הוא אות הצל היחיד** — אותו חישוב באימון, במודל, בגרף 4, ובניווט (אחידות מלאה לאחר הגירה).
- **שיפור אלגוריתם הצל (2026-07-10, בלי הורדת נתונים חדשים):**
  - **footprint מבנים דינמי** (החליף ריבוע קבוע 20×20מ'): **גודל** נגזר מצפיפות מקומית (ממוצע מרחק ל-5 שכנים קרובים, `clip[5,20]`מ'); **כיוון** מיושר לרחוב הקרוב ביותר (`street_azimuth`) במקום ריבוע ישר-צירים.
  - **צל-עצים אוחד ל-`shadow_cov`**: פוליגוני חופה אמיתיים (לא ריבוע!) נגררים לפי גובה-עץ משוער (buckets לפי `area_class`, 3-14מ' — הנחת-מודל, אין שדה גובה אמיתי בנתונים), מסוננים אדפטיבית-לשעה (רק עצים שיכולים להגיע לרחוב בזווית-השמש הנתונה — חוסך זמן ריצה בלי לפגוע בדיוק), ומאוחדים ל-STRtree אחד ביחד עם צללי המבנים לפני חיתוך עם הקשת (`unary_union` פותר double-counting חפיפות אוטומטית). **מחליף** את `canopy_ratio` הסטטי (שלא היה תלוי-שעה בכלל) כאות הצל-מעצים.
  - תוצאה: `canopy_ratio`/`building_height` ירדו לחשיבות **0.000** (decoys אמיתיים כעת), `shadow_cov` עלה ל-**0.764** (מ-0.072).
  - אומת: שדרות עם canopy מתועד (רוטשילד/ח"ן/בן גוריון) עברו מ-median shadow_cov=0.000 בצהריים ל-0.20-0.30 — הצללת-הצהריים מהעצים סוף-סוף נתפסת נכון.
- **ייחוס יום-קיץ (27 נקודות זמן, כל חצי שעה 6:00–19:00):** קיץ = השמש הכי גבוהה → צללים קצרים → הכי פחות צל → המבחן המחמיר ביותר לראוטר. **פערים ידועים:** (א) עונתיות השמש (הניווט נצמד למצב-השמש הקרוב) — *הבניינים/עצים הם שכבה קבועה, הפער בשמש לא בהם*; (ב) footprint מבנים משוער (צפיפות+כיוון, לא קו-מתאר אמיתי) וגובה-עץ משוער (buckets לפי `area_class`, לא מדוד).
- **`build_tci_df` אוחדה:** `app.py` מאציל ל-`src/data.py` — אין יותר כפילות (החוב הטכני נסגר).
- Spatial Join: **מבנים buffer 25מ' מצנטרואידי `buildings_clean.csv`** (הפוליגונים המקוריים חסרים); **עצים buffer 10מ' בחיתוך פוליגונים אמיתי** מ-`national_canopy_clean.parquet`.
- גבהי מבנים חסרים מחושבים ע"י נוסחה (לא ML): `height = 2.95 * floors + 5.21`.
- פיצ'רים סטטיים מראש ב-`edges_features.parquet`; כיסוי-צל מראש ב-`shadow_coverage.parquet`; גרף רחובות ב-cache `tel_aviv_walk.graphml`.
- **ניווט מיושם:** מצב "מוצל" = **A\*** (Haversine heuristic ×0.5, admissible) עם משקל אדיטיבי `(TCI-1)×אורך×factor + λ×אורך` (`factor` = העדפת-ירוק/מדרכה); מצב "מהיר" = OSRM. בטאב הניווט יש **בורר שעה** (6:00–19:00): השמש אמיתית לתאריך של היום בשעה הנבחרת, מזג האוויר חי, ורק `shadow_cov` מיוחס ליום קיץ (snap למצב-השמש הקרוב). בשעות שקיעה/זריחה/זנית אין ניגודיות צל → המסלול המוצל מתכנס למהיר (תקין).
- **מודל-משקל אדיטיבי + תקציב-עיקוף (`_TIER_TARGET_RATIOS`, שכתוב מלא 2026-07-19, קומיט "bugfix"):** מחליף **לחלוטין** את מנגנון ה-`TCI^shade_factor` + `_TIER_DETOUR_CAPS` (תקרת-אחוז/תקרת-דקות) + מדרג-נפילה-בשלושה-שלבים (`tier_escalated`/`best_effort_over_budget`/`detour_cap_unreachable`) שתואר כאן עד 2026-07-18 — **אינם קיימים יותר בקוד**. לכל קשת: `weight = (TCI-1)×אורך×factor + λ×אורך` — איבר-שמש (עלות-חום מצטבר על הקשת, 0 בצל מלא TCI=1, מוכפל ב-`factor` כדי לשמר את העדפת-הרחוב-הירוק) ועוד איבר-אורך (`λ×אורך` = "מחיר-מרחק"; λ גבוה→אורך שולט→מסלול קצר/חם, λ נמוך→איבר-השמש שולט→מסלול ארוך/מוצל).
  **`shade_factor` (1.0/2.0) כבר אינו אקספוננט** — הוא בורר לאיזה **תקציב-עיקוף** (`_TIER_TARGET_RATIOS`) לכוון: "מעט צל"=**1.30×**, "הרבה צל"=**1.60×** מהמסלול הישיר (`shortest_walk_distance`). `plan_route` מוצא ב**חיפוש בינארי** (24 איטרציות, λ∈[0.05,50]) את ה-λ **הקטן ביותר** שעדיין נותן מסלול ≤ תקציב הרמה — פיזבילי כי `distance(λ)` יורד **מונוטונית** ב-λ (ר' "למה השתנה" למטה). **"שאיפה, לא תקרה":** אם המסלול הכי-מוצל-אפשרי (λ=0.05) כבר קצר מהתקציב — לוקחים אותו כמו-שהוא, לא "מרפדים" אורך כדי לנצל את מלוא התקציב.
  **רצפה תחתונה — רק ל"מעט צל" (`_TIER_MIN_RATIOS=1.10`):** אם גם המסלול הכי-מוצל-אפשרי נותן פחות מ-1.10× (העיקוף לא "קונה" צל משמעותי) — נופלים למסלול הקצר-ביותר-על-הגרף עם `fallback="little_shade_below_band"` (ר' Fallback modes למטה). ל"הרבה צל" **אין** רצפה כזו — כל צל שנמצא עד 1.60× מוצג.
  **למה השתנה מהמודל הישן:** `TCI^shade_factor` היה **לא-מונוטוני** — חזקה גבוהה יותר (אמורה לדרוש יותר צל) יכלה במקרים מסוימים לצאת קצרה/חמה יותר מחזקה נמוכה יותר, מה שגרם לבאג "מעט צל יצא ארוך/חם יותר מהרבה צל". המודל האדיטיבי מבטיח מונוטוניות ב-λ, ולכן ההיררכיה בין הרמות לעולם לא מתהפכת, וחיפוש-בינארי אמין.
- **avg_tci משוקלל-אורך, לא לפי מספר מקטעים (תוקן 2026-07-18, ערב):** `_summarize_path`/`_snap_tci_to_latlon_path` חישבו בעבר ממוצע TCI פשוט (`np.mean`) על פני מקטעי המסלול — מקטע של 10מ' השפיע על הממוצע בדיוק כמו מקטע של 300מ', בלי קשר לכמה מהמסלול בפועל עובר בו. תוקן ל-ממוצע משוקלל-מטרים (`_weighted_avg_tci`) — מדד אמין יותר ל"כמה חם היה ההליכה בפועל". נמדד פער אמיתי (לא תיאורטי) של עד 0.78 נקודות TCI (מתוך סקאלת 1-10) בין הישן לחדש על מסלולים אמיתיים, בלי כיוון עקבי (לפעמים הישן הציג מסלול כטוב יותר משהוא באמת, לפעמים כגרוע יותר). `high_exposure_m` כבר היה משוקלל-אורך גם לפני התיקון ולא השתנה — לכן "TCI ממוצע" ו"דקות חשיפה גבוהה שנחסכו" יכולים עדיין להיפרד לפעמים גם אחרי התיקון (סטטיסטיקות שונות במהותן: ממוצע מול משך-חציית-סף), וזה תקין ולא באג.
- **תובנת מסלול = tool use (M4, 2026-07-11):** מסלול-הבסיס להשוואה = `compute_length_route` (הקצר-ביותר על הגרף, weight="length" — **חובה מחרוזת, לא lambda**: ב-MultiDiGraph ה-lambda מקבל `{key:data}` ומזער צמתים במקום מרחק). `HIGH_EXPOSURE_TCI=5.5` (ה-TCI מגיע ל-~6.4 לכל היותר אחה"צ, סף 7 היה מאפס את מדד החשיפה ברוב היום). ר' פירוט בסעיף LLM Agent.
- **פריסה (Streamlit Cloud, 2026-07-11):** `data/tel_aviv_walk.graphml` **מועלה ל-repo** (לא רק cache!) עם `.gitattributes` שמסמן אותו בינארי (`-text`, בלי המרת EOL). בלעדיו הענן מוריד גרף OSM טרי שלא תואם ל-`edges_features.parquet` → מקטעי "TCI לא זמין". `osmnx` מפונטן `>=1.9,<2.0` — osmnx 2.x שינה API ותלויות, הקוד כתוב ל-1.x (מקומית 1.9.3). האפליקציה של rony פרוסה מ-**fork** → כל פרסום דורש push לריפו הקורס + **Sync fork** ב-GitHub.
- **Geocoding מדורג** (`geocode_address()` ב-`routing.py`): (0) **תיקון שגיאת-כתיב בשם רחוב** (2026-07-12) — כשיש מספר בית, שם הרחוב מתוקן מול מילון שמות ה-OSM של הגרף (`_street_names()`, כולל וריאנטים בלי קידומת "שדרות"/"רחוב") דרך `difflib` (סף 0.8), כדי ש-"דיזינגוף 55" (י' מיותרת) לא יחזיר כתובת שגויה מ-Nominatim. פועל רק כשיש מספר בית → שמות POI (בלי מספר) לא "מתוקנים" בטעות לרחוב דומה; (1) Nominatim — 3 וריאנטי שאילתה; (2) Overpass API — שמות POI ללא כתובת; (3) Photon — fuzzy geocoder שסובל שגיאות כתיב וחיפוש שם חנות.
  **ולידציית מספר-בית + תיקון Photon (2026-07-18, ערב מאוחר):** שני באגים אמיתיים, לא תיאורטיים — אומתו חי. (א) Nominatim/Photon נופלים בשקט לרמת-רחוב (`addresstype="road"`, בלי `house_number` בתשובה) כשמספר הבית לא סביר/לא קיים (למשל "יפת 4900" — הרחוב יפת קיים, 4900 לא) — הקוד קיבל את זה כאילו הייתה זו הכתובת המדויקת. תוקן: `_nominatim_search`/`_photon_search` מקבלים `house_number` (נחלץ פעם אחת מהכתובת המקורית ב-`geocode_address`) ודוחים כל תוצאה שמספר-הבית שלה לא תואם בדיוק — כתובת לא-מספרית (שם רחוב/מקום) לא מושפעת. (ב) `_photon_search` שלח `lang="he"` — Photon תומך רק ב-default/de/en/fr, אז **כל** קריאה אליו נדחתה בשקט ע"י ה-API ונבלעה ב-`try/except` הקיים; כל שלב-הגיבוי השלישי מעולם לא עבד בפועל, לא רק במקרה הזה. תוקן ע"י הסרת הפרמטר (התוכן חוזר נכון גם בלעדיו).
- **`shade_level` → `shade_factor` (2 רמות, לא 3, מ-2026-07-18):** AGENT מחלץ `shade_level` (null/"short"/"max") → ממופה ל-`shade_factor` מספרי (1.0/**2.0** בהתאמה). **מ-2026-07-19, `shade_factor` הוא בורר-רמה בלבד** — לא אקספוננט על ה-TCI יותר (ר' "מודל-משקל אדיטיבי" למעלה): הוא בוחר לאיזה תקציב-עיקוף (`_TIER_TARGET_RATIOS`, 1.30×/1.60×) לכוון.
  **היסטוריה (עד 2026-07-18 — מסבירה למה יש בדיוק 2 רמות עם הערכים 1.0/2.0, גם שהמנגנון הטכני מתחתיהן השתנה לגמרי; כבר לא מדויקת טכנית):** בגרסה הישנה `shade_factor` היה ממש אקספוננט (`TCI^shade_factor`, shade_factor=2.0: TCI 2→4, TCI 8→64). "הרבה צל" הורד מ-exponent 3.0 ל-2.0 (2026-07-18, ב-fork של rony): אומת חי שב-3.0 ה-A* מייצר מסלולים משוננים עם backtrack (עד ~81מ' אחורה מהיעד, "העיקול המוזר ביהודה הלוי") עבור שיפור-TCI זניח (~0.3 נק'); ב-exponent ≤2.5 המסלול חלק (0 backtrack) עם כמעט אותו צל. 2.0 = הערך הבטוח מתחת ל"מדרון" 2.5→3.0 (`_TIER_DETOUR_CAPS` ו-`WIDEST_TIER` דאז — שניהם לא קיימים יותר בקוד). **"מאוזן" (2.0) הוסרה** — בבדיקה אמפירית על מסלולים אמיתיים היא כמעט תמיד התכנסה לאותו מסלול בדיוק כמו "הרבה צל" (או לפעמים כמו "מעט צל"), בלי הבדל תפעולי אמיתי; מגבלה מבנית של המרחק בין עוצמות-צל סמוכות ברוב הטיולים, לא באג. ברירת המחדל (גם ב-UI וגם ב-agent כשלא צוינה העדפה) היא "מעט צל"; כל בקשת-צל **לא-ממותנת** מפורשת או כללית ("מוצל"/"ירוק"/"הכי מוצל") ממופה ל-"הרבה צל". כלל-האזהרה האוטומטי על "חם באמצע" (warm→balanced) הוסר גם הוא — נשאר רק כלל ה"חם מאוד" (hot→max). **בקשת-צל ממותנת → "מעט צל" (2026-07-18, ב-fork של rony):** "קצת צל"/"קצת מוצל"/"a little shade" ממופה ל-`short` ולא ל-`max` — במערכת 2-הרמות "קצת" קרוב ל"מעט" יותר מ"מקסימום". מיושם דו-שכבתי כמו כלל-הלילה: הוראת פרומפט + override דטרמיניסטי ב-`_validate` (`_SHADE_DIMINISHERS`, גובר על max רק כשזוהתה בקשת-צל). **סוטה מהחלטת-הבסיס "כל צל → max" — קיים כרגע רק ב-fork.**
- **Pedestrian bonus:** קשתות מסוג footway/pedestrian מקבלות מקדם 0.8× במשקל — עידוד מדרכות וכבישים לא-ראשיים.
- **Fallback modes ב-`plan_route()`:** `model_missing` (אין tci_model.joblib), `night` (sun_altitude≤0°), `overcast` (cloud_cover≥80%), `weights_missing` — בכולם חוזרים ל-**OSRM** (`compute_route`) אוטומטית עם הודעה. `little_shade_below_band` (רק "מעט צל" — גם המסלול הכי-מוצל-אפשרי מתחת ל-1.10× מהישיר, ר' "מודל-משקל אדיטיבי") שונה: נופל ל**מסלול-הקצר-ביותר-על-הגרף** (`compute_length_route`, לא OSRM — כדי לשמר TCI מוצמד לתצוגה/השוואה).
- **`plan_route()` — dependency injection:** מקבל `geocode_fn`, `weather_fn`, `weights_fn` — ניתן לבדיקה ללא Streamlit (testable, stateless). פרמטר נוסף אופציונלי `on_progress(msg: str)` (ברירת מחדל `None`, לא שובר תאימות) נקרא בכל שלב אמיתי (כולל שלבי fallback הפנימיים של `geocode_address()`: Nominatim→Overpass→Photon) — `app.py` מחבר אותו ל-`st.status()` כדי להציג הודעות התקדמות אמיתיות במקום ספינר סתום.
- **העדפת רחובות ירוקים בניווט** (`_preferred_edges` ב-`routing.py`): קשת ברחוב "ירוק" מקבלת מקדם `BOULEVARD_WEIGHT_FACTOR=0.5` במשקל. רחוב ירוק = ממוצע כיסוי-עצים > 35% (`CANOPY_STREET_THRESHOLD`), **או** שדרה עם כיסוי ≥ 24% (`BOULEVARD_CANOPY_FLOOR`, רף רוטשילד — תופס בולווארדים שעצי הטיילת שלהם תת-נספרים בקשתות הכביש: רוטשילד, ח"ן, בן גוריון). כיכר/שביל/סמטה מסוננים. ההחלטה **ברמת הרחוב (ממוצע), לא פר-קשת** — כדי שקשת בודדת עם canopy נמוך לא "תשבור" רחוב ירוק שלם (הראוטר מעדיף את הרחוב כיחידה). **ה-TCI לא מושפע** — זו העדפת ניווט בלבד, מעבר לנוחות הגולמית. **הערה (2026-07-10):** אחרי איחוד צל-עצים ל-`shadow_cov`, שדרות אלה כבר מקבלות `shadow_cov` גבוה אמיתי-ותלוי-שעה (לא רק canopy_ratio סטטי) — ה-workaround הידני הזה עשוי להיות ניתן לפישוט/הסרה בעתיד, אך לא נבדק/שונה כחלק מהשינוי הנוכחי.
- חישובי צל = גיאומטריה 2D (לא 3D ray-casting מלא כמו Shadowmap).
- baseline = `DummyRegressor(mean)` (RMSE≈2.35) — הממוצע ממזער RMSE בהגדרה; median משני בגלל הטיה ימנית.
- **מגבלה מרכזית — יעד סינתטי:** ה-TCI הוא נוסחה שכתבנו, אז המודל "משחזר נוסחה" (R²≈0.9985 מנופח — עלה עוד יותר אחרי איחוד-הצל, כי הנוסחה המפושטת ל-`1-shadow_cov` קלה עוד יותר לשחזור). `shadow_cov` המאוחד קירב למציאות אך לא שבר את המעגליות. השלב שישבור אותה = תוויות אמת מדודות (LST מלוויין דרך Earth Engine).
- `temperature`/`humidity`/`canopy_ratio` = decoys (לא בנוסחה) — נשמרו לבדיקת בחירת פיצ'רים (`canopy_ratio` הצטרף ב-2026-07-10 כשהוסר מהנוסחה לטובת `shadow_cov` המאוחד).
