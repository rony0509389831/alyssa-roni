# CLAUDE.md

## Project Purpose
SHADY — מערכת ניווט עירוני להולכי רגל המוצאת מסלולים מוצלים על בסיס גבהי מבנים, חופת עצים, מיקום השמש ומזג האוויר בזמן אמת.

---

## Data Sources

| מקור | פורמט | תוכן עיקרי |
|------|--------|------------|
| עיריית ת"א (opendata.tel-aviv.gov.il) | GeoJSON | ~45,900 פוליגוני מבנים עם `gova_simplex_2019` (גובה), `ms_komot` (קומות) |
| מפ"י (data.gov.il) | GeoJSON | ~231,000 פוליגוני חופת עצים באזור ת"א עם `geometry` |
| OSMnx | NetworkX Graph | רשת רחובות ת"א עם `length`, `highway`, `geometry` |
| Open-Meteo API | JSON (שעתי) | `temperature_2m`, `relative_humidity_2m`, `cloud_cover` |
שימוש גם בספריות פייתון הבאות עבור ממשק משתמש, זווית שמש וכו':
pandas
streamlit
requests
geopandas
osmnx
folium
streamlit-folium
pyogrio
pysolar
scikit-learn
pytest
**חוסרים ידועים:** ~4% מגבהי המבנים הם NaN — מטופל ע"י מודל ML פנימי (לא ממתין לנתונים נוספים).

---

## ML Model

- **משימה:** רגרסיה — חיזוי Thermal Comfort Index (ציון 1–10) לכל קשת בגרף הרחובות.
- **פיצ'רים:** `azimuth`, `mean_building_height`, `tree_canopy_ratio`, `sun_altitude`, `sun_azimuth`, `temperature`, `humidity`, `cloud_cover`
- **חישוב Sun Position:** PySolar
- **Loss:** MSE | **מטריקה:** RMSE
- **חלוקה:** Train 70% / Validation 15% / Test 15% (ללא פיצול לפי רחוב — כל קבוצה מכילה שעות יממה שונות)
- **Baseline:** Linear Regression (למודל) + Dijkstra גיאומטרי (לניווט)
- **ספריה:** Scikit-Learn

---

## Architecture & Stack

```
OSMnx → GeoPandas (Spatial Join, buffer=5m) → Feature Vector
Open-Meteo + PySolar → Feature Vector
Feature Vector → Scikit-Learn Model → Edge Weights
Edge Weights → NetworkX Dijkstra → Optimal Route
Route → Folium Map → Streamlit UI
```

**Fallback:** אם Open-Meteo לא זמין — טעינת נתוני אקלים ממוצעים שמורים מקומית.

---

## File Structure (מתוכנן)

```
alyssa-roni/
├── app.py                  # Streamlit entry point
├── data/
│   ├── buildings.geojson
│   ├── trees.geojson
│   └── climate_fallback.json
├── src/
│   ├── spatial.py          # GeoPandas processing & spatial joins
│   ├── solar.py            # PySolar wrappers
│   ├── model.py            # ML model training & inference
│   ├── routing.py          # NetworkX / Dijkstra logic
│   └── weather.py          # Open-Meteo API client
├── notebooks/              # EDA & model development
├── requirements.txt
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
- גבהי מבנים חסרים מנובאים ב-ML (לא מדלגים עליהם)
- הניווט משתמש ב-Dijkstra על גרף NetworkX שמגיע מ-OSMnx — לא לבנות גרף מאפס
- חישובי צל מבוססים על גיאומטריה 2D (לא סימולציית 3D מלאה)
