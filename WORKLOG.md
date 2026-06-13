# WORKLOG — SHADY

יומן עבודה. כל ערך = מושב עבודה אחד, עם תאריך, מה נעשה, בעיות שעלו ופתרונן.
הערה: חותמות הזמן הן ברמת המושב (לא ברמת הצעד הבודד).

---

## 2026-06-08 (יום שני), ~20:18 — M3: שלבים 1–5 (בניית בסיס למודל)

**מטרת המושב:** להתקדם ב-M3 עד שלב 5 (baseline "טיפש") לפי מסמך ההנחיות `M3_מסלול_מנתונים_למודל_סטודנטים.md`. עבודה בסגנון לימודי: הסבר מושגי לפני כל שורת קוד.

### מה עשינו

**שלבים 1–2 (החלטות, ללא קוד):**
- שלב 1 — סוג המשימה: **רגרסיה** (TCI הוא מספר רציף 1–10, לא קטגוריה).
- שלב 2 — KPI: **RMSE** כמדד ראשי (תואם ל-Loss=MSE שהמודלים ממזערים), **R²** כבונוס.

**שלב 3 (קוד) — חלוקת train/test:**
- נוצר `src/data.py` עם `build_tci_df(n, seed)` — בונה טבלת אימון (7 פיצ'רים + TCI) מתוך `data/edges_features.parquet`, ללא תלות ב-Streamlit.
- נוצר `src/model.py` עם `load_train_test()` — פיצול 80/20, `random_state=42`, פיצול לפי שורה (לא לפי רחוב).

**שלב 4 (preprocessing) — נותח, כמעט ולא נדרשה עבודה:**
- encoding: לא נדרש — כל 7 הפיצ'רים מספריים.
- missing values: כבר טופל במעלה הזרם (`fillna(0)` ב-`spatial.py` וב-`data.py`).
- scaling: אופציונלי למודלים שלנו (Linear Regression / Random Forest); חובה רק אם נוסיף KNN.

**שלב 5 (קוד) — baseline טיפש:**
- נוספו ל-`model.py`: `evaluate()` (RMSE + R²) ו-`run_baselines()` (DummyRegressor: mean + median).
- **mean** = הרצפה הראשית (הקבוע שממזער RMSE). **median** = רצפה משנית (כי ההתפלגות מוטה ימינה).

### ממצאים מספריים (מאומתים בהרצה)
- התפלגות TCI: mean=4.08, median=3.74, skew=0.78 (הטיה ימנית מתונה), ללא outliers בעייתיים, 0% מסה על קצוות החיתוך [1,10].
- גרף ה-OSM מכיל **58,360 קשתות**; אנחנו דוגמים `n=5000` (≈8.5%) עם החזרה.
- תוצאות baseline על test:

  | strategy | predicts | RMSE | R² |
  |---|---|---|---|
  | mean | 4.062 | **1.772** ← הרצפה | −0.0018 |
  | median | 3.723 | 1.819 | −0.0546 |

- הדגמה משלימה (לא נשמרה בקוד): mean מנצח ב-RMSE, median מנצח ב-MAE (1.434 מול 1.414) — ממחיש שבחירת המדד קובעת מי הרצפה ההוגנת.

### בעיות שעלו ואיך נפתרו

1. **`build_tci_df` היתה כלואה ב-`app.py` (עם `@st.cache_data`).** ייבוא ממנה (`from app import ...`) היה מריץ את כל אפליקציית Streamlit ברמת המודול.
   → **פתרון:** הוצאת לוגיקת בניית הדאטה למודול נקי `src/data.py`, ש-`model.py` (ו-app/notebooks/tests בעתיד) יכולים לייבא בבטחה.

2. **`UnicodeEncodeError` בהדפסת עברית לקונסול Windows** (cp1252 לא תומך בעברית) — הפיל את הסקריפט.
   → **פתרון:** `sys.stdout.reconfigure(encoding="utf-8")` ב-`__main__`, באותה מוסכמה כמו `precompute_features.py`.

3. **כפתור ה-Run ב-VS Code נכשל** (`ModuleNotFoundError: No module named 'src'`) — כי הוא מריץ `python src/model.py` ישירות, והחבילה `src` לא מוכרת.
   → **פתרון:** להריץ כמודול מתיקיית השורש: `python -m src.model`.

4. **"נראה תקוע" בהרצה ראשונה** — לא באג; טעינת הספריות הכבדות (pandas/sklearn/pysolar) לוקחת כמה שניות, והפלט מופיע בסוף.
   → **פתרון:** להמתין ~10–20 שניות בהרצה הראשונה.

### החלטה: מקור אמת לפיצ'רים
- זוהתה אי-התאמה בין CLAUDE.md (רשם 8 פיצ'רים כולל `sun_azimuth`) לבין הקוד (7 פיצ'רים).
- **הוכרע:** מקור האמת הוא ה-TCI של הסטרימליט (`build_tci_df` ב-`app.py`) — 7 פיצ'רים. הקוד צודק; CLAUDE.md תוקן בהתאם.

### חוב טכני / לעתיד
- **כפילות:** `build_tci_df` קיימת גם ב-`app.py` וגם ב-`src/data.py`. נשארה בכוונה כדי לא לסכן את האפליקציה. ניקוי עתידי: להפנות את app.py ל-`src.data` ולמחוק את העותק.
- **כיוון השמש (`sun_azimuth`) — שיפור עתידי לנוסחת ה-TCI:** הנוסחה הנוכחית תופסת רק אורך צל (`cos(sun_altitude)`) ומתעלמת מכיוון השמש. בעולם אמיתי הצל תלוי בזווית היחסית בין אזימוט השמש לכיוון הרחוב. שדרוג: להכניס את האינטראקציה `|sun_azimuth − street_azimuth|` × גובה מבנה לנוסחה — ורק אז `sun_azimuth` הופך לפיצ'ר בעל ערך. כרגע הוא decoy ולכן הושמט בצדק.
- שלב 6 (מודלים אמיתיים: Linear Regression, Random Forest) — הצעד הבא; היעד: לנצח את RMSE≈1.77.

### קבצים שנגעו בהם
- `src/data.py` (נוצר), `src/model.py` (נוצר + עודכן), `CLAUDE.md` (עודכן), `WORKLOG.md` (נוצר).

---

## 2026-06-13, ~14:18 — M3: שלבים 6–8 (מודלים → הערכה → הגשה)

**מטרת המושב:** להשלים את M3 מקצה-לקצה — מודלים אמיתיים, הערכה מול ה-baseline, שמירת מודל וחיבור לסטרימליט. נעשה לאחר מעבר על מצגת מפגש 6 (`l6_pipeline.pptx`).

### מה עשינו
- **שלב 6:** `build_models()` (3 Pipelines: linear+StandardScaler, tree, forest) ו-`train_and_evaluate()`.
- **שלב 7:** טבלת השוואה מאוחדת + `select_winner()` (RMSE מינימלי) ב-`__main__`.
- **שלב 8a:** `save_model()` עם joblib → `data/tci_model.joblib` (bundle: model + סדר פיצ'רים).
- **שלב 8b:** `load_tci_model()` (`@st.cache_resource`) ב-app.py + **אפשרות C** — מפת TCI פר-edge בגרף 4, עם toggle נוסחה/ML וסליידר גובה-שמש.
- **שלב 8c–e:** `joblib` ל-requirements; סעיף M3 ב-README + יישור סעיף 5; עדכון WORKLOG/CLAUDE.
- **תצוגת תוצאות בסטרימליט (טאב אודות):** `model.py` כותב `data/model_results.json`; `load_model_results()` ב-app.py קורא אותו → טבלת השוואה, הכרזת מנצח, feature importances, והערת מהימנות. **מקור אמת אחד** (אין hardcode). תוקנו גם סתירות שהיו בטאב: Spatial Split/70-15-15 → 80/20 לפי שורה · baseline → DummyRegressor · טבלת פיצ'רים 8→7 · Feature Engineering סומן כעתידי.

### תוצאות (test)
baseline(mean) 1.772 → linear 0.400 → tree 0.210 → **forest 0.120 (R²=0.995)**. forest מנצח את הרצפה פי 14.8.

### בדיקות מהימנות (חשוב להגשה)
- פער train↔test: linear ~0 (תקין), tree 0.21 (**overfit** — train RMSE=0!), forest 0.074 (מרוסן). CV של forest: 0.128±0.014 (יציב).
- **תקרת היעד הסינתטי:** R² גבוה משקף שחזור הנוסחה האנליטית, לא חיזוי מציאות. פיצול לפי שורה = הערכה אופטימית. מתועד כמגבלה גלויה.

### החלטות עיצוב
- ממשק החיזוי: נבחרה **אפשרות C** (מפת ML פר-edge) — פותרת את הפער "רחוב = הרבה edges": כל מקטע מקבל TCI נפרד.
- נשמר המנצח כפי שהוערך על train (לא refit על כל הדאטה) כדי שה-RMSE המדווח יתאים למודל השמור.

### בעיות ופתרונן
- `mean_squared_error(squared=False)` deprecated ב-sklearn חדש → השתמשנו ב-`np.sqrt(mean_squared_error(...))`.
- `tci_model.joblib` במשקל ~32MB (RandomForest ברירת מחדל) — מקובל ל-git.
- **טאב אודות נטען לאט:** מפת גרף-4 (`st_folium` בלי `returned_objects=[]`) הפעילה ריצה-מחדש מלאה בכל גרירה/זום, ובכל ריצה נבנו מחדש ~9,800 קווים ונקראו 58k שורות parquet. → **פתרון:** `returned_objects=[]` ל-st_folium + `@st.cache_data` ל-`load_rothschild_edges` (קריאת/סינון הקשתות פעם אחת).

### קבצים שנגעו בהם
- `src/model.py` (build_models, train_and_evaluate, select_winner, save_model, כתיבת model_results.json, __main__)
- `app.py` (load_tci_model, load_model_results, load_rothschild_edges; גרף-4: toggle/slider/חיזוי ML + returned_objects=[]; טאב אודות: טבלת תוצאות + תיקוני סנכרון)
- `requirements.txt`, `README.md`, `CLAUDE.md`, `WORKLOG.md`
- נוצרו: `data/tci_model.joblib`, `data/model_results.json`
