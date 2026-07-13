"""
חוזה-המודל (Integration, מסלול מפחיד 1): "המודל מחזיר מספר בטווח **ומגיב נכון לשינוי**".

בדיקת טווח ∈[1,10] לבדה חלשה — מודל ש"מחזיר תמיד 5" עובר אותה אך חסר-תועלת. לכן כאן
בודקים **התנהגות דיפרנציאלית**: לצל/עננות/שמש יש כיוון-השפעה נכון, ולפיצ'רי-ה-decoy
(טמפרטורה/לחות/חופת-עצים) השפעה זניחה. כל נקודות-הבדיקה נגזרות מהתפלגות-האימון האמיתית
(build_tci_df) — כדי לא לבדוק ערכים שה-RandomForest מעולם לא ראה (extrapolation).
"""
import os

import numpy as np
import pandas as pd
import pytest

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tci_model.joblib")
_EDGES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "edges_features.parquet")
_SHADOW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shadow_coverage.parquet")

# דילוג נקי אם ה-artifacts חסרים (checkout ללא נתונים) — לא כשל-שווא
pytestmark = pytest.mark.skipif(
    not (os.path.exists(_MODEL_PATH) and os.path.exists(_EDGES_PATH) and os.path.exists(_SHADOW_PATH)),
    reason="חסרים model/parquet artifacts — מדלגים על חוזה-המודל",
)


@pytest.fixture(scope="module")
def probe():
    """מכין: מודל טעון + סביבת-בדיקה מהתפלגות-האמת, ומחזיר פונקציית predict לנקודה בודדת."""
    import joblib
    from src.data import build_tci_df, FEATURE_COLS

    bundle = joblib.load(_MODEL_PATH)                           # ה-bundle: {"model", "features"}
    feats = list(bundle["features"])                           # סדר הפיצ'רים כפי שהמודל אומן
    df = build_tci_df(n=1500, seed=0)                          # דגימת-אימון אמיתית (in-distribution)
    base = df[feats].median()                                  # וקטור-בסיס מרכזי וריאלי

    def p(**overrides):
        """מנבא TCI לוקטור-הבסיס עם דריסת ערכי-פיצ'רים ספציפיים (נשאר בטווח-האימון)."""
        row = base.copy()                                      # מתחילים מהבסיס הריאלי
        for k, v in overrides.items():
            row[k] = v                                         # דורסים רק את מה שבודקים
        X = pd.DataFrame([row])[feats]                         # שורה בודדת בסדר-העמודות הנכון
        return float(bundle["model"].predict(X)[0])            # התחזית בפועל

    # ערכי-קצה מתוך ההתפלגות האמיתית (5%/95%) — מובטח שהמודל "ראה" אותם
    q = {c: (df[c].quantile(0.05), df[c].quantile(0.95)) for c in feats}
    return p, q, df, feats


# ---------- כיוון-ההשפעה של הפיצ'רים הפיזיים (מודל "תמיד-5" נכשל בכולם) ----------

def test_more_shade_lowers_tci(probe):
    """יותר צל (shadow_cov גבוה) → נוחות גבוהה → TCI **נמוך** יותר, לאותם תנאי-שמש."""
    p, q, df, _ = probe
    sun_hi = q["sun_altitude"][1]                              # שמש גבוהה (95%) → ניגודיות-צל מקסימלית
    exposed = p(shadow_cov=0.0, sun_altitude=sun_hi, cloud_cover=5.0)   # רחוב חשוף לגמרי
    shaded  = p(shadow_cov=0.9, sun_altitude=sun_hi, cloud_cover=5.0)   # רחוב מוצל מאוד
    assert exposed > shaded + 1.0                             # הבדל ממשי (לא זהה, לא הפוך)


def test_more_clouds_lower_tci(probe):
    """יותר עננות → פחות סיכון-שמש → TCI **נמוך** יותר."""
    p, q, df, _ = probe
    sun_hi = q["sun_altitude"][1]
    clear    = p(cloud_cover=5.0,  shadow_cov=0.0, sun_altitude=sun_hi)   # שמים בהירים
    overcast = p(cloud_cover=50.0, shadow_cov=0.0, sun_altitude=sun_hi)   # עננות (מקסימום-אימון)
    assert clear > overcast                                   # בהיר חשוף יותר מעונן


def test_higher_sun_raises_tci(probe):
    """שמש גבוהה יותר (צהריים) → חשיפה גבוהה יותר → TCI **גבוה** יותר מאשר שמש נמוכה."""
    p, q, df, _ = probe
    sun_lo, sun_hi = q["sun_altitude"]                        # שמש נמוכה מול גבוהה (מההתפלגות)
    low_sun  = p(sun_altitude=sun_lo, shadow_cov=0.0, cloud_cover=5.0)
    high_sun = p(sun_altitude=sun_hi, shadow_cov=0.0, cloud_cover=5.0)
    assert high_sun > low_sun                                 # צהריים חם יותר משעה נמוכה


# ---------- אדישות ל-decoys: המודל למד לשקלל את העמודות הנכונות ----------

def test_decoys_have_negligible_effect_vs_real_feature(probe):
    """שינוי decoy (טמפ'/לחות/חופה) על כל טווחו משנה TCI **הרבה פחות** משינוי הצל האמיתי."""
    p, q, df, _ = probe
    sun_hi = q["sun_altitude"][1]
    # אמפליטודת-ההשפעה של פיצ'ר אמיתי (צל) — הבסיס להשוואה
    real_swing = p(shadow_cov=0.0, sun_altitude=sun_hi, cloud_cover=5.0) \
        - p(shadow_cov=0.9, sun_altitude=sun_hi, cloud_cover=5.0)
    assert real_swing > 1.0                                   # ודא שיש בכלל אות אמיתי להשוות מולו
    for decoy in ("temperature", "humidity", "canopy_ratio"):
        lo, hi = df[decoy].min(), df[decoy].max()             # מריצים את ה-decoy על כל טווחו
        swing = abs(p(**{decoy: hi}) - p(**{decoy: lo}))      # כמה TCI זז מהחלפת ה-decoy לבד
        assert swing < real_swing * 0.3                       # השפעת-decoy << השפעת-הצל (עמודות נכונות)


# ---------- בדיקת-טווח (משנית — לבדה חלשה, כאן רק כרשת-ביטחון) ----------

def test_predictions_stay_in_valid_range(probe):
    """כל תחזית על שורות-אימון אמיתיות נשארת ב-[1,10] (הגדרת ה-TCI)."""
    p, q, df, feats = probe
    import joblib
    bundle = joblib.load(_MODEL_PATH)
    preds = bundle["model"].predict(df[feats].head(300))      # 300 שורות אמת
    assert preds.min() >= 1.0 - 1e-6                          # לא מתחת ל-1
    assert preds.max() <= 10.0 + 1e-6                         # לא מעל 10
