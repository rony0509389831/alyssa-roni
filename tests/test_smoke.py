"""
Smoke tests — בסיס מודל-היהלום (מסלול מפחיד 4: "ה-import והנכסים לא נופלים").

הרחבה מעבר ל-import: טעינת מודל ה-TCI והרצת predict אחד אמיתי. טעינה לבדה מוכיחה רק
שהקובץ נפתח; predict אמיתי מוכיח גם תאימות-גרסת-sklearn, מספר/סדר עמודות נכון, ופלט מספרי.
(טווח והתנהגות נבדקים בנפרד ב-test_model_contract.py.)
"""
import os

import numpy as np
import pandas as pd
import pytest

# נתיב המודל יחסית לשורש הריפו (הקובץ הזה יושב ב-tests/)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tci_model.joblib")


def test_import():
    import src                                                   # טסט-עשן טריוויאלי: החבילה נטענת


def test_model_loads_and_predicts_one_finite_value():
    """טוען את bundle המודל ומריץ predict על שורת-פיצ'רים אחת — הפלט חייב להיות מספר סופי יחיד."""
    if not os.path.exists(_MODEL_PATH):
        pytest.skip("data/tci_model.joblib חסר (checkout ללא artifacts) — מדלגים, לא נכשלים")
    import joblib
    from src.data import FEATURE_COLS                            # מקור-האמת לסדר ושמות הפיצ'רים

    bundle = joblib.load(_MODEL_PATH)                            # טעינת ה-bundle מהדיסק
    assert "model" in bundle and "features" in bundle           # מבנה ה-bundle כצפוי (model + features)
    assert list(bundle["features"]) == list(FEATURE_COLS)       # סדר הפיצ'רים בקובץ == מקור-האמת ב-data.py

    # שורת-קלט אחת בעמודות הנכונות (ערכי-אמצע סבירים) — DataFrame כי כך המודל אומן
    sample = pd.DataFrame([[45.0, 20.0, 0.3, 30.0, 27.0, 65.0, 0.4]], columns=FEATURE_COLS)
    pred = bundle["model"].predict(sample)                      # ה-predict בפועל (רץ scikit-learn אמיתי)
    assert len(pred) == 1                                        # קלט אחד → תחזית אחת
    assert np.isfinite(pred[0])                                  # פלט מספרי סופי (לא NaN/inf/קריסה)
