"""
רשת-ביטחון לרגרסיה (Integration, נתונים אמיתיים): מוודאת שהיררכיית-הצל של המודל
האדיטיבי (2026-07-19) מתקיימת על מסלולים אמיתיים, כך ששום תיקון לא "יחזור אחורה"
בשקט. מריצה plan_route אמיתי (גרף+מודל+כיסוי-צל) על זוגות-קואורדינטות מקובעים (בלי
רשת) בשעת-צהריים ובשעה-קרירה, ובודקת את האינvariant המרכזי — **מונוטוניות ללא היפוך**:

  1. **TCI:** מהיר ≥ מעט צל ≥ הרבה צל (יותר תקציב-עיקוף = קריר יותר, לעולם לא חם יותר;
     בסיס-השוואה זהה = compute_length_route על הגרף). — מתקן את "מעט קריר מהרבה".
  2. **מרחק:** מהיר ≤ מעט צל ≤ הרבה צל (תקציב גדול יותר → λ קטן יותר → ארוך יותר;
     מובטח בהגדרה כי תקציב 1.60× מכיל את 1.30×). — מתקן את "מעט ארוך מהרבה".
"""
import os

import pytest

_ROOT   = os.path.join(os.path.dirname(__file__), "..")
_GRAPH  = os.path.join(_ROOT, "data", "tel_aviv_walk.graphml")
_MODEL  = os.path.join(_ROOT, "data", "tci_model.joblib")
_EDGES  = os.path.join(_ROOT, "data", "edges_features.parquet")
_SHADOW = os.path.join(_ROOT, "data", "shadow_coverage.parquet")

# דילוג נקי אם ה-artifacts חסרים (checkout ללא נתונים) — לא כשל-שווא
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (_GRAPH, _MODEL, _EDGES, _SHADOW)),
    reason="חסרים graphml/model/parquet artifacts — מדלגים על בדיקת-האינvariantים החיה",
)


@pytest.fixture(scope="module")
def env():
    import joblib
    import pandas as pd
    from src import routing
    G = routing.load_graph()
    ef = pd.read_parquet(_EDGES)
    bundle = joblib.load(_MODEL)

    def weights_fn(alt, az, cloud, temp, hum, lam):
        # הפרמטר האחרון הוא λ (מחיר-האורך במודל האדיטיבי) — plan_route קורא בערכי-λ
        # מגוונים בחיפוש-התקציב, לא ב-shade_factor 1.0/2.0.
        return routing.compute_tci_weights(ef, bundle, alt, az, cloud, temp, hum, lam=lam)

    return routing, G, weights_fn


# זוגות-קואורדינטות מקובעים (מוצא, יעד) — כולל דיזינגוף→שאול המלך ורבין→כרמל שהתפתלו
_PAIRS = [
    ("Dizengoff55->Shaul13", (32.075133, 34.775122), (32.075884, 34.783173)),
    ("RabinSq->CarmelMkt",   (32.0806, 34.7805),     (32.0687, 34.7690)),
]


@pytest.mark.parametrize("hr", [13.0, 16.0])   # צהריים (הבעייתי) + אחה"צ קריר
@pytest.mark.parametrize("name,origin,dest", _PAIRS)
def test_shaded_never_worse_than_fast_and_smooth(env, name, origin, dest, hr):
    routing, G, weights_fn = env
    def _plan(sf):
        return routing.plan_route(
            "O", "D", use_shaded=True, nav_hour=hr,
            geocode_fn=lambda a: origin if a == "O" else dest,
            weights_fn=weights_fn,
            weather_fn=lambda *a, **k: {"cloud_cover": 10.0, "temperature": 32.0, "humidity": 45.0},
            has_model=True, graph=G, shade_factor=sf,
        )

    p_little, p_lots = _plan(1.0), _plan(2.0)   # מעט צל / הרבה צל
    little, lots = p_little["route_result"], p_lots["route_result"]
    # "מהיר" = בדיוק מה ש-app.py מציג להשוואה (compute_length_route על הגרף, TCI מדויק)
    fast = routing.compute_length_route(origin, dest, p_lots.get("tci_uv"), G=G)
    ft, lt, mt = fast.get("avg_tci"), little.get("avg_tci"), lots.get("avg_tci")

    # מונוטוניות TCI: מהיר ≥ מעט ≥ הרבה (יותר צל = קריר יותר; אין היפוך). סבילות קטנה.
    if None not in (ft, lt, mt):
        assert lt <= ft + 0.05, f"{name}@{hr:.0f}: מעט TCI {lt:.2f} חם מהמהיר {ft:.2f} (היפוך!)"
        assert mt <= lt + 0.05, f"{name}@{hr:.0f}: הרבה TCI {mt:.2f} חם ממעט {lt:.2f} (לא מונוטוני!)"
    # מונוטוניות מרחק: מהיר ≤ מעט ≤ הרבה (משלמים על צל בזמן). סבילות ~מטר.
    assert little["distance_m"] >= fast["distance_m"] - 1, f"{name}@{hr:.0f}: מעט קצר מהמהיר"
    assert lots["distance_m"] >= little["distance_m"] - 1, f"{name}@{hr:.0f}: הרבה קצר ממעט"
