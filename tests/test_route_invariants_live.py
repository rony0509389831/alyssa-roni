"""
רשת-ביטחון לרגרסיה (Integration, נתונים אמיתיים): מוודאת שכל תיקוני-הניווט מתקיימים
יחד על מסלולים אמיתיים, כך ששום תיקון לא "יחזור אחורה" בשקט. מריצה plan_route אמיתי
(גרף+מודל+כיסוי-צל) על זוגות-קואורדינטות מקובעים (בלי רשת) בשעת-צהריים ובשעה-קרירה,
ובודקת שני invariantים מרכזיים:

  1. **אין היפוך-TCI:** מסלול-הצל לעולם לא חם יותר מהמהיר (בסיס-השוואה זהה =
     compute_length_route על הגרף). — תיקון "מוצל חם מהמהיר".
  2. **אין פיתול:** ה-backtrack של מסלול-הצל מוגבל ע"י MAX_BACKTRACK_M. — תיקון
     "העיקול המוזר ביהודה הלוי".
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

    def weights_fn(alt, az, cloud, temp, hum, sf):
        return routing.compute_tci_weights(ef, bundle, alt, az, cloud, temp, hum, shade_factor=sf)

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
    plan = routing.plan_route(
        "O", "D", use_shaded=True, nav_hour=hr,
        geocode_fn=lambda a: origin if a == "O" else dest,
        weights_fn=weights_fn,
        weather_fn=lambda *a, **k: {"cloud_cover": 10.0, "temperature": 32.0, "humidity": 45.0},
        has_model=True, graph=G, shade_factor=2.0,
    )
    shaded = plan["route_result"]
    # בסיס-ההשוואה = בדיוק מה ש-app.py מציג כ"מהיר" (compute_length_route על הגרף)
    fast = routing.compute_length_route(origin, dest, plan.get("tci_uv"), G=G)
    st, ft = shaded.get("avg_tci"), fast.get("avg_tci")

    # invariant 1 — אין היפוך: המוצל לעולם לא חם מהמהיר
    if st is not None and ft is not None:
        assert st <= ft + 0.01, f"{name}@{hr:.0f}: shaded TCI {st:.2f} > fast {ft:.2f} (היפוך!)"

    # invariant 2 — אין פיתול: ה-backtrack מוגבל ע"י התקרה
    bt = routing._route_backtrack(shaded, dest)
    assert bt <= routing.MAX_BACKTRACK_M + 5, (
        f"{name}@{hr:.0f}: backtrack {bt:.0f}m > תקרה {routing.MAX_BACKTRACK_M:.0f}m (פיתול!)")
