"""
רשת-ביטחון קבועה (Integration, נתונים אמיתיים): מקבעת 5 תרחישים אמיתיים שהתגלו
ידנית (2026-07-19/20) בעת בדיקת גייטי "האם העיקוף שווה את זה" ב-plan_route
(_TIER_MIN_RATIOS, גייט-שיפור-נראה-לעין `shade_gain_not_visible`, וגייט-התכנסות
`same_as_little_shade`) — כדי ששום תיקון עתידי לא "יחזיר" אחד מהם בשקט.

קואורדינטות מקובעות (geocode חד-פעמי, לא רשת בכל הרצה) ומזג-אוויר מקובע (לא
Open-Meteo חי) — כך שהבדיקה דטרמיניסטית ולא תלויה בשעה/עונה האמיתיים של הרגע.
"""
import os
from datetime import datetime

import pytest

_ROOT   = os.path.join(os.path.dirname(__file__), "..")
_GRAPH  = os.path.join(_ROOT, "data", "tel_aviv_walk.graphml")
_MODEL  = os.path.join(_ROOT, "data", "tci_model.joblib")
_EDGES  = os.path.join(_ROOT, "data", "edges_features.parquet")
_SHADOW = os.path.join(_ROOT, "data", "shadow_coverage.parquet")

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (_GRAPH, _MODEL, _EDGES, _SHADOW)),
    reason="חסרים graphml/model/parquet artifacts — מדלגים על בדיקת-הרגרסיה החיה",
)

# geocode חד-פעמי (2026-07-20) של הכתובות מהתרחישים האמיתיים שהתגלו בשיחה
BOGRASHOV8  = (32.0776757, 34.7682139)
LEVINSKY_MKT = (32.0596754, 34.7725201)
FLORENTIN56 = (32.0570212, 34.7670509)
DERECH_BEGIN36 = (32.072279, 34.789606)

# מזג-אוויר מקובע — קרוב לנצפה בפועל בזמן הבדיקה (בוקר קיצי נקי, ת"א)
_WEATHER = {"cloud_cover": 0.0, "temperature": 26.0, "humidity": 85.0}
_NOW = datetime(2026, 7, 20)


@pytest.fixture(scope="module")
def env():
    import joblib
    import pandas as pd
    from src import routing
    G = routing.load_graph()
    ef = pd.read_parquet(_EDGES)
    bundle = joblib.load(_MODEL)

    def weights_fn(alt, az, cloud, temp, hum, lam):
        return routing.compute_tci_weights(ef, bundle, alt, az, cloud, temp, hum, lam=lam)

    return routing, G, weights_fn


def _plan(env, origin, dest, hour, shade_factor):
    routing, G, weights_fn = env
    return routing.plan_route(
        "O", "D", use_shaded=True, nav_hour=hour,
        geocode_fn=lambda a, *_: origin if a == "O" else dest,
        weights_fn=weights_fn,
        weather_fn=lambda: _WEATHER,
        has_model=True, graph=G, now=_NOW, shade_factor=shade_factor,
    )


def test_bograshov_levinsky_830_no_visible_gain(env):
    """8:30 — עיקוף קיים (11% אורך) אבל בלי שיפור TCI נראה-לעין בשתי הרמות."""
    for sf in (1.0, 2.0):
        plan = _plan(env, BOGRASHOV8, LEVINSKY_MKT, 8.5, sf)
        assert plan["fallback"] == "shade_gain_not_visible", (sf, plan["fallback"])
        assert plan["mode"] == "fast"


def test_bograshov_levinsky_900_little_below_band_lots_clean(env):
    """9:00 — 'מעט צל' לא מוצא עיקוף מספיק (below-band), 'הרבה צל' מוצא מסלול-צל נקי."""
    little = _plan(env, BOGRASHOV8, LEVINSKY_MKT, 9.0, 1.0)
    assert little["fallback"] == "little_shade_below_band"

    lots = _plan(env, BOGRASHOV8, LEVINSKY_MKT, 9.0, 2.0)
    assert lots["fallback"] is None
    assert lots["mode"] == "shaded"


def test_bograshov_levinsky_1700_lots_converges_to_little(env):
    """17:00 — 'הרבה צל' מתכנס בדיוק ל'מעט צל' (אותו מסלול)."""
    little = _plan(env, BOGRASHOV8, LEVINSKY_MKT, 17.0, 1.0)
    assert little["fallback"] is None
    assert little["mode"] == "shaded"

    lots = _plan(env, BOGRASHOV8, LEVINSKY_MKT, 17.0, 2.0)
    assert lots["fallback"] == "same_as_little_shade"
    assert lots["mode"] == "shaded"
    assert lots["route_result"]["distance_m"] == pytest.approx(
        little["route_result"]["distance_m"], abs=1.0)


def test_florentin_begin_900_lots_no_gain_shows_little_route(env):
    """
    9:00 — פלורנטין 56 -> דרך בגין 36: 'הרבה צל' נמצא ארוך יותר מ'מעט צל' (מרחק שונה,
    לא אותו מסלול) אבל בלי שיפור TCI-מעוגל — הגייט המוכלל (לא רק שוויון-מרחק) צריך
    להחליף אותו במסלול הקצר של 'מעט צל', לא רק להוסיף הודעה מעל המסלול הארוך.
    """
    little = _plan(env, FLORENTIN56, DERECH_BEGIN36, 9.0, 1.0)
    lots = _plan(env, FLORENTIN56, DERECH_BEGIN36, 9.0, 2.0)

    assert lots["fallback"] == "same_as_little_shade"
    assert lots["mode"] == "shaded"
    # המסלול המוצג ל"הרבה צל" הוא בפועל מסלול "מעט צל" (קצר, לא 60 הדקות המקוריות)
    assert lots["route_result"]["distance_m"] == pytest.approx(
        little["route_result"]["distance_m"], abs=1.0)
