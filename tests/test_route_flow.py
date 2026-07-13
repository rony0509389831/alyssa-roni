r"""
זרימת-הניווט וליבת אלגוריתם-הצל (Integration, מסלולים מפחידים 1+2).

הערך המרכזי של SHADY אינו "להחזיר מסלול" אלא **לבחור מסלול שונה לפי הצל**. לכן בונים
גרף סינתטי זעיר ומבוקר (ground-truth ידוע) עם שתי דרכים מ-A ל-B:
  • קצרה-חשופה  : A→X→B   (300מ', TCI גבוה=8 → חשיפה גבוהה)
  • ארוכה-מוצלת : A→Y→Z→B (600מ', TCI נמוך=2 → אין חשיפה גבוהה)
ובודקים: (א) שהמסלול המוצל **שונה** מהמהיר ולכיוון הנכון; (ב) שהמשקלים **באמת נכנסים
ל-A\***; (ג) אינvariantים אמיתיים על המסלול (לא רק "is not None"). הכל דטרמיניסטי, בלי
רשת ובלי artifacts.
"""
import math

import networkx as nx
import pytest

from src.routing import (
    compute_length_route, compute_shaded_route, plan_route, sun_position,
)

# קואורדינטות הצמתים (lon=x, lat=y) — צמודות כדי שהיוריסטיקת ה-A* תישאר admissible
_XY = {
    1: (34.7700, 32.0800),   # A (מוצא)
    2: (34.7705, 32.0800),   # X — אמצע הדרך הקצרה-החשופה
    3: (34.7710, 32.0800),   # B (יעד)
    4: (34.7700, 32.0805),   # Y — הדרך הארוכה-המוצלת
    5: (34.7705, 32.0805),   # Z
}
_A_LATLON = (_XY[1][1], _XY[1][0])   # (lat, lon) של המוצא
_B_LATLON = (_XY[3][1], _XY[3][0])   # (lat, lon) של היעד

# קשתות (מכוונות): דרך קצרה 1→2→3 (150+150), דרך ארוכה 1→4→5→3 (200×3)
_SHORT_EDGES = [(1, 2), (2, 3)]
_LONG_EDGES = [(1, 4), (4, 5), (5, 3)]
_LEN = {(1, 2): 150.0, (2, 3): 150.0, (1, 4): 200.0, (4, 5): 200.0, (5, 3): 200.0}

# TCI פר-קשת: הדרך הקצרה חשופה (8>5.5 → חשיפה גבוהה), הארוכה מוצלת (2<5.5 → אין)
_TCI_BY_UV = {**{e: 8.0 for e in _SHORT_EDGES}, **{e: 2.0 for e in _LONG_EDGES}}


def _mini_two_route_graph():
    """גרף MultiDiGraph עם שתי דרכים מקבילות A→B, אחת קצרה-חשופה ואחת ארוכה-מוצלת."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"                        # ox.nearest_nodes דורש crs
    for n, (x, y) in _XY.items():
        G.add_node(n, x=x, y=y)                         # כל צומת עם קואורדינטות
    for u, v in _SHORT_EDGES + _LONG_EDGES:
        G.add_edge(u, v, key=0, length=_LEN[(u, v)])    # קשתות מכוונות עם אורך
    return G


def _haversine_m(a, b):
    """מרחק ישר במטרים בין שתי נקודות (lat, lon) — לבדיקת קרבת קצות-המסלול למבוקש."""
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(h))


# ---------- מסלול מפחיד (2)+(1): הצל משנה את בחירת המסלול ----------

def test_shade_changes_route_choice():
    """מהיר(אורך) בוחר בקצרה-החשופה; מוצל(משקל-צל) בוחר בארוכה-המוצלת — מסלול שונה וכיוון נכון."""
    G = _mini_two_route_graph()
    # משקלי-צל: מענישים את הדרך החשופה (משקל גבוה) → A* המוצל יעדיף את הארוכה-המוצלת
    w_shaded = {**{e: _LEN[e] * 8.0 for e in _SHORT_EDGES},   # חשוף → משקל כבד
                **{e: _LEN[e] * 2.0 for e in _LONG_EDGES}}    # מוצל → משקל קל
    baseline = compute_length_route(_A_LATLON, _B_LATLON, tci_by_uv=_TCI_BY_UV, G=G)   # מסלול "מהיר"
    shaded = compute_shaded_route(_A_LATLON, _B_LATLON, w_shaded, _TCI_BY_UV, G=G)     # מסלול "מוצל"

    assert shaded["route_latlon"] != baseline["route_latlon"]              # בחרו מסלול **שונה**
    assert round(baseline["distance_m"]) == 300                            # המהיר = הקצרה (300מ')
    assert round(shaded["distance_m"]) == 600                              # המוצל = הארוכה (600מ')
    assert shaded["distance_m"] >= baseline["distance_m"]                  # מוצל לוקח זמן/מרחק ≥
    assert shaded["high_exposure_m"] < baseline["high_exposure_m"]         # מוצל = פחות חשיפה גבוהה
    assert shaded["high_exposure_m"] == 0.0                                # הארוכה-המוצלת: אפס חשיפה
    assert round(baseline["high_exposure_m"]) == 300                       # הקצרה-החשופה: כל 300מ' חשופים


# ---------- מסלול מפחיד (1): המשקלים באמת נכנסים ל-A* ----------

def test_weights_actually_drive_astar():
    """אותו גרף, שני מילוני-משקל הפוכים → שני מסלולים שונים. מוכיח שהמסלול נגזר מהמשקל,
    ולא ממרחק קבוע (אם מישהו יחליף בטעות למשקל-אורך — הטסט הזה יאדים)."""
    G = _mini_two_route_graph()
    favor_long = {**{e: 1000.0 for e in _SHORT_EDGES}, **{e: 300.0 for e in _LONG_EDGES}}
    favor_short = {**{e: 100.0 for e in _SHORT_EDGES}, **{e: 1000.0 for e in _LONG_EDGES}}
    r_long = compute_shaded_route(_A_LATLON, _B_LATLON, favor_long, _TCI_BY_UV, G=G)   # משקל מעדיף ארוך
    r_short = compute_shaded_route(_A_LATLON, _B_LATLON, favor_short, _TCI_BY_UV, G=G) # משקל מעדיף קצר

    assert r_long["route_latlon"] != r_short["route_latlon"]     # החלפת משקל → מסלול אחר
    assert round(r_long["distance_m"]) == 600                    # favor_long → הדרך הארוכה
    assert round(r_short["distance_m"]) == 300                   # favor_short → הדרך הקצרה


# ---------- אינvariantים אמיתיים (מחליף "route is not None") ----------

def test_route_invariants_hold():
    """מסלול תקין: ≥2 נקודות, אורך חיובי, קצוות קרובים למבוקש, והכל בתוך תיבת תל-אביב."""
    G = _mini_two_route_graph()
    w = {**{e: _LEN[e] * 8.0 for e in _SHORT_EDGES}, **{e: _LEN[e] * 2.0 for e in _LONG_EDGES}}
    r = compute_shaded_route(_A_LATLON, _B_LATLON, w, _TCI_BY_UV, G=G)
    coords = r["route_latlon"]

    assert len(coords) >= 2                                      # מסלול = לפחות מוצא ויעד
    assert r["distance_m"] > 0                                   # אורך חיובי (לא מסלול-אפס)
    assert _haversine_m(coords[0], _A_LATLON) < 50               # תחילת המסלול = ליד המוצא המבוקש
    assert _haversine_m(coords[-1], _B_LATLON) < 50              # סוף המסלול = ליד היעד המבוקש
    for lat, lon in coords:                                      # אף נקודה לא "בורחת" מחוץ לת"א
        assert 32.0 <= lat <= 32.15                              # קו-רוחב בתחום תל-אביב
        assert 34.7 <= lon <= 34.85                              # קו-אורך בתחום תל-אביב


# ---------- מסלול מפחיד (9): מוצא==יעד = קלט-קצה → שגיאה חיננית ----------

def test_same_origin_and_destination_raises_valueerror():
    """מוצא==יעד → אותו צומת → ValueError צפוי (לא קריסה סתמית), בשני מנועי-הניתוב."""
    G = _mini_two_route_graph()
    with pytest.raises(ValueError):                              # מסלול-אורך: מוצא==יעד → שגיאה מפורשת
        compute_length_route(_A_LATLON, _A_LATLON, tci_by_uv=_TCI_BY_UV, G=G)
    with pytest.raises(ValueError):                              # מסלול-מוצל: אותו טיפול
        compute_shaded_route(_A_LATLON, _A_LATLON, {}, _TCI_BY_UV, G=G)


# ---------- fallback לילה ב-plan_route (בלי רשת: compute_route מוזרק) ----------

def test_plan_route_night_falls_back_to_fast(monkeypatch):
    """בקשה מוצלת בשעת לילה (23:00) → plan_route נופל אוטומטית ל-mode='fast'/fallback='night'."""
    import src.routing as routing
    # מזייפים את compute_route (OSRM) כדי לא לגעת ברשת — בודקים את **החלטת ה-fallback**, לא את OSRM
    monkeypatch.setattr(routing, "compute_route",
                        lambda o, d: {"route_latlon": [o, d], "distance_m": 1.0, "duration_min": 1.0})
    # ודא שהשעה 23:00 אכן מתחת לאופק (תנאי-הלילה שמפעיל את ה-fallback)
    alt, _ = sun_position(23.0)
    assert alt <= 0                                              # 23:00 → שמש מתחת לאופק
    out = plan_route("A", "B", use_shaded=True, nav_hour=23.0,
                     geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
                     has_model=True)                             # מבקשים מוצל, אבל זה לילה
    assert out["mode"] == "fast"                                 # נכפה מהיר
    assert out["fallback"] == "night"                            # סיבת-הנפילה = לילה
