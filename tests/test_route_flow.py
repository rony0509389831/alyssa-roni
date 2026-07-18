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


# ---------- תקרת-עיקוף פר-רמה: חיפוש עדין (צעד 0.1, רצפה 0.1) + נפילה כשאין מסלול בתקציב ----------

def _stub_weights_fn_factory(tci_by_uv):
    """weights_fn מדומה: משקל = אורך × TCI^shade_factor לכל קשת (כמו compute_tci_weights
    האמיתי, בלי מודל/נתונים אמיתיים) — מאפשר לבדוק את לולאת ההתפשרות ב-plan_route
    בלי רשת/פרקט/מודל אמיתיים."""
    def _fn(alt_r, az_r, cloud_r, temp_r, hum_r, shade_factor):
        wdict = {e: _LEN[e] * (tci_by_uv[e] ** shade_factor) for e in tci_by_uv}
        return wdict, tci_by_uv
    return _fn


def test_fastest_tier_finds_cheap_middle_ground_when_one_exists():
    """גרף זה (קצר=300מ'/TCI=8, ארוך=600מ'/TCI=1) יש בו "נקודת-מתיקה" זולה:
    ב-shade_factor=1/3≈0.333 בדיוק שני המסלולים שווים במשקל; מתחתיה הקצר זול
    יותר. חיפוש עדין בצעדי 0.1 (לא 0.5!) יורד 1.0→0.9→...→0.3 ומגלה ש-0.3 כבר
    proceeds למסלול הקצר (300מ', בתוך תקרת 1.10× של 'הכי מהר' = 330מ') — בלי
    לרדת עד רצפה עיוורת-ל-TCI. זה בדיוק התיקון על-פני צעד-0.5 הישן, שהיה מדלג
    מעל הנקודה הזו וממשיך היישר לרצפה."""
    G = _mini_two_route_graph()
    tci_map = {**{e: 8.0 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}

    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_stub_weights_fn_factory(tci_map),
        has_model=True, graph=G, shade_factor=1.0,   # "הכי מהר"
    )

    assert out["mode"] == "shaded"                            # מסלול-צל אמיתי, לא נפילה
    assert out["capped"] is True                              # ההתפשרות אכן הופעלה
    assert out["shade_factor_used"] == 0.3                    # נמצאה נקודת-המתיקה, לא 0.0/1.0
    assert round(out["route_result"]["distance_m"]) == 300    # בתוך תקרת 1.10×


def test_fastest_tier_escalates_to_widest_when_its_own_budget_fails(monkeypatch):
    """הדרך הארוכה-המוצלת (381מ' לעומת בסיס 300מ' — כ-1.27×) חורגת מהתקציב
    הצר של "מעט צל" (2026-07-18: 1.20×/+10-דק', סף=360מ') גם ברצפה (0.1) —
    ה-TCI הגבוה מספיק (20.0) בקצרה-החשופה כדי שהארוכה תישאר מועדפת בכל
    shade_factor שנבדק, אז אין שום נקודת-מתיקה זולה בתוך 360מ'. "הרבה צל"
    לעומת זאת (תקרתו 1.60×=480מ') מקבל אותה בשקט, בלי צורך בהתפשרות כלל.
    התוצאה: במקום לוותר על צל לגמרי, plan_route מנסה אוטומטית את "הרבה צל"
    (התקרה הרחבה ביותר) לפני שהוא נכנע — וזו כן עומדת בתקציב שלה, אז
    "מעט צל" מקבל בפועל בדיוק את אותו מסלול כמו "הרבה צל", עם
    fallback='tier_escalated' (מגולה למשתמש), לא נפילה שקטה למסלול מהיר וחשוף."""
    import src.routing as routing
    monkeypatch.setattr(
        routing, "compute_route",
        lambda o, d, tci_by_uv=None, G=None: {
            "route_latlon": [o, d], "distance_m": 300.0, "duration_min": 3.75,
        },
    )
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    for n, (x, y) in _XY.items():
        G.add_node(n, x=x, y=y)
    _gentle_len = {(1, 2): 150.0, (2, 3): 150.0, (1, 4): 127.0, (4, 5): 127.0, (5, 3): 127.0}
    for u, v in _SHORT_EDGES + _LONG_EDGES:
        G.add_edge(u, v, key=0, length=_gentle_len[(u, v)])
    tci_map = {**{e: 20.0 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}

    def _weights_fn(alt_r, az_r, cloud_r, temp_r, hum_r, shade_factor):
        wdict = {e: _gentle_len[e] * (tci_map[e] ** shade_factor) for e in tci_map}
        return wdict, tci_map

    max_shade = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_weights_fn, has_model=True, graph=G, shade_factor=2.0,   # "הרבה צל" (exponent 2.0)
    )
    fastest = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_weights_fn, has_model=True, graph=G, shade_factor=1.0,   # "מעט צל"
    )

    assert max_shade["mode"] == "shaded"
    assert max_shade["capped"] is False                            # 1.27× כבר בתוך תקרת 1.60× — אין ריכוך
    assert max_shade["fallback"] is None                            # הצלחה טבעית, לא מדורגת
    assert round(max_shade["route_result"]["distance_m"]) == 381
    assert fastest["mode"] == "shaded"                              # לא ויתור! הורחב ל"הרבה צל"
    assert fastest["fallback"] == "tier_escalated"                  # מגולה למשתמש
    assert fastest["shade_factor_used"] == 2.0                      # WIDEST_TIER (הרבה צל) = 2.0
    assert round(fastest["route_result"]["distance_m"]) == 381      # אותו מסלול בדיוק כמו max_shade


def test_neither_tier_fits_but_best_effort_beats_fast(monkeypatch):
    """הדרך הארוכה-הקרירה ביותר (600מ', TCI=1.0) נשארת מועדפת בכל shade_factor
    שנבדק (גם ברצפה 0.1) בזכות פער-TCI קיצוני מול הקצרה (2000.0) — אז היא
    חורגת מהתקציב גם של 'הכי מהר' (1.10×=330מ') וגם של 'הרבה צל' (1.60×=480מ').
    בכל זאת היא קרירה משמעותית מהמסלול המהיר (avg_tci=1.0 מול 8.0 המדומה) —
    אז plan_route מציג אותה כ"מוצל" עם fallback='best_effort_over_budget'
    (חורג מהתקציב הרגיל, מגולה בבירור), במקום לזרוק את כל החיפוש ולהראות
    מסלול מהיר-וחם בלי שום הסבר."""
    import src.routing as routing
    monkeypatch.setattr(
        routing, "compute_route",
        lambda o, d, tci_by_uv=None, G=None: {
            "route_latlon": [o, d], "distance_m": 300.0, "duration_min": 3.75,
            "avg_tci": 8.0, "tci_list": [8.0], "high_exposure_m": 300.0,
        },
    )
    G = _mini_two_route_graph()
    tci_map = {**{e: 2000.0 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}

    def _weights_fn(alt_r, az_r, cloud_r, temp_r, hum_r, shade_factor):
        wdict = {e: _LEN[e] * (tci_map[e] ** shade_factor) for e in tci_map}
        return wdict, tci_map

    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_weights_fn, has_model=True, graph=G, shade_factor=1.0,
    )

    assert out["mode"] == "shaded"
    assert out["fallback"] == "best_effort_over_budget"
    assert round(out["route_result"]["distance_m"]) == 600      # הדרך הארוכה-הקרירה, גם שחורגת
    assert out["route_result"]["avg_tci"] == 1.0                 # קרירה באמת — לא "מוצל" שקרי
    assert out["capped"] is True


def test_neither_tier_fits_and_best_effort_not_cooler_falls_back_to_fast(monkeypatch):
    """כמו למעלה, אבל הפעם המסלול המהיר (המדומה) קריר אפילו יותר מהניסיון
    הכי טוב שנמצא (avg_tci=0.5 מול 1.0) — אין שום תועלת אמיתית בהצגת מסלול
    "מוצל" שחורג מהתקציב. נופלים בכנות למסלול מהיר (fallback='detour_cap_unreachable'),
    בדיוק כמו לפני התיקון — הנפילה הכנה עדיין קיימת, רק אחרי שני ניסיונות ולא אחד."""
    import src.routing as routing
    monkeypatch.setattr(
        routing, "compute_route",
        lambda o, d, tci_by_uv=None, G=None: {
            "route_latlon": [o, d], "distance_m": 300.0, "duration_min": 3.75,
            "avg_tci": 0.5, "tci_list": [0.5], "high_exposure_m": 0.0,
        },
    )
    G = _mini_two_route_graph()
    tci_map = {**{e: 2000.0 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}

    def _weights_fn(alt_r, az_r, cloud_r, temp_r, hum_r, shade_factor):
        wdict = {e: _LEN[e] * (tci_map[e] ** shade_factor) for e in tci_map}
        return wdict, tci_map

    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_weights_fn, has_model=True, graph=G, shade_factor=1.0,
    )

    assert out["mode"] == "fast"
    assert out["fallback"] == "detour_cap_unreachable"
    assert round(out["route_result"]["distance_m"]) == 300


def test_backoff_binary_search_returns_maximal_fitting_factor():
    """החיפוש הבינארי (שהחליף את הסריקה הלינארית, 2026-07-18) מחזיר את ה-shade_factor
    ה**גבוה ביותר** על רשת ה-0.1 שעדיין נכנס לתקציב — לא רק 'איזשהו' factor, וגם לא
    קופץ מעל הסף. כאן הסף האנליטי נופל בין 0.6 ל-0.7: short=300מ'/TCI=2.9 מול
    long=600מ'/TCI=1, תקציב 'מעט צל'=1.20×300=360מ'. short נבחר כל עוד
    300×2.9^sf<600 ⇔ 2.9^sf<2 ⇔ sf<0.65 — אז ה-decile הגבוה-ביותר שנכנס הוא 0.6
    (ב-0.7 כבר נבחרת הדרך הארוכה 600מ' שחורגת). מוודא שהבינארי מתכנס לערך-אמצע הנכון."""
    G = _mini_two_route_graph()
    tci_map = {**{e: 2.9 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}

    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_stub_weights_fn_factory(tci_map),
        has_model=True, graph=G, shade_factor=1.0,   # "מעט צל"
    )

    assert out["mode"] == "shaded"
    assert out["shade_factor_used"] == 0.6                    # decile גבוה-ביותר שנכנס (לא 0.1, לא 1.0)
    assert round(out["route_result"]["distance_m"]) == 300    # הדרך הקצרה, בתוך התקציב
    assert out["capped"] is True


# ---------- שער רווח-צל: לא לעקוף כשהשיפור זניח (הבאג ש-rony דיווחה) ----------

def _two_route_graph_with_lengths(short_tci, long_tci):
    """גרף A→B: קצר 300מ' (2×150) מול ארוך 340מ' (113+113+114), עם TCI פר-דרך."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    for n, (x, y) in _XY.items():
        G.add_node(n, x=x, y=y)
    lens = {(1, 2): 150.0, (2, 3): 150.0, (1, 4): 113.0, (4, 5): 113.0, (5, 3): 114.0}
    for u, v in _SHORT_EDGES + _LONG_EDGES:
        G.add_edge(u, v, key=0, length=lens[(u, v)])
    tci_map = {**{e: short_tci for e in _SHORT_EDGES}, **{e: long_tci for e in _LONG_EDGES}}

    def _wf(a, b, c, d, e, sf):
        return {x: lens[x] * (tci_map[x] ** sf) for x in tci_map}, tci_map
    return G, _wf


def test_negligible_shade_gain_returns_direct():
    """עיקוף חסר-תועלת: הדרך המוצלת (340מ') חוסכת רק 0.5 TCI מול הישירה (300מ', TCI=2)
    — מתחת ל-MIN_TCI_GAIN=1.0. אין טעם לעקוף כשהאזור כבר קריר, אז plan_route מחזיר
    את המסלול הישיר (fallback='shade_gain_negligible'). זה בדיוק הבאג שדווח: ב-10:00
    הכל TCI~2 והראוטר עדיין מתפתל בשביל שיפור זעיר."""
    G, _wf = _two_route_graph_with_lengths(short_tci=2.0, long_tci=1.5)   # פער 0.5 < 1.0
    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_wf, has_model=True, graph=G, shade_factor=2.0,
    )
    assert out["mode"] == "shaded"
    assert out["fallback"] == "shade_gain_negligible"
    assert round(out["route_result"]["distance_m"]) == 300      # הישיר, לא העיקוף 340
    assert out["shade_factor_used"] == 0.0


def test_meaningful_shade_gain_keeps_detour():
    """כשהעיקוף כן חוסך צל משמעותי (7 נקודות: ישיר TCI=8 → מוצל TCI=1) — העיקוף נשמר,
    השער לא נכנס. מוודא שהתיקון לא 'הורג' עיקופים מוצדקים."""
    G, _wf = _two_route_graph_with_lengths(short_tci=8.0, long_tci=1.0)   # פער 7 > 1.0
    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_wf, has_model=True, graph=G, shade_factor=2.0,
    )
    assert out["mode"] == "shaded"
    assert out.get("fallback") != "shade_gain_negligible"       # העיקוף המוצדק נשמר
    assert round(out["route_result"]["distance_m"]) == 340      # הדרך המוצלת הארוכה
