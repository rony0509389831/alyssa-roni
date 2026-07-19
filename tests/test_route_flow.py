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
    _route_backtrack,
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


# ---------- מודל אדיטיבי: תקציב-עיקוף כשאיפה (λ מכוונן), לא תקרה ----------

def _stub_weights_fn_factory(tci_by_uv, lens=_LEN):
    """weights_fn מדומה למודל **האדיטיבי** (2026-07-19): משקל = (TCI-1)·אורך + λ·אורך
    לכל קשת (כמו weights_from_edge_tci האמיתי, factor=1, בלי מודל/נתונים) — מאפשר
    לבדוק את חיפוש-התקציב ב-plan_route בלי רשת/פרקט/מודל. הפרמטר האחרון = λ (מחיר-אורך).
    `lens` = אורכי-הקשתות (חייב להתאים לאורכי הגרף שנבדק, אחרת המשקל והמרחק לא עקביים)."""
    def _fn(alt_r, az_r, cloud_r, temp_r, hum_r, lam):
        wdict = {e: (tci_by_uv[e] - 1.0) * lens[e] + lam * lens[e] for e in tci_by_uv}
        return wdict, tci_by_uv
    return _fn


# גרף A→B: קצרה=300מ' (2×150), מוצלת=460מ' (155+155+150) — בין 1.30×300=390 ל-1.60×300=480.
_LEN460 = {(1, 2): 150.0, (2, 3): 150.0, (1, 4): 155.0, (4, 5): 155.0, (5, 3): 150.0}


def _mini_two_route_graph_long460():
    """כמו _mini_two_route_graph אך הדרך הארוכה = 460מ' (נכנס לתקציב 'הרבה צל' בלבד)."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    for n, (x, y) in _XY.items():
        G.add_node(n, x=x, y=y)
    for u, v in _SHORT_EDGES + _LONG_EDGES:
        G.add_edge(u, v, key=0, length=_LEN460[(u, v)])
    return G


def test_little_shade_never_exceeds_130():
    """מעט צל נשאר תחת תקרת 1.30× גם כשהמסלול המוצל ביותר ארוך בהרבה."""
    G = _mini_two_route_graph()
    tci_map = {**{e: 8.0 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}
    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_stub_weights_fn_factory(tci_map),
        has_model=True, graph=G, shade_factor=1.0,   # "מעט צל" (תקציב 1.30×)
    )
    assert out["mode"] == "fast"
    assert out["fallback"] == "little_shade_below_band"
    assert round(out["route_result"]["distance_m"]) == 300
    assert out["detour_ratio"] <= 1.30


def test_shadiest_tier_has_hard_160_cap():
    G = _mini_two_route_graph()
    tci_map = {**{e: 8.0 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}
    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_stub_weights_fn_factory(tci_map),
        has_model=True, graph=G, shade_factor=2.0,
    )
    assert out["detour_ratio"] <= 1.60
    assert round(out["route_result"]["distance_m"]) == 300


def test_little_shade_below_band_chooses_fastest():
    """מתחת ל-1.10× אין הצדקה למסלול נפרד: מוחזר המסלול המהיר ממש."""
    G = _mini_two_route_graph()
    tci_map = {e: 4.0 for e in _SHORT_EDGES + _LONG_EDGES}
    out = plan_route(
        "A", "B", use_shaded=True, nav_hour=13.0,
        geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
        weights_fn=_stub_weights_fn_factory(tci_map),
        has_model=True, graph=G, shade_factor=1.0,
    )
    assert out["mode"] == "fast"
    assert out["fallback"] == "little_shade_below_band"
    assert round(out["route_result"]["distance_m"]) == 300


def test_above_little_band_stays_separate_from_shadiest_tier():
    """מסלול צל ב-1.53× זמין רק להרבה צל; מעט צל נשאר עד 1.30×."""
    G = _mini_two_route_graph_long460()
    tci_map = {**{e: 8.0 for e in _SHORT_EDGES}, **{e: 1.0 for e in _LONG_EDGES}}

    def _run(sf):
        return plan_route(
            "A", "B", use_shaded=True, nav_hour=13.0,
            geocode_fn=lambda addr: _A_LATLON if addr == "A" else _B_LATLON,
            weights_fn=_stub_weights_fn_factory(tci_map, lens=_LEN460),
            has_model=True, graph=G, shade_factor=sf,
        )

    little, lots = _run(1.0), _run(2.0)
    assert round(lots["route_result"]["distance_m"]) == 460       # הרבה = המוצלת
    assert round(little["route_result"]["distance_m"]) == 300     # מעט לא חורג מ-1.30×
    assert little["detour_ratio"] <= 1.30
    assert little["mode"] == "fast"


# ---------- מדד ה-backtrack (מדד-אימות-חלקות; נשאר לבדיקות החיות) ----------

def test_route_backtrack_helper():
    """מסלול שמתקדם מונוטונית ליעד → 0; מסלול שמתרחק ואז חוזר → אורך ההתרחקות הרציפה."""
    dest = (32.080, 34.780)
    # מונוטוני: כל נקודה קרובה יותר ליעד (ממערב-מזרח לאותו קו-אורך)
    mono = {"route_latlon": [(32.080, 34.770), (32.080, 34.775), (32.080, 34.780)]}
    assert _route_backtrack(mono, dest) == 0.0
    # הלוך-חזור: קרוב (471מ') → מתרחק (943מ', +472) → חוזר ליעד (0)
    back = {"route_latlon": [(32.080, 34.775), (32.080, 34.770), (32.080, 34.780)]}
    assert _route_backtrack(back, dest) > 400        # ההתרחקות ~472מ' זוהתה
    # מסלול ריק/נקודתי → 0 (הגנה)
    assert _route_backtrack({"route_latlon": [(32.08, 34.78)]}, dest) == 0.0


# ---------- תיקון-כתיב לא מחליף רחוב תקין ברחוב בעל שלד-שם דומה ----------

def test_correct_street_spelling_prefers_real_street(monkeypatch):
    """הבאג (2026-07-19): "שדרות שאול המלך 55" תוקן ל-"שדרות דוד המלך 55" כי difflib
    בחר את "שדרות דוד המלך" (שלד "שדרות...המלך" משותף גבר על המילה המבדילה), למרות
    ש"שאול המלך" קיים בגרף. עכשיו: שם תקין (עם/בלי קידומת) לא משתנה, וקידומת מנורמלת
    לצורה שבגרף — לא לרחוב אחר."""
    import src.routing as routing
    fake = ["שאול המלך", "שדרות דוד המלך", "דוד המלך", "דיזנגוף", "רוטשילד", "שדרות רוטשילד"]
    monkeypatch.setattr(routing, "_street_names", lambda: fake)
    assert routing._correct_street_spelling("שדרות שאול המלך 55") == "שאול המלך 55"   # לא "דוד המלך"!
    assert routing._correct_street_spelling("שאול המלך 55") == "שאול המלך 55"          # תקין → ללא שינוי
    assert routing._correct_street_spelling("דיזינגוף 55") == "דיזנגוף 55"             # תיקון-שגיאת-כתיב נשמר
    assert routing._correct_street_spelling("שוק הכרמל") == "שוק הכרמל"                # אין מספר-בית → לא נוגעים
