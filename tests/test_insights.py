"""בדיקות ל-compute_route_insights + compute_length_route (ללא רשת; גרף סינתטי זעיר)."""
import networkx as nx
from src.routing import compute_route_insights, compute_length_route, WALK_SPEED_MPM


def _mini_graph():
    """גרף MultiDiGraph זעיר: ישיר 1→3 = כביש מתפתל 1000מ', עוקף 1→2→3 = 200מ'.
    הקואורדינטות צמודות (קו-אווירי ≤ אורך הקשת) כדי שהיוריסטיקת ה-haversine תהיה
    admissible — כמו בגרף רחובות אמיתי (אורכי osmnx תמיד ≥ קו אווירי)."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(1, x=34.7700, y=32.080)
    G.add_node(2, x=34.7708, y=32.080)
    G.add_node(3, x=34.7710, y=32.080)
    for u, v in [(1, 3), (3, 1)]:
        G.add_edge(u, v, key=0, length=1000.0)
    for u, v in [(1, 2), (2, 1), (2, 3), (3, 2)]:
        G.add_edge(u, v, key=0, length=100.0)
    return G


def test_street_typo_correction():
    # מילון מדומה (עוקף טעינת גרף) לבדיקת לוגיקת difflib של _correct_street_spelling
    from src import routing
    saved = routing._STREET_NAMES_CACHE
    routing._STREET_NAMES_CACHE = ["דיזנגוף", "רוטשילד", "שינקין", "אלנבי", "בן יהודה"]
    try:
        c = routing._correct_street_spelling
        assert c("דיזינגוף 55") == "דיזנגוף 55"     # י' מיותרת → תוקן
        assert c("דיזנגוף 55") == "דיזנגוף 55"       # תקין → ללא שינוי
        assert c("רוטשילדד 20") == "רוטשילד 20"      # ד' כפולה → תוקן
        assert c("שוק הכרמל") == "שוק הכרמל"          # בלי מספר בית → לא נוגעים (POI)
        assert c("בלהבלה 5") == "בלהבלה 5"            # לא מוכר → ללא שינוי (בלי תיקון-שווא)
    finally:
        routing._STREET_NAMES_CACHE = saved


def test_street_correction_prefix_variant_and_poi_guard():
    # (מסלול מפחיד 3, DATA test) — הרחבה: וריאנט-קידומת "שדרות" ושמירה על POI.
    # _street_names האמיתי מזריק גם צורה מנוקת-קידומת של כל שם; כאן המילון-המדומה
    # כבר מכיל את שתי הצורות כדי לבדוק את difflib בלבד (בלי טעינת גרף).
    from src import routing
    saved = routing._STREET_NAMES_CACHE
    routing._STREET_NAMES_CACHE = ["שדרות רוטשילד", "רוטשילד", "דיזנגוף", "אלנבי"]
    try:
        c = routing._correct_street_spelling
        # שגיאת-כתיב בשם עם קידומת "שדרות" → מתוקן לצורה המלאה הקרובה במילון
        assert c("שדרות רוטשילדד 20") == "שדרות רוטשילד 20"   # ד' כפולה בשם-עם-קידומת → תוקן
        # שם עם מספר-בית תקין נשאר כמות-שהוא (אין שגיאה לתקן)
        assert c("רוטשילד 20") == "רוטשילד 20"                # תואם ישירות → ללא שינוי
        # POI (שם עסק, בלי מספר-בית) לא "מתוקן" בטעות לרחוב דומה — שומר על הכוונה המקורית
        assert c("קפה רוטשילד") == "קפה רוטשילד"              # אין מספר-בית → לא נוגעים (POI)
    finally:
        routing._STREET_NAMES_CACHE = saved


def test_compute_length_route_picks_shortest_by_length_on_multigraph():
    # רגרסיה: weight חייב להיות "length" ולא lambda — אחרת ב-MultiDiGraph נמדד מספר
    # צמתים (הדוגמה: הישיר 1→3 הוא קפיצה אחת אך 1000מ'; העוקף 1→2→3 הוא 200מ').
    G = _mini_graph()
    out = compute_length_route((32.080, 34.7700), (32.080, 34.7710), tci_by_uv={}, G=G)
    assert round(out["distance_m"]) == 200      # בחר את העוקף הקצר, לא את הישיר הארוך


def test_insights_basic():
    shaded   = {"duration_min": 30.0, "avg_tci": 2.0, "high_exposure_m": 100.0}
    baseline = {"duration_min": 24.0, "avg_tci": 4.0, "high_exposure_m": 400.0}
    ins = compute_route_insights(shaded, baseline)
    assert ins["extra_min"] == 6.0                       # 30 - 24
    assert ins["tci_saved"] == 2.0                       # 4 - 2 (מוצל קריר יותר)
    assert ins["high_exposure_min_saved"] == round(300.0 / WALK_SPEED_MPM, 1)
    assert ins["avg_tci_shaded"] == 2.0
    assert ins["avg_tci_baseline"] == 4.0


def test_insights_no_detour_all_zero():
    same = {"duration_min": 20.0, "avg_tci": 3.0, "high_exposure_m": 150.0}
    ins = compute_route_insights(same, dict(same))
    assert ins["extra_min"] == 0.0
    assert ins["tci_saved"] == 0.0
    assert ins["high_exposure_min_saved"] == 0.0


def test_insights_handles_missing_avg_tci():
    shaded   = {"duration_min": 30.0, "avg_tci": None, "high_exposure_m": 0.0}
    baseline = {"duration_min": 24.0, "avg_tci": None, "high_exposure_m": 0.0}
    ins = compute_route_insights(shaded, baseline)
    assert ins["tci_saved"] is None                      # אין TCI → אין חיסכון-TCI
    assert ins["extra_min"] == 6.0
