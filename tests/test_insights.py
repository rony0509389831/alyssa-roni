"""בדיקות ל-compute_route_insights (חישוב תובנות המסלול) — פונקציה טהורה, ללא רשת/גרף."""
from src.routing import compute_route_insights, WALK_SPEED_MPM


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
