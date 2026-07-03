"""בדיקות ל-src/agent.py — ולידציה ו-Fallback בלבד (ללא קריאת רשת)."""
from src.agent import (
    _validate, _coerce_hour, _coerce_mode, _coerce_shade_level, _coerce_recommendation,
    extract_route_params, _ERROR_MSG,
)

_TODAY = "2026-07-03"


def test_valid_full():
    out = _validate({"origin": "כיכר רבין", "destination": "שוק הכרמל",
                     "hour": 15, "mode": "shaded"}, today_str=_TODAY)
    assert out == {
        "origin": "כיכר רבין",
        "destination": "שוק הכרמל",
        "hour": 15.0,
        "date": None,
        "mode": "shaded",
        "shade_level": None,
        "recommendation": None,
    }


def test_missing_origin_or_dest_is_error():
    assert "error" in _validate({"destination": "שוק הכרמל"}, today_str=_TODAY)
    assert "error" in _validate({"origin": "כיכר רבין", "destination": "  "}, today_str=_TODAY)
    assert "error" in _validate("not a dict", today_str=_TODAY)


def test_hour_clamped_and_snapped():
    assert _coerce_hour(15) == 15.0
    assert _coerce_hour(3) is None       # מתחת לטווח → None (שעת לילה)
    assert _coerce_hour(23) is None      # מעל הטווח → None (שעת לילה)
    assert _coerce_hour(8.3) == 8.5      # snap ל-0.5 הקרוב
    assert _coerce_hour(None) is None
    assert _coerce_hour("not a number") is None


def test_mode_normalization_defaults_to_shaded():
    assert _coerce_mode("fast") == "fast"
    assert _coerce_mode("מהיר") == "fast"
    assert _coerce_mode("shaded") == "shaded"
    assert _coerce_mode("מוצל") == "shaded"
    assert _coerce_mode(None) == "shaded"
    assert _coerce_mode("גיבוב לא צפוי") == "shaded"


def test_shade_level_coercion():
    assert _coerce_shade_level("short") == "short"
    assert _coerce_shade_level("balanced") == "balanced"
    assert _coerce_shade_level("shaded") == "shaded"
    assert _coerce_shade_level("max") == "max"
    assert _coerce_shade_level("הכי מהיר") == "short"
    assert _coerce_shade_level("מוצל") == "shaded"
    assert _coerce_shade_level("מקסימלי") == "max"
    assert _coerce_shade_level(None) is None
    assert _coerce_shade_level("גיבוב") is None    # לא ברור → לא לשנות


def test_recommendation_coercion():
    assert _coerce_recommendation(None) is None
    assert _coerce_recommendation("   ") is None
    assert len(_coerce_recommendation("א" * 150)) == 100
    assert _coerce_recommendation("🌙 חשוך") == "🌙 חשוך"


def test_empty_text_returns_error_without_network():
    assert extract_route_params("", "fake-key") == {"error": _ERROR_MSG}
    assert extract_route_params("   ", "fake-key") == {"error": _ERROR_MSG}
