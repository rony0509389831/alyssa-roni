"""בדיקות ל-src/agent.py — ולידציה ו-Fallback בלבד (ללא קריאת רשת)."""
from src.agent import (
    _validate, _coerce_hour, _coerce_mode, extract_route_params, _ERROR_MSG,
)


def test_valid_full():
    out = _validate({"origin": "כיכר רבין", "destination": "שוק הכרמל",
                     "hour": 15, "mode": "shaded"})
    assert out == {"origin": "כיכר רבין", "destination": "שוק הכרמל",
                   "hour": 15.0, "mode": "shaded"}


def test_missing_origin_or_dest_is_error():
    assert "error" in _validate({"destination": "שוק הכרמל"})
    assert "error" in _validate({"origin": "כיכר רבין", "destination": "  "})
    assert "error" in _validate("not a dict")


def test_hour_clamped_and_snapped():
    assert _coerce_hour(15) == 15.0
    assert _coerce_hour(3) == 6.0        # מתחת לטווח -> clamp ל-6
    assert _coerce_hour(23) == 19.0      # מעל הטווח -> clamp ל-19
    assert _coerce_hour(8.3) == 8.5      # snap ל-0.5 הקרוב
    assert _coerce_hour(None) is None
    assert _coerce_hour("not a number") is None


def test_mode_normalization_defaults_to_shaded():
    assert _coerce_mode("fast") == "fast"
    assert _coerce_mode("מהיר") == "fast"
    assert _coerce_mode("shaded") == "shaded"
    assert _coerce_mode("מוצל") == "shaded"
    assert _coerce_mode(None) == "shaded"      # ברירת מחדל
    assert _coerce_mode("גיבוב לא צפוי") == "shaded"


def test_empty_text_returns_error_without_network():
    assert extract_route_params("", "fake-key") == {"error": _ERROR_MSG}
    assert extract_route_params("   ", "fake-key") == {"error": _ERROR_MSG}
