"""בדיקות ל-src/agent.py — ולידציה ו-Fallback בלבד (ללא קריאת רשת)."""
from datetime import date
from src.agent import (
    _validate, _coerce_hour, _coerce_mode, _coerce_shade_level, _coerce_recommendation,
    extract_route_params, _ERROR_MSG,
    format_insight_fallback, recommend_route_insight, _weekday_reference,
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


def test_valid_night_hour_is_kept_not_nulled():
    out = _validate({"origin": "רוטשילד", "destination": "כיכר רבין", "hour": 22,
                     "mode": "fast", "shade_level": "short",
                     "recommendation": "night"}, today_str=_TODAY)
    assert out["hour"] == 22.0
    assert out["mode"] == "fast"
    assert out["shade_level"] == "short"


def test_night_hour_overrides_explicit_shade_preference():
    # "ב-5 בבוקר בדרך הכי מוצלת שיש" — גם אם ה-LLM החזיר shaded/max, שעת לילה
    # חייבת לכפות מסלול מהיר (הכלל הדטרמיניסטי ב-_validate).
    out = _validate({"origin": "כיכר רבין", "destination": "שוק הכרמל", "hour": 5,
                     "mode": "shaded", "shade_level": "max",
                     "recommendation": None}, today_str=_TODAY)
    assert out["hour"] == 5.0            # השעה שנמסרה נשמרת
    assert out["mode"] == "fast"
    assert out["shade_level"] == "short"
    assert out["recommendation"] == "🌙 שעות לילה — מסלול מהיר מומלץ"

    # שעת ערב מאוחרת (22) עם בקשת צל — אותו כלל
    out2 = _validate({"origin": "רוטשילד", "destination": "שוק הכרמל", "hour": 22,
                      "mode": "shaded", "shade_level": "max"}, today_str=_TODAY)
    assert out2["mode"] == "fast" and out2["shade_level"] == "short"

    # בלי שעה מפורשת אבל השמש מתחת לאופק (sun_altitude<=0) — גם כן מהיר
    out3 = _validate({"origin": "רוטשילד", "destination": "שוק הכרמל",
                      "mode": "shaded", "shade_level": "max"},
                     today_str=_TODAY, sun_altitude=-3.0)
    assert out3["mode"] == "fast" and out3["shade_level"] == "short"

    # שעת יום (15) עם בקשת צל — לא מושפע מהכלל
    out4 = _validate({"origin": "רוטשילד", "destination": "שוק הכרמל", "hour": 15,
                      "mode": "shaded", "shade_level": "max"}, today_str=_TODAY)
    assert out4["mode"] == "shaded" and out4["shade_level"] == "max"


def test_diminished_shade_maps_to_short():
    """בקשת-צל ממותנת ("קצת מוצלת") → "מעט צל" (short), לא "הרבה צל" (max)."""
    out = _validate({"origin": "לוינסקי", "destination": "בוגרשוב", "hour": 12,
                     "mode": "shaded", "shade_level": "max"},
                    today_str=_TODAY,
                    user_text="מלוינסקי לבוגרשוב בדרך קצת מוצלת")
    assert out["shade_level"] == "short"


def test_plain_shade_request_stays_max():
    """בקשת-צל לא-ממותנת ("בדרך מוצלת") נשארת max — ה-override חל רק עם מילת-המעטה."""
    out = _validate({"origin": "רוטשילד", "destination": "שוק הכרמל", "hour": 12,
                     "mode": "shaded", "shade_level": "max"},
                    today_str=_TODAY,
                    user_text="מרוטשילד לשוק הכרמל בדרך מוצלת")
    assert out["shade_level"] == "max"


def test_missing_origin_or_dest_is_error():
    assert "error" in _validate({"destination": "שוק הכרמל"}, today_str=_TODAY)
    assert "error" in _validate({"origin": "כיכר רבין", "destination": "  "}, today_str=_TODAY)
    assert "error" in _validate("not a dict", today_str=_TODAY)


def test_hour_clamped_and_snapped():
    assert _coerce_hour(15) == 15.0
    assert _coerce_hour(3) == 3.0        # שעת לילה נשמרת כערך אמיתי (לא None) — ר' night rule
    assert _coerce_hour(23) == 23.0      # שעת לילה נשמרת כערך אמיתי (לא None) — ר' night rule
    assert _coerce_hour(8.3) == 8.5      # snap ל-0.5 הקרוב
    assert _coerce_hour(None) is None
    assert _coerce_hour("not a number") is None
    assert _coerce_hour(-1) is None      # מחוץ לטווח שעון תקין
    assert _coerce_hour(24) is None      # מחוץ לטווח שעון תקין


def test_mode_normalization_defaults_to_shaded():
    assert _coerce_mode("fast") == "fast"
    assert _coerce_mode("מהיר") == "fast"
    assert _coerce_mode("shaded") == "shaded"
    assert _coerce_mode("מוצל") == "shaded"
    assert _coerce_mode(None) == "shaded"
    assert _coerce_mode("גיבוב לא צפוי") == "shaded"


def test_shade_level_coercion():
    """רק 2 רמות (2026-07-18): כל בקשת-צל, מפורשת או כללית, ממופה ל-max."""
    assert _coerce_shade_level("short") == "short"
    assert _coerce_shade_level("shaded") == "max"
    assert _coerce_shade_level("max") == "max"
    assert _coerce_shade_level("הכי מהיר") == "short"
    assert _coerce_shade_level("מוצל") == "max"
    assert _coerce_shade_level("מקסימלי") == "max"
    assert _coerce_shade_level(None) is None
    assert _coerce_shade_level("גיבוב") is None    # לא ברור → לא לשנות


def test_recommendation_coercion():
    assert _coerce_recommendation(None) is None
    assert _coerce_recommendation("   ") is None
    assert _coerce_recommendation("night") == "🌙 שעות לילה — מסלול מהיר מומלץ"
    assert _coerce_recommendation("HOT") == "☀️ חם ושמשי — בחרתי צל מקסימלי"
    assert _coerce_recommendation("warm") == "🌤 מזג אוויר חם — בחרתי מסלול מוצל"
    assert _coerce_recommendation("🌙 חשוך משובש") is None   # טקסט חופשי משובש → None


def test_weekday_reference_next_week():
    # 2026-07-11 = יום שבת. "יום שני בשבוע הבא" צריך להיות 13.7 (לא היום).
    ref = _weekday_reference(date(2026, 7, 11))
    monday = [l for l in ref.splitlines() if l.startswith("2026-07-13")][0]
    assert "יום שני" in monday and "שבוע הבא" in monday
    # מאמצע השבוע (רביעי 8.7): חמישי 9.7 = השבוע, שני 13.7 = שבוע הבא
    ref2 = _weekday_reference(date(2026, 7, 8))
    thu = [l for l in ref2.splitlines() if l.startswith("2026-07-09")][0]
    mon = [l for l in ref2.splitlines() if l.startswith("2026-07-13")][0]
    assert "השבוע" in thu
    assert "שבוע הבא" in mon


def test_empty_text_returns_error_without_network():
    assert extract_route_params("", "fake-key") == {"error": _ERROR_MSG}
    assert extract_route_params("   ", "fake-key") == {"error": _ERROR_MSG}


def test_format_insight_fallback():
    # מסלול עם עיקוף וחיסכון — משפט מלא
    s = format_insight_fallback({"extra_min": 4.0, "tci_saved": 1.6,
                                 "high_exposure_min_saved": 6.0})
    assert s.startswith("🌿")
    assert "4 דק'" in s and "1.6" in s and "חשיפה גבוהה" in s
    # אין עיקוף ואין חיסכון — הודעת "כבר מיטבי"
    s2 = format_insight_fallback({"extra_min": 0.0, "tci_saved": 0.0,
                                  "high_exposure_min_saved": 0.0})
    assert "כבר מיטבי" in s2


def test_recommend_route_insight_fallback_without_network():
    # מפתח ריק → אין קריאת רשת, מחזיר את המשפט המחושב מ-metrics_fn
    metrics = {"extra_min": 3.0, "tci_saved": 2.0, "high_exposure_min_saved": 5.0}
    out = recommend_route_insight("מרבין לכרמל", "", lambda: metrics)
    assert out == format_insight_fallback(metrics)
    # גם אם metrics_fn קורס — לא מתרסק, מחזיר מחרוזת (ריקה)
    def _boom():
        raise RuntimeError("no route")
    assert recommend_route_insight("x", "", _boom) == ""
