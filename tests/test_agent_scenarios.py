"""
בדיקות-תרחיש דטרמיניסטיות ל-src/agent.py — ללא מפתח, ללא רשת.

זו **שכבת-בסיס דקה** (Unit) של מודל-היהלום: היא בודקת את מנגנוני-ההגנה הדטרמיניסטיים
שעוטפים את ה-LLM (ולידציה, כלל-לילה, טווח-תאריכים), לא את הבנת ה-LLM עצמו — זו נבדקת
ב-test_agent_llm.py (Integration חי). כל טסט מאמת **ערך/התנהגות**, לא רק "לא-None"
(עיקרון "green != real").
"""
from datetime import date, timedelta
from src.agent import (
    _validate, _coerce_hour, _coerce_mode, _coerce_date, _date_out_of_range,
    _weekday_reference, _ERROR_MSG, _DATE_RANGE_ERR,
)

# תאריך-ייחוס קבוע לכל הטסטים (2026-07-13 = יום שני) — עצמאי מהתאריך האמיתי של ההרצה,
# כדי שהטסטים יהיו דטרמיניסטיים ולא "יישברו מחר".
_TODAY_D = date(2026, 7, 13)
_TODAY = _TODAY_D.isoformat()   # "2026-07-13"


# ---------- מסלול מפחיד (3): הוולידציה דוחה זבל ----------

def test_missing_origin_or_destination_is_rejected():
    """DATA test: בקשת ניווט בלי מוצא או בלי יעד = זבל → חייבת להחזיר {error}, לא לנחש כתובת."""
    only_dest = _validate({"destination": "שוק הכרמל", "hour": 10}, today_str=_TODAY)
    assert "error" in only_dest                                  # יש יעד אבל אין מוצא → שגיאה
    only_origin = _validate({"origin": "כיכר רבין", "hour": 10}, today_str=_TODAY)
    assert "error" in only_origin                                # יש מוצא אבל אין יעד → שגיאה
    blank_dest = _validate({"origin": "כיכר רבין", "destination": "   "}, today_str=_TODAY)
    assert "error" in blank_dest                                 # יעד רק רווחים → נחשב חסר → שגיאה
    not_dict = _validate("זו לא dict בכלל", today_str=_TODAY)
    assert not_dict == {"error": _ERROR_MSG}                     # פלט-LLM לא-תקין → שגיאה ידידותית


# ---------- מסלול מפחיד (3) + פיצ'ר חדש (TDD): תאריך מחוץ לאופק מזג-האוויר ----------

def test_far_future_date_returns_weather_error():
    """תאריך מעבר ל-7 ימים → שגיאת 'מזג אוויר לא זמין' (במקום נפילה שקטה להיום)."""
    far = (_TODAY_D + timedelta(days=10)).isoformat()            # today+10 = מחוץ לטווח
    out = _validate({"origin": "כיכר רבין", "destination": "שוק הכרמל",
                     "date": far}, today_str=_TODAY)
    assert out == {"error": _DATE_RANGE_ERR}                     # שגיאת-תאריך ספציפית, לא ניווט


def test_date_within_window_is_accepted():
    """תאריך בתוך 7 הימים → עובר כרגיל (לא נחסם)."""
    near = (_TODAY_D + timedelta(days=3)).isoformat()            # today+3 = בתוך הטווח
    out = _validate({"origin": "כיכר רבין", "destination": "שוק הכרמל",
                     "date": near}, today_str=_TODAY)
    assert "error" not in out                                    # אין שגיאה
    assert out["date"] == near                                   # התאריך נשמר כמות-שהוא


def test_past_date_returns_weather_error():
    """תאריך-עבר → גם הוא מחוץ לאופק התחזית → שגיאה (אין מזג-אוויר לעבר)."""
    past = (_TODAY_D - timedelta(days=1)).isoformat()            # אתמול = לפני today
    out = _validate({"origin": "כיכר רבין", "destination": "שוק הכרמל",
                     "date": past}, today_str=_TODAY)
    assert out == {"error": _DATE_RANGE_ERR}                     # עבר → שגיאה


def test_date_out_of_range_helper_boundaries():
    """גבולות _date_out_of_range במדויק: today ו-today+7 בתוך הטווח, today+8 מחוץ."""
    assert _date_out_of_range(None, _TODAY) is False                                # אין תאריך = היום → בטווח
    assert _date_out_of_range(_TODAY, _TODAY) is False                              # היום עצמו → בטווח
    assert _date_out_of_range((_TODAY_D + timedelta(days=7)).isoformat(), _TODAY) is False   # +7 = הקצה, בטווח
    assert _date_out_of_range((_TODAY_D + timedelta(days=8)).isoformat(), _TODAY) is True    # +8 → מחוץ לטווח


# ---------- מסלול מפחיד (1) + לילה: כל שעת-חושך כופה מסלול מהיר ----------

def test_night_hour_forces_fast_route():
    """שעת לילה (22) → נכפה mode=fast, shade_level=short, recommendation=לילה (דטרמיניסטי)."""
    out = _validate({"origin": "רוטשילד", "destination": "כיכר רבין", "hour": 22,
                     "mode": "shaded", "shade_level": "max"}, today_str=_TODAY)
    assert out["mode"] == "fast"                                 # לילה → מהיר, גובר על "shaded" שנתבקש
    assert out["shade_level"] == "short"                         # לילה → אין טעם בצל
    assert out["recommendation"] == "🌙 שעות לילה — מסלול מהיר מומלץ"   # סימון-לילה


def test_daytime_boundaries_are_not_night():
    """גבולות היום: 6.0 ו-19.0 הם עדיין 'יום' (מותר מוצל); 5.5 ו-19.5 הם 'לילה' (נכפה מהיר)."""
    day_start = _validate({"origin": "א", "destination": "ב", "hour": 6.0,
                           "mode": "shaded"}, today_str=_TODAY)
    assert day_start["mode"] == "shaded"                         # 06:00 = תחילת היום, לא לילה
    day_end = _validate({"origin": "א", "destination": "ב", "hour": 19.0,
                         "mode": "shaded"}, today_str=_TODAY)
    assert day_end["mode"] == "shaded"                           # 19:00 = סוף היום, לא לילה
    just_before = _validate({"origin": "א", "destination": "ב", "hour": 5.5,
                             "mode": "shaded"}, today_str=_TODAY)
    assert just_before["mode"] == "fast"                         # 05:30 → לפני היום → לילה → מהיר
    just_after = _validate({"origin": "א", "destination": "ב", "hour": 19.5,
                            "mode": "shaded"}, today_str=_TODAY)
    assert just_after["mode"] == "fast"                          # 19:30 → אחרי היום → לילה → מהיר


def test_night_by_sun_altitude_when_no_hour():
    """בלי שעה מפורשת אבל השמש מתחת/על האופק (sun_altitude<=0) → לילה → מהיר."""
    below = _validate({"origin": "א", "destination": "ב", "mode": "shaded"},
                      today_str=_TODAY, sun_altitude=-2.0)
    assert below["mode"] == "fast"                               # שמש מתחת לאופק → לילה
    horizon = _validate({"origin": "א", "destination": "ב", "mode": "shaded"},
                        today_str=_TODAY, sun_altitude=0.0)
    assert horizon["mode"] == "fast"                             # שמש בדיוק על האופק (0) → נחשב לילה


# ---------- מסלול מפחיד (9): קלט-קצה דטרמיניסטי (לא רק happy-path) ----------

def test_unexpected_date_format_is_dropped_to_none():
    """פורמט תאריך לא-צפוי (dd/mm/yyyy, מילה) → _coerce_date מחזיר None (=היום), לא קורס."""
    assert _coerce_date("15/07/2026", _TODAY) is None            # פורמט אירופאי → לא YYYY-MM-DD → None
    assert _coerce_date("מחר", _TODAY) is None                   # מילה עברית (ה-LLM היה אמור לפרש) → None
    assert _coerce_date("2026-07-20", _TODAY) == "2026-07-20"    # פורמט תקין → נשמר


def test_same_origin_and_destination_still_validates_here():
    """מוצא==יעד: הסוכן לא חוסם (שניהם קיימים) — הטיפול-בקצה קורה בראוטר (ר' test_route_flow)."""
    out = _validate({"origin": "כיכר רבין", "destination": "כיכר רבין", "hour": 12},
                    today_str=_TODAY)
    assert out["origin"] == out["destination"] == "כיכר רבין"    # שניהם עוברים — לא באחריות הסוכן לחסום


# ---------- בסיס-coercion דק (בדיקות קלות — לא עיקר הביטחון) ----------

def test_thin_base_coercion_defaults():
    """בדיקות-בסיס קלות: ברירות-מחדל של הקוארסרים. שכבה דקה בכוונה — הערך המרכזי בטסטים למעלה."""
    assert _coerce_mode(None) == "shaded"                        # ברירת-מחדל: מוצל
    assert _coerce_hour(10) == 10.0                              # שעה שלמה → float
    assert _coerce_hour(8.3) == 8.5                              # snap ל-0.5 הקרוב


# ---------- מסלול מפחיד (1): טבלת-התאריכים שמוזרקת ל-LLM נכונה ----------

def test_weekday_reference_is_correct_for_fixed_today():
    """הטבלה שה-LLM בוחר ממנה תאריכים: ISO + שם-יום + תגית, לתאריך-ייחוס קבוע (2026-07-13, שני)."""
    ref = _weekday_reference(_TODAY_D)
    lines = ref.splitlines()
    assert lines[0].startswith(_TODAY)                           # שורה 0 = היום
    assert "יום שני" in lines[0] and "היום" in lines[0]          # 13.7 = יום שני, מתויג "היום"
    tomorrow = (_TODAY_D + timedelta(days=1)).isoformat()
    assert lines[1].startswith(tomorrow)                         # שורה 1 = מחר (14.7)
    # "שבוע הבא" צריך להופיע על תאריך בשבוע ISO הבא (ראשון 19.7 ואילך)
    next_sunday = [l for l in lines if l.startswith("2026-07-19")][0]
    assert "שבוע הבא" in next_sunday                             # 19.7 (ראשון) מתויג "שבוע הבא"
