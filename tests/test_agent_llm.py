"""
בדיקות-הבנה חיות ל-LLM (Integration, מסלולים מפחידים 1+3) — דורשות מפתח Groq.

זו שכבת-הליבה של מודל-היהלום: הסוכן מול Groq **אמיתי**. רק כאן נבדקת ההבנה עצמה — לא
ב-mock (שהוא ה-unit השביר שהיהלום פוסל: "green != real"). כדי לחסוך מכסה, ברירת-המחדל
היא מודל **קטן** (llama-3.1-8b-instant); אפשר לשדרג ל-70b דרך SHADY_TEST_MODEL.
כל טסט בודק את **כל השדות הקריטיים לבקשה** (לא רק אחד), ומריץ **כמה וריאציות ניסוח**.

הרצה:  GROQ_API_KEY=... python -m pytest tests/test_agent_llm.py -v
(בלי מפתח — כל הקובץ מדולג אוטומטית, ולכן pytest tests/ הרגיל נשאר חינמי.)
"""
import os
from datetime import date

import pytest

# טעינת מפתח מ-.env בשורש הריפו (gitignored) — לנוחות הרצה מקומית; נופל בחן אם אין קובץ
_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV):
    for _line in open(_ENV, encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_KEY = os.getenv("GROQ_API_KEY")

# גדר: בלי מפתח אמיתי — מדלגים על כל הקובץ (dummy מ-.env של דמו לא נחשב)
pytestmark = pytest.mark.skipif(
    not _KEY or _KEY == "dummy",
    reason="אין GROQ_API_KEY אמיתי — מדלגים על בדיקות ה-LLM החיות",
)

_TODAY_D = date(2026, 7, 13)   # יום שני קבוע → "מחר"=14.7, "מחרתיים"=15.7 (דטרמיניסטי)


@pytest.fixture(autouse=True)
def _use_small_model(monkeypatch):
    """מחליף למודל קטן/זול לחיסכון-מכסה. ניתן לשדרג: SHADY_TEST_MODEL=llama-3.3-70b-versatile."""
    monkeypatch.setattr("src.agent.MODEL",
                        os.getenv("SHADY_TEST_MODEL", "llama-3.1-8b-instant"))


# שגיאות-תשתית זמניות של Groq (rate-limit / רשת / מודל-לא-זמין) — לא כשל-הבנה של המודל.
# מזוהות לפי הניסוח; שגיאות-הניווט האמיתיות ("לא הצלחתי להבין"/"מזג האוויר") לא מכילות אותן.
_TRANSIENT = ("מוגבלת", "חיבור לרשת", "אינו זמין")


def _run(text, **ctx):
    """קריאה אחת חיה לסוכן עם תאריך-ייחוס קבוע. מחזיר את ה-dict המחולץ (או {error}).

    אם Groq החזיר שגיאת-תשתית זמנית (429/רשת) — מדלגים על הטסט (skip) במקום להיכשל:
    זו מגבלת-מכסה, לא הוכחה שהמודל לא הבין. כך אדום = כשל-הבנה אמיתי בלבד."""
    from src.agent import extract_route_params
    out = extract_route_params(text, _KEY, today=_TODAY_D, **ctx)
    _err = out.get("error", "") if isinstance(out, dict) else ""
    if any(t in _err for t in _TRANSIENT):
        pytest.skip(f"שגיאת-תשתית זמנית מ-Groq (לא כשל-הבנה): {_err}")
    return out


def _has(field, *variants):
    """True אם השדה מכיל אחת מהחלופות (עברית/אנגלית) — עמיד לשפת-הפלט של ה-LLM."""
    s = (field or "").lower()
    return any(v.lower() in s for v in variants)


# ---------- הערה 4: כל השדות הקריטיים (שומר מפני החלפת מוצא↔יעד) ----------

def test_all_critical_fields_extracted_correctly():
    """'מחרתיים מדיזנגוף לרוטשילד, הכי מהר' — בודקים מוצא, יעד, תאריך ומצב יחד (לא רק שדה אחד)."""
    out = _run("מחרתיים מדיזנגוף לרוטשילד, הכי מהר")
    assert "error" not in out                                  # בקשה תקינה — לא אמורה לשגות
    assert _has(out["origin"], "דיזנגוף", "dizengoff")         # מוצא = דיזנגוף (לא הוחלף עם היעד!)
    assert _has(out["destination"], "רוטשילד", "rothschild")   # יעד = רוטשילד (לא הוחלף עם המוצא!)
    assert out["date"] == "2026-07-15"                         # "מחרתיים" מ-13.7 = 15.7
    assert out["mode"] == "fast"                               # "הכי מהר" → מסלול מהיר
    # שים לב: את נוסח ה-recommendation (טקסט חופשי) **לא** בודקים — רק את משמעות-הקלט


# ---------- הערה 5: כמה וריאציות ניסוח לכל כוונה ----------

# הערה: "לא אכפת לי מהצל" הוחזר לרשימה אחרי שהוגדר במפורש בפרומפט של הסוכן שאדישות-לצל =
# העדפת-מהירות (mode=fast). אימות ראשוני הראה ששני המודלים פירשו אותו כ"אין העדפה"→shaded;
# ההחלטה המוצרית (לבקשת המשתמשת) היא ש"לא אכפת לי מצל" צריך להוביל למסלול מהיר, וכך תוקן.
@pytest.mark.parametrize("phrase", [
    "הכי מהר",              # ניסוח ישיר
    "אני ממהרת",            # רמז-מהירות עקיף (לחץ-זמן)
    "לא אכפת לי מהצל",      # אדישות-לצל = העדפת-מהירות (הוגדר מפורשות בפרומפט)
    "תקצרי לי את הדרך",     # בקשת קיצור מפורשת
    "המסלול הכי קצר",       # ניסוח "קצר" מפורש
])
def test_fast_intent_across_phrasings(phrase):
    """כל ניסוח שמרמז על מהירות → mode=='fast'. בודק הבנה, לא משפט-יחיד שאולי בפרומפט."""
    out = _run(f"מכיכר רבין לשוק הכרמל, {phrase}")             # מעטפת-ניווט מלאה (מוצא+יעד קיימים)
    assert "error" not in out                                 # בקשה תקינה
    assert out["mode"] == "fast"                              # הכוונה = מהיר, בכל ניסוח


@pytest.mark.parametrize("phrase", [
    "הכי מוצל",             # בקשת-צל ישירה
    "צל מקסימלי",           # ניסוח "מקסימלי"
    "הכי נעים שיש",         # נוחות = צל (ניסוח עקיף)
])
def test_shaded_intent_across_phrasings(phrase):
    """כל ניסוח שמרמז על צל/נוחות → mode=='shaded' (בשעת-יום, בלי כלל-הלילה)."""
    out = _run(f"מכיכר רבין לשוק הכרמל בשעה 14, {phrase}")     # 14:00 = יום → כלל-הלילה לא מתערב
    assert "error" not in out
    assert out["mode"] == "shaded"                           # הכוונה = מוצל, בכל ניסוח


# ---------- הערה 9: קלט מבולגן (לא רק happy-path) ----------

def test_nonsense_request_returns_error():
    """שטות שאינה בקשת-ניווט ('להזמין פיצה', בלי מוצא) → {error}, לא מסלול מומצא."""
    out = _run("אני רוצה להזמין פיצה למחר בשעה עשר לדיזינגוף 55")
    assert "error" in out                                    # אין מוצא / לא ניווט → שגיאה


def test_partial_input_missing_origin_returns_error():
    """קלט חלקי — יעד בלבד בלי מוצא → {error} (לא ממציאים מוצא)."""
    out = _run("אני רוצה להגיע לשוק הכרמל")
    assert "error" in out                                    # חסר מוצא → שגיאה


def test_far_future_date_is_rejected_end_to_end():
    """תאריך רחוק מפורש ('30 בדצמבר') → מקצה-לקצה מגיע לשגיאת מזג-האוויר (הפיצ'ר החדש)."""
    out = _run("מכיכר רבין לשוק הכרמל ב-30 בדצמבר")
    assert "error" in out                                    # >7 ימים → שגיאת תאריך


def test_hebrew_prefixes_are_stripped():
    """קידומות דקדוקיות 'מ'/'ל' מוסרות משמות-המקום ('מדיזנגוף'→'דיזנגוף')."""
    out = _run("מדיזנגוף לרוטשילד בשעה 15")
    assert "error" not in out
    assert _has(out["origin"], "דיזנגוף", "dizengoff")       # השם נכון
    assert not (out["origin"] or "").startswith("מדיזנגוף")  # הקידומת 'מ' לא נשארה דבוקה
    assert _has(out["destination"], "רוטשילד", "rothschild")


def test_spoken_hebrew_is_understood():
    """עברית מדוברת עם ציוני-דרך — מחלץ מוצא/יעד סבירים ומזהה כוונת-צל ('בלי להישרף')."""
    out = _run("אני ליד כיכר רבין ורוצה להגיע לשדרות רוטשילד בשעה 13 בלי להישרף בדרך")
    assert "error" not in out
    assert _has(out["origin"], "רבין", "rabin")              # מוצא = כיכר רבין
    assert _has(out["destination"], "רוטשילד", "rothschild") # יעד = רוטשילד
    assert out["mode"] == "shaded"                           # "בלי להישרף" → מוצל
