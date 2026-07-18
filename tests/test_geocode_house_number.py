"""בדיקות לוולידציית מספר-בית ב-geocode_address (2026-07-18) — ללא רשת אמיתית
(requests.get מזויף). מכסה שני באגים: (1) Nominatim/Photon נופלים בשקט לרמת-
רחוב כשמספר הבית לא סביר ("יפת 4900") ותוצאת-הרחוב התקבלה בטעות כאילו הייתה
מדויקת; (2) Photon שלח פרמטר "lang"="he" לא-נתמך, מה שהפיל כל קריאה אליו
בשקט — כל שלב-הגיבוי הזה מעולם לא עבד בפועל."""
import pytest
import requests

from src import routing


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


# ---------- _nominatim_search: ולידציית house_number ----------

def test_nominatim_rejects_street_level_fallback_for_implausible_house_number(monkeypatch):
    """Nominatim נופל לרמת-רחוב (בלי house_number בתשובה) כש-4900 לא קיים
    ברחוב יפת — התוצאה חייבת להידחות, לא להתקבל כאילו הייתה מדויקת."""
    def _street_level_get(url, **kwargs):
        return _FakeResponse(200, [{
            "lat": "32.0514569", "lon": "34.7529653",
            "addresstype": "road", "class": "highway",
            "address": {"road": "יפת", "city": "תל אביב-יפו"},   # אין house_number בכלל
        }])
    monkeypatch.setattr(requests, "get", _street_level_get)

    point = routing._nominatim_search("יפת 4900, תל אביב", house_number="4900")
    assert point is None


def test_nominatim_accepts_result_with_matching_house_number(monkeypatch):
    def _house_level_get(url, **kwargs):
        return _FakeResponse(200, [{
            "lat": "32.0353748", "lon": "34.7479907",
            "addresstype": "house", "class": "place",
            "address": {"road": "יפת", "house_number": "40", "city": "תל אביב-יפו"},
        }])
    monkeypatch.setattr(requests, "get", _house_level_get)

    point = routing._nominatim_search("יפת 40, תל אביב", house_number="40")
    assert point == (32.0353748, 34.7479907)


def test_nominatim_house_number_check_skipped_for_place_names(monkeypatch):
    """שאילתה בלי מספר-בית (שם רחוב/מקום) לא מושפעת מהבדיקה החדשה בכלל —
    house_number=None (ברירת המחדל) מדלג על הוולידציה."""
    def _road_get(url, **kwargs):
        return _FakeResponse(200, [{
            "lat": "32.06", "lon": "34.77",
            "addresstype": "road", "class": "highway",
            "address": {"road": "רוטשילד"},
        }])
    monkeypatch.setattr(requests, "get", _road_get)

    point = routing._nominatim_search("רוטשילד")
    assert point == (32.06, 34.77)


# ---------- _photon_search: הסרת lang + ולידציית house_number ----------

def test_photon_search_no_longer_sends_unsupported_lang_param(monkeypatch):
    """Photon דוחה lang='he' בשקט (הבאג שתוקן) — מוודאים שהפרמטר לא נשלח יותר,
    וש-house_number תואם מתקבל כרגיל."""
    captured = {}

    def _fake_get(url, **kwargs):
        captured.update(kwargs.get("params") or {})
        return _FakeResponse(200, {"features": [{
            "geometry": {"coordinates": [34.7479907, 32.0353748]},
            "properties": {"housenumber": "40", "street": "יפת"},
        }]})
    monkeypatch.setattr(requests, "get", _fake_get)

    point = routing._photon_search("יפת 40", house_number="40")
    assert "lang" not in captured
    assert point == (32.0353748, 34.7479907)


def test_photon_search_rejects_mismatched_house_number(monkeypatch):
    def _fake_get(url, **kwargs):
        return _FakeResponse(200, {"features": [{
            "geometry": {"coordinates": [34.7529653, 32.0514569]},
            "properties": {"street": "יפת"},   # אין housenumber בתוצאה — לא תואם ל-4900
        }]})
    monkeypatch.setattr(requests, "get", _fake_get)

    point = routing._photon_search("יפת 4900", house_number="4900")
    assert point is None


# ---------- geocode_address: מקצה-לקצה (Nominatim + Photon ממוקקים ישירות) ----------

def test_geocode_address_rejects_implausible_house_number_end_to_end(monkeypatch):
    """גם Nominatim וגם Photon 'לא מוצאים' התאמת-בית אמיתית (מדומה) — geocode_address
    חייב להיכשל בכנות (ValueError), לא להחזיר נקודה שגויה על הרחוב."""
    monkeypatch.setattr(routing, "_nominatim_search", lambda q, house_number=None: None)
    monkeypatch.setattr(routing, "_photon_search", lambda q, house_number=None: None)
    monkeypatch.setattr(routing, "_overpass_search", lambda name: None)

    with pytest.raises(ValueError):
        routing.geocode_address("יפת 4900")


def test_geocode_address_accepts_valid_house_number_end_to_end(monkeypatch):
    """מספר-הבית שנחלץ מהכתובת המקורית ("40") אכן מועבר ל-_nominatim_search."""
    monkeypatch.setattr(
        routing, "_nominatim_search",
        lambda q, house_number=None: (32.0353748, 34.7479907) if house_number == "40" else None,
    )
    lat, lon = routing.geocode_address("יפת 40")
    assert (lat, lon) == (32.0353748, 34.7479907)
