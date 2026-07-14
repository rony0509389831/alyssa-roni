"""בדיקות ל-_request_with_retry + retry ב-_nominatim_search (ללא רשת אמיתית — requests.get מזויף)."""
import requests

from src import routing


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


def test_request_with_retry_recovers_from_transient_timeout(monkeypatch):
    calls = {"n": 0}

    def _flaky_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.Timeout("simulated transient timeout")
        return _FakeResponse(200, [{"lat": "32.08", "lon": "34.78"}])

    monkeypatch.setattr(requests, "get", _flaky_get)
    monkeypatch.setattr(routing.time, "sleep", lambda *_: None)  # לא לישון בפועל בטסט

    point = routing._nominatim_search("כיכר רבין")

    assert calls["n"] == 2                       # ניסיון ראשון נכשל, שני הצליח
    assert point == (32.08, 34.78)


def test_request_with_retry_recovers_from_rate_limit_status(monkeypatch):
    calls = {"n": 0}

    def _rate_limited_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429, [])
        return _FakeResponse(200, [{"lat": "32.05", "lon": "34.75"}])

    monkeypatch.setattr(requests, "get", _rate_limited_get)
    monkeypatch.setattr(routing.time, "sleep", lambda *_: None)

    point = routing._nominatim_search("יפת 20")

    assert calls["n"] == 2
    assert point == (32.05, 34.75)


def test_no_retry_on_legitimate_empty_result(monkeypatch):
    """200 עם תוצאה ריקה = 'לא נמצא' לגיטימי — לא ניסיון חוזר."""
    calls = {"n": 0}

    def _empty_get(url, **kwargs):
        calls["n"] += 1
        return _FakeResponse(200, [])

    monkeypatch.setattr(requests, "get", _empty_get)
    monkeypatch.setattr(routing.time, "sleep", lambda *_: None)

    point = routing._nominatim_search("מקום לא קיים בעליל")

    assert calls["n"] == 1                        # קריאה בודדת בלבד
    assert point is None


def test_raises_after_retries_exhausted(monkeypatch):
    calls = {"n": 0}

    def _always_fails(url, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("simulated persistent failure")

    monkeypatch.setattr(requests, "get", _always_fails)
    monkeypatch.setattr(routing.time, "sleep", lambda *_: None)

    try:
        routing._nominatim_search("כיכר רבין")
        assert False, "expected an exception after retries exhausted"
    except requests.exceptions.ConnectionError:
        pass

    assert calls["n"] == 2                        # ניסיון ראשון + ניסיון-חוזר אחד, לא יותר
