"""
לקוח Open-Meteo לנתוני מזג אוויר נוכחיים בתל אביב.
Fallback: קריאה מ-data/climate_fallback.json אם ה-API לא זמין.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import requests

TA_LAT, TA_LON = 32.08, 34.77
_FALLBACK = Path("data/climate_fallback.json")
_TIMEOUT = 5

_OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={TA_LAT}&longitude={TA_LON}"
    "&current=temperature_2m,relative_humidity_2m,cloud_cover"
    "&timezone=Asia%2FJerusalem"
)


def get_current_weather() -> dict:
    """מחזיר dict עם temperature, humidity, cloud_cover, source ('live'/'fallback'/'default')."""
    try:
        _t0 = time.monotonic()
        resp = requests.get(_OPEN_METEO_URL, timeout=_TIMEOUT)
        print(f"[TIMING] Open-Meteo weather: {time.monotonic() - _t0:.2f}s", flush=True)
        resp.raise_for_status()
        cur = resp.json()["current"]
        return {
            "temperature": cur["temperature_2m"],
            "humidity": cur["relative_humidity_2m"],
            "cloud_cover": cur["cloud_cover"],
            "source": "live",
        }
    except Exception:
        return _fallback()


def _fallback() -> dict:
    try:
        with open(_FALLBACK, encoding="utf-8") as f:
            monthly = json.load(f)
        month_idx = datetime.now().month - 1
        m = monthly[month_idx]
        return {
            "temperature": m["temperature"],
            "humidity": m["humidity"],
            "cloud_cover": m["cloud_cover"],
            "source": "fallback",
        }
    except Exception:
        return {"temperature": 27.0, "humidity": 65.0, "cloud_cover": 30.0, "source": "default"}
