"""
Client for the National Weather Service (NWS) API (api.weather.gov).

Unlike MassiveClient, this needs NO API key - NWS is free and unauthenticated.
It only requires a descriptive User-Agent header identifying the calling app,
per NWS API etiquette (https://www.weather.gov/documentation/services-web-api).

Two-step resolution model, same shape as the assignment brief:
  1. Resolve a human location ("Chicago, IL" or "41.8781,-87.6298") to a NWS
     grid point via GET /points/{lat},{lon}.
  2. Use that grid point to fetch active alerts (by point) and the forecast
     discussion (narrative text) for that location.
"""

import os
import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import requests

_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_GEOCODE_URL = os.environ.get(
    "GEOCODE_BASE_URL", "https://nominatim.openstreetmap.org/search"
)
_USER_AGENT = os.environ.get(
    "WEATHER_USER_AGENT",
    "(superfan-radar-weather-practice, contact@example.com)",
)
_DEFAULT_TIMEOUT = 30

# Matches "41.8781,-87.6298" or "41.8781, -87.6298"
_LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


class WeatherClient:
    """Thin wrapper around the NWS API, plus a free Nominatim geocoding fallback."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get(self, url_or_path: str, params: dict[str, Any] | None = None) -> Any:
        """GET helper. Accepts either a path (joined to base_url) or a full URL,
        since NWS responses often embed absolute URLs (e.g. properties.forecast)
        that should be fetched as-is rather than re-prefixed."""
        url = url_or_path if url_or_path.startswith("http") else f"{self.base_url}{url_or_path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Location resolution
    # ------------------------------------------------------------------

    def geocode(self, location: str) -> tuple[float, float]:
        """Resolve a free-text location ('Chicago, IL') to (lat, lon) via the
        free Nominatim geocoding API (no key required). If `location` is
        already a 'lat,lon' pair, parses it directly instead of calling out."""
        m = _LATLON_RE.match(location)
        if m:
            return float(m.group(1)), float(m.group(2))

        resp = self._session.get(
            _GEOCODE_URL,
            params={"q": location, "format": "json", "limit": 1, "countrycodes": "us"},
            headers={"User-Agent": _USER_AGENT},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            raise ValueError(f"Could not geocode location: {location!r}")
        return float(results[0]["lat"]), float(results[0]["lon"])

    def get_grid_point(self, lat: float, lon: float) -> dict:
        """GET /points/{lat},{lon} - resolves a lat/lon to the NWS grid office
        (gridId, gridX, gridY) and the forecast/forecastHourly URLs for it."""
        data = self.get(f"/points/{lat:.4f},{lon:.4f}")
        return data.get("properties", {})

    # ------------------------------------------------------------------
    # Alerts (free-text description + instruction fields)
    # ------------------------------------------------------------------

    def get_active_alerts_for_point(self, lat: float, lon: float) -> list[dict]:
        """GET /alerts/active?point={lat},{lon} - active alerts that apply to
        this exact point (more precise than the state-wide ?area= filter for
        a per-location pipeline like this one)."""
        data = self.get("/alerts/active", params={"point": f"{lat},{lon}"})
        return data.get("features", [])

    # ------------------------------------------------------------------
    # Forecasts (free-text detailedForecast narrative per period)
    # ------------------------------------------------------------------

    def get_forecast(self, grid_props: dict) -> list[dict]:
        """GET the forecast URL embedded in a /points response. Returns the
        list of forecast periods, each with a detailedForecast narrative
        string (e.g. 'Sunny, with a high near 78...')."""
        forecast_url = grid_props.get("forecast")
        if not forecast_url:
            return []
        data = self.get(forecast_url)
        return data.get("properties", {}).get("periods", [])

    # ------------------------------------------------------------------
    # Normalization -> document records ready for weather_documents
    # ------------------------------------------------------------------

    def fetch_documents_for_location(self, location: str) -> list[dict]:
        """Resolve one location (geocode -> grid point) and return a list of
        normalized document dicts covering both its active alerts and its
        current forecast periods. Matches the shape expected by
        app.py's _upsert_weather_batch()."""
        lat, lon = self.geocode(location)
        grid_props = self.get_grid_point(lat, lon)

        docs: list[dict] = []
        docs.extend(self._normalize_alerts(location, self.get_active_alerts_for_point(lat, lon)))
        docs.extend(self._normalize_forecast(location, self.get_forecast(grid_props)))
        return docs

    @staticmethod
    def _normalize_alerts(location: str, alerts: list[dict]) -> list[dict]:
        docs = []
        for feature in alerts:
            props = feature.get("properties", {}) or {}
            description = (props.get("description") or "").strip()
            instruction = (props.get("instruction") or "").strip()
            narrative = "\n\n".join(p for p in (description, instruction) if p)
            docs.append(
                {
                    "id": props.get("id") or feature.get("id"),
                    "location": location,
                    "source_type": "alert",
                    "headline": props.get("event", "Weather Alert"),
                    "narrative_text": narrative,
                    "issued_at": props.get("sent") or props.get("effective"),
                    "payload": feature,
                }
            )
        return docs

    @staticmethod
    def _normalize_forecast(location: str, periods: list[dict]) -> list[dict]:
        docs = []
        for period in periods:
            narrative = (period.get("detailedForecast") or "").strip()
            if not narrative:
                continue
            start_time = period.get("startTime") or datetime.now(timezone.utc).isoformat()
            # Forecasts don't carry a stable id from NWS - derive one so
            # re-syncing the same location/period upserts instead of duplicating.
            dedup_key = f"{location}|{period.get('number')}|{start_time}"
            doc_id = "forecast-" + sha256(dedup_key.encode("utf-8")).hexdigest()[:24]
            docs.append(
                {
                    "id": doc_id,
                    "location": location,
                    "source_type": "forecast",
                    "headline": period.get("name", "Forecast"),
                    "narrative_text": narrative,
                    "issued_at": start_time,
                    "payload": period,
                }
            )
        return docs