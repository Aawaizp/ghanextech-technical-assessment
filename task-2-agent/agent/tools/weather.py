"""Current conditions and forecast from Open-Meteo (no API key required)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import math
import logging
import time
from urllib.parse import urlencode

import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
logger = logging.getLogger(__name__)
REQUEST_HEADERS = {
    "User-Agent": "GhanexAssessmentAgent/1.0 (https://github.com/Aawaizp/ghanextech-technical-assessment)",
    "Accept": "application/json",
}
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    97: "heavy thunderstorm",
    99: "thunderstorm with heavy hail",
}


class WeatherError(RuntimeError):
    pass


class WeatherRateLimit(WeatherError):
    def __init__(self, message, retry_after=300):
        super().__init__(message)
        self.retry_after = retry_after


def retry_delay(value):
    """Honor Retry-After seconds or an HTTP date; otherwise wait five minutes."""
    try:
        if str(value).isdigit():
            return max(1, int(value))
        target = parsedate_to_datetime(value)
        return max(1, math.ceil((target - datetime.now(timezone.utc)).total_seconds()))
    except (ValueError, TypeError, OverflowError):
        return 300


def _get_json(http: requests.Session, url: str, params: dict) -> dict:
    stage = "location lookup" if url == GEOCODING_URL else "forecast"
    for attempt in range(2):
        try:
            response = http.get(url, params=params, timeout=(5, 10), headers=REQUEST_HEADERS)
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else response.status_code
            logger.warning("Open-Meteo %s failed: HTTP %s (attempt %s)", stage, status, attempt + 1)
            if status in (502, 503, 504) and attempt == 0:
                time.sleep(0.5)
                continue
            if status == 429:
                raise WeatherRateLimit(
                    f"Open-Meteo {stage}: rate limit reached (HTTP 429). Please wait before retrying",
                    retry_delay(response.headers.get("Retry-After")),
                ) from exc
            elif status in (401, 403):
                detail = f"access denied (HTTP {status}) from this app's server"
            elif status == 400:
                detail = "request rejected (HTTP 400). The weather request needs checking"
            else:
                detail = f"service returned HTTP {status}; please retry later"
            raise WeatherError(f"Open-Meteo {stage}: {detail}") from exc
        except requests.exceptions.SSLError as exc:
            logger.warning("Open-Meteo %s failed: TLS verification", stage)
            raise WeatherError(f"Open-Meteo {stage}: secure connection could not be verified") from exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            kind = "timed out" if isinstance(exc, requests.Timeout) else "connection failed"
            logger.warning("Open-Meteo %s %s (attempt %s)", stage, kind, attempt + 1)
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise WeatherError(f"Open-Meteo {stage}: {kind} after one retry; please try later") from exc
        except requests.RequestException as exc:
            logger.warning("Open-Meteo %s failed: %s", stage, type(exc).__name__)
            raise WeatherError(f"Open-Meteo {stage}: network request failed; please retry") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherError(f"Open-Meteo {stage}: service returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise WeatherError(f"Open-Meteo {stage}: service returned an invalid response")
    if payload.get("error"):
        raise WeatherError(f"Open-Meteo {stage}: service could not process the request")
    return payload


def _number(value, *, minimum=None, maximum=None, allow_none=True):
    if value is None and allow_none:
        return None
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise WeatherError("Weather service returned an invalid numeric value")
    return value


def _condition(code) -> str:
    if code is None:
        return "unavailable"
    if type(code) is not int:
        raise WeatherError("Weather service returned an invalid condition code")
    return WEATHER_CODES.get(code, f"weather code {code}")


def _place_label(place: dict) -> str:
    if not isinstance(place, dict) or not isinstance(place.get("name"), str):
        raise WeatherError("Location service returned an invalid place")
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    if any(part is not None and not isinstance(part, str) for part in parts):
        raise WeatherError("Location service returned an invalid place")
    return ", ".join(dict.fromkeys(part for part in parts if part))


def get_weather(location: str, *, session: requests.Session | None = None) -> dict:
    """Return seven days starting today in the resolved location's timezone.

    ``session`` is for dependency injection; it is not an agent tool argument.
    Null measurements remain null rather than being mistaken for zero.
    """
    if not isinstance(location, str) or not location.strip():
        raise WeatherError("Location must be a non-empty city name")
    location = location.strip()
    if len(location) > 160:
        raise WeatherError("Location cannot exceed 160 characters")
    # Open-Meteo searches a name, not a complete address. Qualifiers select a match.
    parts = [part.strip() for part in location.split(",") if part.strip()]
    if not parts:
        raise WeatherError("Location must be a non-empty city name")
    http = requests.Session() if session is None else session
    try:
        geo_params = {"name": parts[0], "count": 10, "language": "en", "format": "json"}
        geo_payload = _get_json(http, GEOCODING_URL, geo_params)
        matches = geo_payload.get("results", [])
        if not isinstance(matches, list) or any(not isinstance(p, dict) for p in matches):
            raise WeatherError("Location service returned invalid search results")
        if len(parts) > 1:
            matches = [
                place for place in matches
                if all(
                    qualifier.casefold() in {
                        str(place.get(field, "")).casefold()
                        for field in ("admin1", "admin2", "admin3", "admin4", "country", "country_code")
                    }
                    for qualifier in parts[1:]
                )
            ]
        if not matches:
            raise WeatherError(f"No location found for '{location}'. Try a city name, optionally followed by country.")
        place = matches[0]
        resolved_location = _place_label(place)
        latitude = _number(place["latitude"], minimum=-90, maximum=90, allow_none=False)
        longitude = _number(place["longitude"], minimum=-180, maximum=180, allow_none=False)
        forecast_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 7,
            "timezone": "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        }
        payload = _get_json(http, FORECAST_URL, forecast_params)
        current = payload.get("current")
        daily = payload.get("daily")
        timezone = payload.get("timezone")
        required_current = (
            "time", "weather_code", "temperature_2m", "apparent_temperature",
            "relative_humidity_2m", "wind_speed_10m",
        )
        required_daily = (
            "time", "weather_code", "temperature_2m_max", "temperature_2m_min",
            "precipitation_probability_max",
        )
        if (
            not isinstance(current, dict) or not isinstance(daily, dict)
            or not isinstance(timezone, str) or not timezone.strip()
            or any(key not in current for key in required_current)
            or any(not isinstance(daily.get(key), list) or len(daily[key]) != 7 for key in required_daily)
        ):
            raise WeatherError("Weather service returned an incomplete seven-day forecast; please retry")
        if not isinstance(current["time"], str):
            raise WeatherError("Weather service returned an invalid observation time")
        observation_date = datetime.fromisoformat(current["time"]).date()
        dates = [date.fromisoformat(value) for value in daily["time"]]
        if any(value != dates[0] + timedelta(days=index) for index, value in enumerate(dates)):
            raise WeatherError("Weather service returned inconsistent forecast dates")
        forecast = [
            {
                "date": day.isoformat(),
                "condition": _condition(code),
                "max_c": _number(maximum),
                "min_c": _number(minimum),
                "precipitation_probability_percent": _number(precipitation, minimum=0, maximum=100),
            }
            for day, code, maximum, minimum, precipitation in zip(
                dates, daily["weather_code"], daily["temperature_2m_max"],
                daily["temperature_2m_min"], daily["precipitation_probability_max"], strict=True,
            )
        ]
        alternatives = list(dict.fromkeys(_place_label(other) for other in matches[1:]))
        alternatives = [label for label in alternatives if label != resolved_location][:3]
        return {
            "location": resolved_location,
            "location_note": (
                "Using the first matching location. Confirm the region if another place was intended."
                if alternatives else "Location resolved by Open-Meteo geocoding."
            ),
            "alternative_locations": alternatives,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "local_date": observation_date.isoformat(),
            "tomorrow_date": (observation_date + timedelta(days=1)).isoformat(),
            "day_after_tomorrow_date": (observation_date + timedelta(days=2)).isoformat(),
            "forecast_start_date": dates[0].isoformat(),
            "current": {
                "time": current["time"],
                "condition": _condition(current["weather_code"]),
                "temperature_c": _number(current["temperature_2m"]),
                "feels_like_c": _number(current["apparent_temperature"]),
                "humidity_percent": _number(current["relative_humidity_2m"], minimum=0, maximum=100),
                "wind_kmh": _number(current["wind_speed_10m"], minimum=0),
            },
            "forecast": forecast,
            "source": "Open-Meteo",
            "source_url": f"{FORECAST_URL}?{urlencode(forecast_params)}",
            "geocoding_source_url": f"{GEOCODING_URL}?{urlencode(geo_params)}",
        }
    except WeatherError:
        raise
    except requests.Timeout as exc:
        raise WeatherError("Weather service timed out; please retry") from exc
    except requests.RequestException as exc:
        raise WeatherError("Weather service is unavailable; please retry") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise WeatherError("Weather service returned an invalid response; please retry") from exc
    finally:
        if session is None:
            http.close()
