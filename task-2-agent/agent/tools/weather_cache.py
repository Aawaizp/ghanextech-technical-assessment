"""Bounded process-local cache for public weather data and a shared cooldown."""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import math
from threading import Lock
import time

from .weather import WeatherError, WeatherRateLimit, get_weather


class WeatherCache:
    def __init__(self, loader=get_weather, ttl=300, capacity=64):
        self.loader = loader
        self.ttl = ttl
        self.capacity = capacity
        self.entries = OrderedDict()
        self.blocked_until = 0
        self.lock = Lock()

    def get(self, location):
        if not isinstance(location, str) or not location.strip() or len(location) > 160:
            raise WeatherError("Location must be a non-empty city name of at most 160 characters")
        key = ",".join(" ".join(part.split()).casefold() for part in location.split(","))
        # Serialize misses so concurrent sessions cannot stampede a limited service.
        with self.lock:
            now = time.monotonic()
            for expired in [k for k, (saved, _) in self.entries.items() if now - saved >= self.ttl]:
                del self.entries[expired]
            if key in self.entries:
                saved, data = self.entries[key]
                self.entries.move_to_end(key)
                result = deepcopy(data)
                result["cache"] = {"hit": True, "age_seconds": int(now - saved), "ttl_seconds": self.ttl}
                return result
            if now < self.blocked_until:
                remaining = math.ceil(self.blocked_until - now)
                raise WeatherError(f"Weather requests are paused after Open-Meteo's rate limit. Try again in {remaining} seconds. No new API request was sent.")
            try:
                result = deepcopy(self.loader(location))
            except WeatherRateLimit as exc:
                self.blocked_until = time.monotonic() + exc.retry_after
                raise WeatherError(f"{exc} Weather requests paused for {exc.retry_after} seconds.") from exc
            result["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
            self.entries[key] = (time.monotonic(), deepcopy(result))
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            result["cache"] = {"hit": False, "age_seconds": 0, "ttl_seconds": self.ttl}
            return result


_weather_cache = WeatherCache()


def get_cached_weather(location):
    return _weather_cache.get(location)
