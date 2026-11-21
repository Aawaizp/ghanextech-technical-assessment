import unittest
from unittest.mock import Mock, patch

from agent.tools.weather import WeatherError, WeatherRateLimit, retry_delay
from agent.tools.weather_cache import WeatherCache


class WeatherCacheTests(unittest.TestCase):
    def test_reuses_normalized_location_and_returns_independent_copy(self):
        loader = Mock(return_value={"current": {"temperature_c": 21}})
        cache = WeatherCache(loader)
        with patch("agent.tools.weather_cache.time.monotonic", return_value=100):
            first = cache.get("Solapur, India")
            first["current"]["temperature_c"] = 999
            second = cache.get(" solapur ,  india ")
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(second["current"]["temperature_c"], 21)
        self.assertTrue(second["cache"]["hit"])
        self.assertIn("fetched_at_utc", second)

    def test_expired_data_is_not_served(self):
        loader = Mock(return_value={"current": {}})
        cache = WeatherCache(loader)
        with patch("agent.tools.weather_cache.time.monotonic", return_value=100):
            cache.get("Solapur")
        with patch("agent.tools.weather_cache.time.monotonic", return_value=401):
            self.assertFalse(cache.get("Solapur")["cache"]["hit"])
        self.assertEqual(loader.call_count, 2)

    def test_cooldown_blocks_other_locations_and_recovers_after_expiry(self):
        loader = Mock(side_effect=[WeatherRateLimit("HTTP 429", 600), {"current": {}}])
        cache = WeatherCache(loader)
        with patch("agent.tools.weather_cache.time.monotonic", return_value=100):
            with self.assertRaisesRegex(WeatherError, "600 seconds"):
                cache.get("Solapur")
        with patch("agent.tools.weather_cache.time.monotonic", return_value=400):
            with self.assertRaisesRegex(WeatherError, "No new API request"):
                cache.get("Pune")
        self.assertEqual(loader.call_count, 1)
        with patch("agent.tools.weather_cache.time.monotonic", return_value=701):
            cache.get("Pune")
        self.assertEqual(loader.call_count, 2)

    def test_valid_cache_can_be_used_during_cooldown(self):
        loader = Mock(side_effect=[{"current": {}}, WeatherRateLimit("HTTP 429")])
        cache = WeatherCache(loader)
        cache.get("Solapur")
        with self.assertRaises(WeatherError):
            cache.get("Pune")
        self.assertTrue(cache.get("Solapur")["cache"]["hit"])
        self.assertEqual(loader.call_count, 2)

    def test_capacity_and_errors_are_not_cached(self):
        loader = Mock(return_value={"current": {}})
        cache = WeatherCache(loader, capacity=2)
        for location in ("A", "B", "C"):
            cache.get(location)
        self.assertEqual(len(cache.entries), 2)
        self.assertNotIn("a", cache.entries)
        failing = WeatherCache(Mock(side_effect=WeatherError("unavailable")))
        with self.assertRaises(WeatherError):
            failing.get("A")
        self.assertFalse(failing.entries)

    def test_retry_after_seconds_and_http_date(self):
        self.assertEqual(retry_delay("900"), 900)
        self.assertEqual(retry_delay(None), 300)
        self.assertEqual(retry_delay("invalid"), 300)
        self.assertGreater(retry_delay("Wed, 01 Jan 2099 00:00:00 GMT"), 300)
