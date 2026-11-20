import copy
import unittest
from unittest.mock import Mock, patch

import requests

from agent.tools.weather import FORECAST_URL, WeatherError, get_weather


PLACE = {
    "name": "Solapur", "admin1": "Maharashtra", "country": "India",
    "country_code": "IN", "latitude": 17.67, "longitude": 75.91,
}
FORECAST = {
    "timezone": "Asia/Kolkata",
    "current": {
        "time": "2026-11-20T23:45", "weather_code": 0, "temperature_2m": 21.8,
        "apparent_temperature": 20.5, "relative_humidity_2m": 44, "wind_speed_10m": 6.1,
    },
    "daily": {
        "time": [f"2026-11-{day}" for day in range(20, 27)],
        "weather_code": [3, 1, 1, 3, 2, 0, 61],
        "temperature_2m_max": [30.3, 31.6, 32.3, 32.6, 32.0, 31.0, 29.0],
        "temperature_2m_min": [17.0, 18.8, 18.2, 21.4, 20.0, 20.1, 18.0],
        "precipitation_probability_max": [0, 0, 0, 0, 10, 0, 70],
    },
}


def response(payload):
    result = Mock()
    result.json.return_value = payload
    return result


def session_for(payload=None, places=None):
    session = Mock()
    session.get.side_effect = [
        response({"results": [copy.deepcopy(PLACE)] if places is None else places}),
        response(copy.deepcopy(FORECAST) if payload is None else payload),
    ]
    return session


class WeatherTests(unittest.TestCase):
    def test_midnight_observation_does_not_use_yesterdays_forecast_date(self):
        payload = copy.deepcopy(FORECAST)
        payload['current']['time'] = '2026-11-21T00:15'
        result = get_weather('Solapur', session=session_for(payload))
        self.assertEqual(result['local_date'], '2026-11-21')
        self.assertEqual(result['tomorrow_date'], '2026-11-22')
        self.assertEqual(result['day_after_tomorrow_date'], '2026-11-23')
        self.assertEqual(result['forecast_start_date'], '2026-11-20')

    def test_always_requests_seven_days_and_preserves_local_dates(self):
        session = session_for()
        result = get_weather("Solapur", session=session)
        request = session.get.call_args_list[1]
        self.assertEqual(request.args[0], FORECAST_URL)
        self.assertEqual(request.kwargs["params"]["forecast_days"], 7)
        self.assertEqual(request.kwargs["params"]["timezone"], "auto")
        self.assertEqual(result["local_date"], "2026-11-20")
        self.assertEqual(result["timezone"], "Asia/Kolkata")
        self.assertEqual(result["forecast"][1]["date"], "2026-11-21")
        self.assertEqual(result["forecast"][1]["max_c"], 31.6)
        self.assertEqual(len(result["forecast"]), 7)
        self.assertEqual(result["current"]["temperature_c"], 21.8)
        self.assertIn("forecast_days=7", result["source_url"])
        session.close.assert_not_called()

    def test_shows_resolved_place_and_ambiguity(self):
        other = dict(PLACE, admin1="Another region")
        result = get_weather("Solapur", session=session_for(places=[PLACE, other]))
        self.assertEqual(result["location"], "Solapur, Maharashtra, India")
        self.assertEqual(result["alternative_locations"], ["Solapur, Another region, India"])
        self.assertIn("Confirm the region", result["location_note"])

    def test_country_qualifier_selects_matching_location(self):
        other = dict(PLACE, country="Other country", country_code="XX")
        session = session_for(places=[other, PLACE])
        result = get_weather("Solapur, India", session=session)
        self.assertEqual(result["location"], "Solapur, Maharashtra, India")
        self.assertEqual(session.get.call_args_list[0].kwargs["params"]["name"], "Solapur")
        self.assertEqual(result["alternative_locations"], [])

    def test_no_location_is_a_clear_error(self):
        session = session_for(places=[])
        with self.assertRaisesRegex(WeatherError, "No location found"):
            get_weather("Zzzzxxyy123", session=session)
        self.assertEqual(session.get.call_count, 1)

    def test_rejects_invalid_arguments_before_requesting(self):
        for location in (None, 42, [], "", " ", ",,", "a" * 161):
            session = Mock()
            with self.subTest(location=location), self.assertRaises(WeatherError):
                get_weather(location, session=session)
            session.get.assert_not_called()

    def test_rejects_incomplete_or_mismatched_forecast_arrays(self):
        for field in FORECAST["daily"]:
            payload = copy.deepcopy(FORECAST)
            payload["daily"][field].pop()
            with self.subTest(field=field), self.assertRaisesRegex(WeatherError, "incomplete"):
                get_weather("Solapur", session=session_for(payload))

    def test_rejects_bad_payloads_and_nonfinite_values(self):
        payloads = [[], {"error": True}, {"current": []}, copy.deepcopy(FORECAST), copy.deepcopy(FORECAST)]
        payloads[-2]["current"]["temperature_2m"] = float("nan")
        payloads[-1]["daily"]["precipitation_probability_max"][0] = 101
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(WeatherError):
                get_weather("Solapur", session=session_for(payload))

    def test_unknown_measurements_remain_unavailable(self):
        payload = copy.deepcopy(FORECAST)
        payload["current"]["temperature_2m"] = None
        payload["daily"]["weather_code"][0] = None
        payload["daily"]["precipitation_probability_max"][0] = None
        result = get_weather("Solapur", session=session_for(payload))
        self.assertIsNone(result["current"]["temperature_c"])
        self.assertEqual(result["forecast"][0]["condition"], "unavailable")
        self.assertIsNone(result["forecast"][0]["precipitation_probability_percent"])

    def test_rejects_missing_or_invalid_time_and_coordinates(self):
        for bad_value in (True, None, float("inf"), "17.67", 91):
            with self.subTest(latitude=bad_value), self.assertRaises(WeatherError):
                get_weather("Solapur", session=session_for(places=[dict(PLACE, latitude=bad_value)]))
        payload = copy.deepcopy(FORECAST)
        payload["daily"]["time"][1] = "2026-11-23"
        with self.assertRaisesRegex(WeatherError, "inconsistent forecast dates"):
            get_weather("Solapur", session=session_for(payload))

    def test_closes_owned_session_on_success_and_failure(self):
        session = session_for()
        with patch("agent.tools.weather.requests.Session", return_value=session):
            get_weather("Solapur")
        session.close.assert_called_once()
        session = Mock()
        session.get.side_effect = requests.Timeout("private implementation detail")
        with patch("agent.tools.weather.requests.Session", return_value=session):
            with self.assertRaisesRegex(WeatherError, "timed out"):
                get_weather("Solapur")
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
