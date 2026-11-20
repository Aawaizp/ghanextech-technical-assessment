"""Network failure behavior needed to diagnose hosted weather requests."""
import unittest
from unittest.mock import Mock, patch

import requests

from agent.tools.weather import GEOCODING_URL, FORECAST_URL, WeatherError, _get_json


def http_response(status, payload=None):
    response = requests.Response()
    response.status_code = status
    response._content = b'{"results": []}'
    response.url = FORECAST_URL
    return response


class WeatherNetworkTests(unittest.TestCase):
    def test_rate_limit_is_specific_and_not_retried(self):
        session = Mock()
        session.get.return_value = http_response(429)
        with self.assertRaisesRegex(WeatherError, "forecast: rate limit reached.*429"):
            _get_json(session, FORECAST_URL, {})
        self.assertEqual(session.get.call_count, 1)

    def test_access_denied_identifies_location_lookup(self):
        session = Mock()
        session.get.return_value = http_response(403)
        with self.assertRaisesRegex(WeatherError, "location lookup: access denied.*403"):
            _get_json(session, GEOCODING_URL, {})
        self.assertEqual(session.get.call_count, 1)

    def test_temporary_server_error_recovers_once(self):
        session = Mock()
        session.get.side_effect = [http_response(503), http_response(200)]
        with patch("agent.tools.weather.time.sleep"):
            self.assertEqual(_get_json(session, FORECAST_URL, {}), {"results": []})
        self.assertEqual(session.get.call_count, 2)

    def test_connection_retry_is_bounded_and_details_are_sanitized(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("private proxy details")
        with patch("agent.tools.weather.time.sleep"):
            with self.assertRaises(WeatherError) as caught:
                _get_json(session, GEOCODING_URL, {})
        self.assertEqual(session.get.call_count, 2)
        self.assertIn("connection failed", str(caught.exception))
        self.assertNotIn("private proxy", str(caught.exception))

    def test_timeout_can_recover(self):
        session = Mock()
        session.get.side_effect = [requests.Timeout("timeout"), http_response(200)]
        with patch("agent.tools.weather.time.sleep"):
            self.assertEqual(_get_json(session, FORECAST_URL, {}), {"results": []})

    def test_certificate_failure_does_not_disable_verification_or_retry(self):
        session = Mock()
        session.get.side_effect = requests.exceptions.SSLError("certificate error")
        with self.assertRaisesRegex(WeatherError, "secure connection"):
            _get_json(session, FORECAST_URL, {})
        self.assertEqual(session.get.call_count, 1)
        self.assertNotIn("verify", session.get.call_args.kwargs)

    def test_bad_request_is_not_retried(self):
        session = Mock()
        session.get.return_value = http_response(400)
        with self.assertRaisesRegex(WeatherError, "request rejected.*400"):
            _get_json(session, FORECAST_URL, {})
        self.assertEqual(session.get.call_count, 1)
