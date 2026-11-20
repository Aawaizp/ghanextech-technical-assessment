import unittest
from unittest.mock import Mock, patch

import requests

from agent.tools.currency import CurrencyError, convert_currency


class CurrencyTests(unittest.TestCase):
    def make_session(self, payload=None, status=200):
        session = Mock(spec=requests.Session)
        response = session.get.return_value
        response.status_code = status
        response.json.return_value = payload or {
            "amount": 1,
            "base": "USD",
            "date": "2026-11-20",
            "rates": {"INR": 84.25},
        }
        return session

    def test_converts_with_publication_date_and_reference_rate_label(self):
        session = self.make_session()
        result = convert_currency(100, " usd ", "inr", session=session)
        self.assertEqual(result["converted_amount"], 8425)
        self.assertEqual(result["rate_date"], "2026-11-20")
        self.assertEqual(result["rate_type"], "daily_reference")
        self.assertIn("not a live", result["note"])
        self.assertEqual(session.get.call_args.kwargs["params"], {"base": "USD", "symbols": "INR"})
        self.assertEqual(session.get.call_args.kwargs["timeout"], 10)
        session.close.assert_not_called()

    def test_rejects_invalid_amounts_without_network(self):
        session = self.make_session()
        for amount in [0, -1, True, "100", None, float("nan"), float("inf"), 1_000_000_001]:
            with self.subTest(amount=amount), self.assertRaises(CurrencyError):
                convert_currency(amount, "USD", "INR", session=session)
        session.get.assert_not_called()

    def test_rejects_invalid_currency_codes_without_network(self):
        session = self.make_session()
        for code in ["dollars", "$", "US", "USD/INR", "U1D", "uſd", None]:
            with self.subTest(code=code), self.assertRaises(CurrencyError):
                convert_currency(10, code, "INR", session=session)
            with self.subTest(target=code), self.assertRaises(CurrencyError):
                convert_currency(10, "USD", code, session=session)
        session.get.assert_not_called()

    def test_same_currency_explains_no_conversion_is_needed(self):
        session = self.make_session()
        with self.assertRaisesRegex(CurrencyError, "no conversion is needed"):
            convert_currency(10, "USD", "USD", session=session)
        session.get.assert_not_called()

    def test_unsupported_pair_does_not_invent_a_rate(self):
        for status in [400, 404, 422]:
            session = self.make_session(status=status)
            with self.subTest(status=status), self.assertRaisesRegex(CurrencyError, "No supported reference rate"):
                convert_currency(100, "USD", "ZZZ", session=session)

    def test_rejects_missing_or_invalid_rates_and_dates(self):
        for updates in [
            {"base": "EUR"},
            {"amount": 100},
            {"rates": {}},
            {"rates": {"INR": -1}},
            {"rates": {"INR": True}},
            {"rates": {"INR": float("nan")}},
            {"rates": {"INR": float("inf")}},
            {"date": "2026-02-30"},
            {"date": "20261120"},
            {"date": None},
        ]:
            payload = {"base": "USD", "date": "2026-11-20", "rates": {"INR": 84.25}, **updates}
            with self.subTest(updates=updates), self.assertRaisesRegex(CurrencyError, "valid dated rate"):
                convert_currency(100, "USD", "INR", session=self.make_session(payload))

    def test_timeouts_and_http_errors_have_safe_messages(self):
        session = self.make_session()
        session.get.side_effect = requests.Timeout("private network details")
        with self.assertRaisesRegex(CurrencyError, "timed out") as error:
            convert_currency(100, "USD", "INR", session=session)
        self.assertNotIn("private", str(error.exception))
        session.get.side_effect = None
        session.get.return_value.raise_for_status.side_effect = requests.HTTPError("private network details")
        with self.assertRaisesRegex(CurrencyError, "temporarily unavailable") as error:
            convert_currency(100, "USD", "INR", session=session)
        self.assertNotIn("private", str(error.exception))

    def test_invalid_json_is_reported_and_owned_session_is_closed(self):
        session = self.make_session()
        session.get.return_value.json.side_effect = ValueError("not json")
        with patch("agent.tools.currency.requests.Session", return_value=session):
            with self.assertRaisesRegex(CurrencyError, "invalid response"):
                convert_currency(100, "USD", "INR")
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
