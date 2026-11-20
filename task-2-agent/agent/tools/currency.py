"""Currency conversion using Frankfurter's daily ECB reference rates."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import math
import re

import requests


RATES_URL = "https://api.frankfurter.dev/v1/latest"
MAX_AMOUNT = 1_000_000_000


class CurrencyError(RuntimeError):
    pass


def _currency_code(value: str) -> str:
    if not isinstance(value, str):
        raise CurrencyError("Use three-letter currency codes, such as USD, INR or EUR")
    code = value.strip()
    if not re.fullmatch(r"[A-Za-z]{3}", code):
        raise CurrencyError("Use three-letter currency codes, such as USD, INR or EUR")
    return code.upper()


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    session: requests.Session | None = None,
) -> dict:
    """Convert a positive amount using the latest available reference rate.

    The returned date is the rate publication date, which may precede today.
    Results are rounded to six decimals, not currency-specific cash units.
    """
    if (
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not 0 < amount <= MAX_AMOUNT
    ):
        raise CurrencyError(f"amount must be a finite number greater than 0 and at most {MAX_AMOUNT}")
    base = _currency_code(from_currency)
    target = _currency_code(to_currency)
    if base == target:
        raise CurrencyError("The source and target currency are the same; no conversion is needed")

    http = session if session is not None else requests.Session()
    try:
        response = http.get(
            RATES_URL, params={"base": base, "symbols": target}, timeout=10
        )
        if response.status_code in (400, 404, 422):
            raise CurrencyError(f"No supported reference rate is available for {base} to {target}")
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise CurrencyError("The exchange-rate service timed out; please try again") from exc
    except requests.RequestException as exc:
        raise CurrencyError("The exchange-rate service is temporarily unavailable") from exc
    except ValueError as exc:
        raise CurrencyError("The exchange-rate service returned an invalid response") from exc
    finally:
        if session is None:
            http.close()

    try:
        if not isinstance(payload, dict) or payload.get("base") != base or payload.get("amount", 1) != 1:
            raise ValueError("Wrong base currency")
        rate_date = payload["date"]
        if not isinstance(rate_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", rate_date):
            raise ValueError("Invalid publication date")
        date.fromisoformat(rate_date)
        raw_rate = payload["rates"][target]
        if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)):
            raise ValueError("Invalid rate")
        if not math.isfinite(raw_rate) or raw_rate <= 0:
            raise ValueError("Invalid rate")
        with localcontext() as context:
            context.prec = 50
            converted = (Decimal(str(amount)) * Decimal(str(raw_rate))).quantize(Decimal("0.000001"))
        converted_amount = float(converted)
        if not math.isfinite(converted_amount):
            raise ValueError("Invalid conversion")
    except (KeyError, TypeError, ValueError, OverflowError, InvalidOperation) as exc:
        raise CurrencyError("The exchange-rate service did not return a valid dated rate for this pair") from exc

    return {
        "amount": amount,
        "from_currency": base,
        "to_currency": target,
        "exchange_rate": raw_rate,
        "converted_amount": converted_amount,
        "rate_date": rate_date,
        "rate_type": "daily_reference",
        "source": "Frankfurter / European Central Bank",
        "source_url": f"{RATES_URL}?base={base}&symbols={target}",
        "note": "Daily reference rate, not a live market quote. Rounded to six decimals; bank fees are excluded.",
    }
