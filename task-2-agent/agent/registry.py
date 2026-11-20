"""Public tool schemas and runtime validation before dispatch."""

import json
import logging
import math
import re

from .tools.calculator import calculate
from .tools.currency import convert_currency
from .tools.knowledge import search_knowledge
from .tools.weather import get_weather

logger = logging.getLogger(__name__)


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name, "description": description,
            "parameters": {
                "type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema("calculate",
        "Evaluate an entire arithmetic expression accurately. Use explicit * for multiplication.",
        {"expression": {
            "type": "string", "minLength": 1, "maxLength": 200,
            "description": "For example '(125 * 1.18) / 4'. Supports arithmetic and math functions.",
        }},
    ),
    _schema("get_weather",
        "Fetch current weather and SEVEN daily forecasts starting today in the place's timezone. "
        "Includes tomorrow and the day after tomorrow. Use for actual weather, not general science questions.",
        {"location": {
            "type": "string", "minLength": 1, "maxLength": 160,
            "description": "City or place name. Include region or country when known, e.g. 'Solapur, India'.",
        }},
    ),
    _schema("search_knowledge",
        "Search Wikipedia for sourced background or facts needing verification. Returns up to three "
        "short article introductions; these may omit detailed statistics. Not a news or full-web search.",
        {"query": {
            "type": "string", "minLength": 1, "maxLength": 200,
            "description": "A concise topic or article search, such as 'Pune population'.",
        }},
    ),
    _schema("convert_currency",
        "Convert using the latest available daily ECB reference exchange rate via Frankfurter. "
        "Returns converted amount, rate date and source. Not an intraday quote; excludes bank fees.",
        {
            "amount": {"type": "number", "exclusiveMinimum": 0, "maximum": 1_000_000_000},
            "from_currency": {
                "type": "string", "pattern": "^[A-Za-z]{3}$",
                "description": "Three-letter source currency code, e.g. USD.",
            },
            "to_currency": {
                "type": "string", "pattern": "^[A-Za-z]{3}$",
                "description": "Three-letter destination currency code, e.g. INR.",
            },
        },
    ),
]
TOOL_FUNCTIONS = {
    "calculate": calculate, "get_weather": get_weather,
    "search_knowledge": search_knowledge, "convert_currency": convert_currency,
}
PARAMETERS = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_SCHEMAS}


def _validate(arguments: dict, schema: dict) -> None:
    """Validate the small schema subset we use; provider schemas alone are not a guard."""
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    if set(arguments) - set(schema["properties"]):
        raise ValueError("Unexpected tool parameter. Use only the parameters in the tool schema.")
    for name in schema["required"]:
        if name not in arguments:
            raise ValueError(f"Missing required parameter: {name}.")
    for name, value in arguments.items():
        spec = schema["properties"][name]
        if spec["type"] == "string":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            if len(value) < spec.get("minLength", 0) or len(value) > spec.get("maxLength", 200):
                raise ValueError(f"{name} has an invalid length.")
            if "pattern" in spec and not re.fullmatch(spec["pattern"], value):
                raise ValueError(f"{name} must be a three-letter currency code.")
        elif spec["type"] == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number.")
            if not spec["exclusiveMinimum"] < value <= spec["maximum"] or not math.isfinite(value):
                raise ValueError(f"{name} must be greater than 0 and at most {spec['maximum']}.")


def execute_tool(name: str, arguments: dict) -> dict:
    """Allow only registered functions and return errors as data for the agent."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        _validate(arguments, PARAMETERS[name])
        data = function(**arguments)
        json.dumps(data, allow_nan=False)
        return {"ok": True, "data": data}
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}
    except Exception:
        logger.exception("Unexpected failure in tool %s", name)
        return {"ok": False, "error": "The tool could not complete the request. Please retry."}
