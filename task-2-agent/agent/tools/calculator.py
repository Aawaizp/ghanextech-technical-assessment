"""Restricted arithmetic evaluator; never executes Python source."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable


class CalculatorError(ValueError):
    pass


_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_ARGUMENT_COUNTS = {name: (1, 1) for name in _FUNCTIONS}
_ARGUMENT_COUNTS.update({"round": (1, 2), "log": (1, 2)})


def _checked(value: int | float) -> int | float:
    """Bound intermediate values too, so a later operation cannot hide overflow."""
    if type(value) not in (int, float):
        raise CalculatorError("Only real numbers are supported")
    if abs(value) > 1e100 or not math.isfinite(value):
        raise CalculatorError("Result is outside the supported numeric range")
    return value


def _evaluate(node: ast.AST, depth: int = 0) -> int | float:
    if depth > 20:
        raise CalculatorError("Expression is too deeply nested")
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        return _checked(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _checked(_CONSTANTS[node.id])
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _checked(_UNARY[type(node.op)](_evaluate(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left, depth + 1)
        right = _evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalculatorError("Exponent magnitude cannot exceed 100")
        return _checked(_BINARY[type(node.op)](left, right))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        minimum, maximum = _ARGUMENT_COUNTS[node.func.id]
        if not minimum <= len(node.args) <= maximum:
            raise CalculatorError(
                f"{node.func.id} expects {minimum} to {maximum} argument(s)"
            )
        arguments = [_evaluate(arg, depth + 1) for arg in node.args]
        if node.func.id == "round" and len(arguments) == 2:
            digits = arguments[1]
            if type(digits) is not int or abs(digits) > 100:
                raise CalculatorError("Rounding digits must be an integer from -100 to 100")
        return _checked(_FUNCTIONS[node.func.id](*arguments))
    raise CalculatorError(f"Unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression using a strict AST allowlist."""
    if not isinstance(expression, str):
        raise CalculatorError("Expression must be a string")
    expression = expression.strip()
    if not expression:
        raise CalculatorError("Expression cannot be empty")
    if len(expression) > 200:
        raise CalculatorError("Expression cannot exceed 200 characters")
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise CalculatorError("Expression contains too many operations")
        value = _evaluate(tree)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        if isinstance(exc, CalculatorError):
            raise
        raise CalculatorError(f"Invalid calculation: {exc}") from exc
    return {"expression": expression, "result": value}
