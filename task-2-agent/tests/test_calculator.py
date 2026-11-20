import unittest

from agent.tools.calculator import CalculatorError, calculate


class CalculatorTests(unittest.TestCase):
    def test_calculates_expression(self):
        self.assertEqual(calculate("sqrt(81) + 2 ** 3")["result"], 17)
        self.assertAlmostEqual(calculate("log(8, 2) + sin(pi / 2)")["result"], 4)
        self.assertEqual(calculate(" round(12.345, 2) "), {"expression": "round(12.345, 2)", "result": 12.35})

    def test_rejects_unsafe_or_excessive_expressions(self):
        expressions = (
            "__import__('os').system('whoami')",
            "(1).__class__",
            "2 ** 101",
            "[1, 2]",
            "(lambda: 1)()",
            "sqrt(x=9)",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(CalculatorError):
                    calculate(expression)

    def test_reports_division_by_zero(self):
        with self.assertRaisesRegex(CalculatorError, "Invalid calculation"):
            calculate("3 / 0")

    def test_rejects_non_finite_and_oversized_intermediate_values(self):
        for expression in (
            "1e309", "-1e309", "1e309 - 1e309", "sqrt(1e309)",
            "1e101 - 1e101", "(1e100 * 10) / 10", "(-1) ** 0.5",
        ):
            with self.subTest(expression=expression), self.assertRaises(CalculatorError):
                calculate(expression)

    def test_validates_function_arguments(self):
        for expression in (
            "sqrt()", "sqrt(4, 9)", "log(2, 3, 4)", "round(1, 2.5)",
            "round(1, 101)", "sqrt(-1)", "log(0)", "True", '"1"',
        ):
            with self.subTest(expression=expression), self.assertRaises(CalculatorError):
                calculate(expression)

    def test_limits_input_size_operation_count_and_depth(self):
        cases = (("1" * 201, "200 characters"),
                 ("+".join(["1"] * 65), "too many operations"),
                 ("+" * 25 + "1", "deeply nested"))
        for expression, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(CalculatorError, message):
                calculate(expression)

    def test_invalid_input_types(self):
        for expression in (None, True, 123, [], "", "   "):
            with self.subTest(expression=expression), self.assertRaises(CalculatorError):
                calculate(expression)


if __name__ == "__main__":
    unittest.main()
