import unittest

from agent.registry import execute_tool


class RegistryTests(unittest.TestCase):
    def test_unknown_tool_returns_error(self):
        self.assertEqual(
            execute_tool("delete_everything", {}),
            {"ok": False, "error": "Unknown tool: delete_everything"},
        )

    def test_tool_error_does_not_crash_agent(self):
        result = execute_tool("calculate", {"expression": "1 / 0"})
        self.assertFalse(result["ok"])
        self.assertIn("division", result["error"].lower())


if __name__ == "__main__":
    unittest.main()