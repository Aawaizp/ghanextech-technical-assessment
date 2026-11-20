import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.core import GroqToolAgent


def model_reply(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class AgentTests(unittest.TestCase):
    def test_greeting_answers_without_tools(self):
        agent = GroqToolAgent(["test-key"])
        with patch.object(agent, "_ask_model", return_value=model_reply("Hello!")):
            result = agent.run([{"role": "user", "content": "Hi"}])

        self.assertEqual(result.answer, "Hello!")
        self.assertEqual(result.tool_trace, [])

    def test_multiple_tools_then_general_final_answer(self):
        agent = GroqToolAgent(["test-key"], max_tool_rounds=2)
        replies = [
            model_reply(tool_calls=[
                tool_call("call-1", "search_knowledge", '{"query":"Mars"}')
            ]),
            model_reply(tool_calls=[
                tool_call("call-2", "search_knowledge", '{"query":"Venus"}')
            ]),
            model_reply("The available results cover Mars but not Venus."),
        ]

        with (
            patch.object(agent, "_ask_model", side_effect=replies) as ask_model,
            patch(
                "agent.core.execute_tool",
                side_effect=lambda name, args: {
                    "ok": True,
                    "data": {"query": args["query"]},
                },
            ),
        ):
            result = agent.run([
                {"role": "user", "content": "Compare Mars and Venus"}
            ])

        self.assertEqual(
            [call["arguments"]["query"] for call in result.tool_trace],
            ["Mars", "Venus"],
        )
        self.assertIn("Mars", result.answer)
        self.assertNotIn("Pune", result.answer)
        self.assertEqual(ask_model.call_args.kwargs["tool_choice"], "none")
        self.assertEqual(
            ask_model.call_args.kwargs["model"],
            "qwen/qwen3.8-27b",
        )


if __name__ == "__main__":
    unittest.main()