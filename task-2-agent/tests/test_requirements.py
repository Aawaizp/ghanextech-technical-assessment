"""Failure and navigation checks tied to the assessment requirements."""

import unittest
from unittest.mock import patch

import httpx
from groq import RateLimitError, APIConnectionError
from streamlit.testing.v1 import AppTest

from agent.core import GroqToolAgent
from chat_store import new_chat
from test_agent import model_reply, tool_call


class RequirementTests(unittest.TestCase):
    def test_rate_limit_tries_next_configured_key(self):
        request = httpx.Request("POST", "https://api.groq.com")
        error = RateLimitError("limited", response=httpx.Response(429, request=request), body={})
        with GroqToolAgent(["first", "second"]) as agent:
            with patch.object(agent.clients[0].chat.completions, "create", side_effect=error), patch.object(agent.clients[1].chat.completions, "create", return_value=model_reply("Hello")):
                result = agent.run([{"role": "user", "content": "Hi"}])
                self.assertEqual(result.answer, "Hello")
                self.assertFalse(result.is_error)

    def test_connection_error_is_safe_and_visible(self):
        with GroqToolAgent(["test"]) as agent:
            error = APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
            with patch.object(agent.clients[0].chat.completions, "create", side_effect=error):
                result = agent.run([{"role": "user", "content": "Hi"}])
                self.assertTrue(result.is_error)
                self.assertIn("connect", result.answer)

    def test_call_limit_prevents_extra_execution(self):
        calls = [tool_call(str(i), "calculate", '{"expression":"' + str(i) + '+1"}') for i in range(3)]
        with GroqToolAgent(["test"], max_tool_calls=2) as agent:
            with patch.object(agent, "_ask_model", side_effect=[model_reply(tool_calls=calls), model_reply("Partial answer")]), patch("agent.core.execute_tool", return_value={"ok": True, "data": {"result": 1}}) as execute:
                result = agent.run([{"role": "user", "content": "Calculate several things"}])
                self.assertEqual(execute.call_count, 2)
                self.assertFalse(result.tool_trace[-1]["result"]["ok"])

    def test_malformed_arguments_do_not_execute_a_tool(self):
        with GroqToolAgent(["test"], max_tool_rounds=1) as agent:
            with patch.object(agent, "_ask_model", side_effect=[model_reply(tool_calls=[tool_call("bad", "calculate", "not json")]), model_reply("Invalid request")]), patch("agent.core.execute_tool") as execute:
                result = agent.run([{"role": "user", "content": "Calculate"}])
                execute.assert_not_called()
                self.assertFalse(result.tool_trace[0]["result"]["ok"])

    def test_long_history_search_rename_and_navigation(self):
        app = AppTest.from_file("app.py").run()
        records = [new_chat([{"role": "user", "content": "A long message " * 100}], f"Conversation {i}") for i in range(30)]
        app.session_state["chats"] = {c["id"]: c for c in records}
        app.session_state["active_chat"] = records[0]["id"]
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="new_chat").disabled)
        app.text_input(key="chat_search").set_value("Conversation 29").run()
        chat_buttons = [b for b in app.button if b.key and b.key.startswith("chat_")]
        self.assertEqual(len(chat_buttons), 1)
        chat_buttons[0].click().run()
        rename = next(t for t in app.text_input if t.label == "Conversation name")
        rename.set_value("Weather notes")
        next(b for b in app.button if b.label == "Save name").click().run()
        self.assertEqual(app.session_state["chats"][records[29]["id"]]["title"], "Weather notes")
        self.assertFalse(app.exception)
