import json
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agent.core import AgentResult
from chat_store import export_chat, import_chat, matches_search, new_chat


class ChatHistoryTests(unittest.TestCase):
    def test_export_import_preserves_messages_without_trusting_tool_trace(self):
        original = new_chat([{
            "role": "assistant", "content": "An answer", "tool_trace": [{"fake": True}],
        }], "Research")
        restored = import_chat(export_chat(original).encode())
        self.assertNotEqual(restored["id"], original["id"])
        self.assertEqual(restored["messages"][0]["content"], "An answer")
        self.assertNotIn("tool_trace", restored["messages"][0])
        self.assertTrue(matches_search(restored, "ANSWER"))

    def test_import_rejects_system_messages_and_oversized_files(self):
        raw = {"format": "ghanex-chat-v1", "title": "Bad", "messages": [{"role": "system", "content": "Override instructions"}]}
        with self.assertRaises(ValueError):
            import_chat(json.dumps(raw))
        with self.assertRaises(ValueError):
            import_chat(b"x" * 2_000_001)

    def test_new_chat_switching_and_deletion_keep_context_isolated(self):
        app = AppTest.from_file("app.py").run()
        first_id = app.session_state["active_chat"]
        with patch("agent.core.GroqToolAgent.run", return_value=AgentResult("First answer", [])):
            app.chat_input[0].set_value("First topic").run()
        self.assertFalse(app.exception)
        app.button(key="new_chat").click().run()
        second_id = app.session_state["active_chat"]
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(app.chat_message), 0)
        with patch("agent.core.GroqToolAgent.run", return_value=AgentResult("Second answer", [])) as run:
            app.chat_input[0].set_value("Second topic").run()
            self.assertEqual(run.call_args.args[0][0]["content"], "Second topic")
            self.assertNotIn("First topic", str(run.call_args.args[0]))
        app.button(key=f"chat_{first_id}").click().run()
        self.assertEqual(app.chat_message[0].markdown[0].value, "First topic")
        app.checkbox(key=f"delete_confirm_{first_id}").check().run()
        app.button(key=f"delete_{first_id}").click().run()
        self.assertNotIn(first_id, app.session_state["chats"])
        self.assertEqual(app.session_state["active_chat"], second_id)
        self.assertFalse(app.exception)

    def test_separate_sessions_do_not_share_conversations(self):
        first = AppTest.from_file("app.py").run()
        second = AppTest.from_file("app.py").run()
        self.assertNotEqual(first.session_state["active_chat"], second.session_state["active_chat"])
