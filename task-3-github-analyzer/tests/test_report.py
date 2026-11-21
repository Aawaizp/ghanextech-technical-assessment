"""Evidence checks and bounded model recovery, without network requests."""

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
from groq import APIStatusError, RateLimitError

from analyzer.report import AnalysisError, analyze, validate_report
from analyzer.evidence import prepare_evidence


def snapshot():
    return {
        "repository": "example/project",
        "warnings": ["No tests were executed."],
        "evidence": [{
            "id": "file-1", "path": "README.md",
            "text": "A Python HTTP client.\nRun tests with pytest.\nContributions are welcome.",
        }],
    }


def report():
    finding = {
        "title": "Documented test command",
        "detail": "The README supplies a pytest command.",
        "evidence_ids": ["file-1"],
        "evidence_quotes": [{"evidence_id": "file-1", "quote": "Run tests with pytest."}],
    }
    recommendation = {
        **finding, "action": "Check that the documented test command works.",
        "validation": "Run pytest in a prepared local environment.",
    }
    return {
        "summary": "A Python HTTP client.", "tech_stack": ["Python"],
        "strengths": [copy.deepcopy(finding)], "architecture": [],
        "recommendations": [copy.deepcopy(recommendation) for _ in range(3)],
        "contribution_opportunities": [copy.deepcopy(recommendation)],
        "limitations": ["A small file sample."],
    }


def draft():
    value = report()
    for section in ("strengths", "architecture", "recommendations", "contribution_opportunities"):
        for finding in value[section]:
            for reference in finding["evidence_quotes"]:
                reference["quote"] = "excerpt-file-1-0001"
    return value


def reply(payload, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(payload)), finish_reason=finish_reason,
    )])


class ReportTests(unittest.TestCase):
    def test_rejects_default_branch_contradiction(self):
        data = snapshot()
        data['default_branch'] = 'develop'
        for claim in ('Add a trigger for the default (main) branch.', 'The default branch is main.'):
            value = report()
            value['recommendations'][0]['action'] = claim
            with self.assertRaisesRegex(AnalysisError, 'Default branch contradicts'):
                validate_report(json.dumps(value), data)

    def test_accepts_correct_branch_and_does_not_ban_other_branch_names(self):
        data = snapshot()
        data['default_branch'] = 'develop'
        value = report()
        value['recommendations'][0]['action'] = 'Compare main and develop; the default branch is develop.'
        validate_report(json.dumps(value), data)

    def test_rejects_unestablished_absence_but_allows_investigation(self):
        value = report()
        value['recommendations'][0]['detail'] = 'No test guards against this regression.'
        with self.assertRaisesRegex(AnalysisError, 'Unsupported absence'):
            validate_report(json.dumps(value), snapshot())
        value['recommendations'][0]['detail'] = 'Inspect existing tests for this documented behavior before proposing a change.'
        validate_report(json.dumps(value), snapshot())

    def test_default_branch_is_explicit_in_model_input(self):
        data = snapshot()
        data['default_branch'] = 'release/stable'
        client, run = self.run_mocked([reply(draft())], data)
        run()
        payload = json.loads(client.chat.completions.create.call_args.kwargs['messages'][1]['content'])
        self.assertEqual(payload['default_branch'], 'release/stable')

    def test_rejects_unsafe_or_premature_recommendations(self):
        for action in (
            'Add pull_request_target to the workflow.',
            'Create a test expecting an error above 4046 bytes.',
        ):
            value = report()
            value['recommendations'][0]['action'] = action
            with self.assertRaises(AnalysisError):
                validate_report(json.dumps(value), snapshot())

    def test_accepts_investigation_first_recommendation(self):
        value = report()
        value['recommendations'][0]['action'] = 'Inspect existing tests and trace the size limit enforcement path.'
        validate_report(json.dumps(value), snapshot())
        value['recommendations'][0]['action'] = 'Inspect existing agent API tests; add edge case coverage only if a gap is found.'
        validate_report(json.dumps(value), snapshot())

    def test_rejects_workflow_claim_ignoring_pull_request_trigger(self):
        data = snapshot()
        data['evidence'].append({
            'id': 'workflow', 'path': '.github/workflows/ci.yml',
            'text': 'on:\n  push:\n    branches: [develop]\n  pull_request:\n    types: [opened]',
        })
        value = report()
        finding = value['recommendations'][0]
        finding['evidence_ids'] = ['workflow']
        finding['evidence_quotes'] = [{'evidence_id':'workflow','quote':'pull_request'}]
        finding['detail'] = 'Workflow only runs on the develop branch.'
        with self.assertRaisesRegex(AnalysisError, 'pull_request is also configured'):
            validate_report(json.dumps(value), data)

    def test_exact_quote_with_whitespace_normalization(self):
        value = report()
        value["strengths"][0]["evidence_quotes"][0]["quote"] = "Run tests\nwith pytest."
        self.assertEqual(validate_report(json.dumps(value), snapshot())["summary"], value["summary"])

    def test_rejects_invented_quote(self):
        value = report()
        value["strengths"][0]["evidence_quotes"][0]["quote"] = "Tests guarantee correctness."
        with self.assertRaisesRegex(AnalysisError, "Source quote could not be verified"):
            validate_report(json.dumps(value), snapshot())

    def test_rejects_unknown_source(self):
        value = report()
        value["strengths"][0]["evidence_ids"] = ["file-999"]
        with self.assertRaisesRegex(AnalysisError, "Unknown evidence"):
            validate_report(json.dumps(value), snapshot())

    def test_each_citation_needs_its_own_quote(self):
        data = snapshot()
        data["evidence"].append({"id": "file-2", "path": "test.py", "text": "assert result == 42"})
        value = report()
        value["strengths"][0]["evidence_ids"].append("file-2")
        with self.assertRaisesRegex(AnalysisError, "Every cited source needs a quote"):
            validate_report(json.dumps(value), data)

    def test_rejects_quote_attached_to_different_source(self):
        value = report()
        value["strengths"][0]["evidence_quotes"][0]["evidence_id"] = "tree"
        with self.assertRaisesRegex(AnalysisError, "Quote does not match"):
            validate_report(json.dumps(value), snapshot())

    def test_requires_three_recommendations(self):
        value = report()
        value["recommendations"] = value["recommendations"][:2]
        with self.assertRaisesRegex(AnalysisError, "incomplete report"):
            validate_report(json.dumps(value), snapshot())

    def test_schema_feedback_identifies_field_without_echoing_input(self):
        value = report()
        value["strengths"][0]["evidence_quotes"][0]["quote"] = "PRIVATE_MODEL_TEXT" * 30
        with self.assertRaises(AnalysisError) as caught:
            validate_report(json.dumps(value), snapshot())
        message = str(caught.exception)
        self.assertIn("strengths.0.evidence_quotes.0.quote", message)
        self.assertIn("string_too_long", message)
        self.assertNotIn("PRIVATE_MODEL_TEXT", message)

    def run_mocked(self, replies, data=None):
        client = MagicMock()
        client.chat.completions.create.side_effect = replies
        factory = patch("analyzer.report.Groq")
        mocked = factory.start()
        self.addCleanup(factory.stop)
        mocked.return_value.__enter__.return_value = client
        return client, lambda: analyze(data or snapshot(), ["test-key"])

    def test_corrects_invalid_quote_once(self):
        invalid = draft()
        invalid["strengths"][0]["evidence_quotes"][0]["quote"] = "Invented source quotation."
        client, run = self.run_mocked([reply(invalid), reply(draft())])
        result = run()
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertIn("No tests were executed.", result["limitations"])
        sent = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertIn("validation_error", sent[-1]["content"])

    def test_stops_after_one_failed_correction(self):
        invalid = draft()
        invalid["strengths"][0]["evidence_quotes"][0]["quote"] = "Invented source quotation."
        client, run = self.run_mocked([reply(invalid), reply(invalid)])
        with self.assertRaisesRegex(AnalysisError, "after one"):
            run()
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_rejects_quotes_outside_model_excerpts(self):
        data = snapshot()
        data["evidence"][0]["text"] = "x" * 900 + " Run tests with pytest."
        _, run = self.run_mocked([reply(report()), reply(report())], data)
        with self.assertRaisesRegex(AnalysisError, "Unknown excerpt ID"):
            run()

    def test_output_truncation_does_not_retry_same_budget(self):
        client, run = self.run_mocked([reply(report(), "length")])
        with self.assertRaisesRegex(AnalysisError, "output allowance"):
            run()
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_oversized_request_has_actionable_error(self):
        response = httpx.Response(413, request=httpx.Request("POST", "https://api.groq.com/test"))
        error = APIStatusError("private provider details", response=response, body=None)
        client, run = self.run_mocked([error])
        with self.assertRaisesRegex(AnalysisError, "HTTP 413") as caught:
            run()
        self.assertNotIn("private provider details", str(caught.exception))
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_missing_key_fails_without_network(self):
        with patch("analyzer.report.Groq") as client:
            with self.assertRaisesRegex(AnalysisError, "Configure a Groq"):
                analyze(snapshot(), [" "])
            client.assert_not_called()

    def test_success_uses_one_request_and_restores_original_text(self):
        client, run = self.run_mocked([reply(draft())])
        result = run()
        self.assertEqual(client.chat.completions.create.call_count, 1)
        quote = result["strengths"][0]["evidence_quotes"][0]["quote"]
        self.assertEqual(quote, snapshot()["evidence"][0]["text"])

    def test_changed_evidence_invalidates_saved_report(self):
        state = {}
        client, _ = self.run_mocked([reply(draft()), reply(draft())])
        analyze(snapshot(), ["key"], state)
        changed = snapshot()
        changed["evidence"][0]["text"] += "\nAdditional documented information."
        analyze(changed, ["key"], state)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_excerpts_are_bounded_and_include_later_content(self):
        data = snapshot()
        text = "".join(f"Line {i:04d}: useful source content here.\n" for i in range(100))
        data["evidence"][0]["text"] = text
        excerpts, sources = prepare_evidence(data)
        self.assertLessEqual(len(excerpts), 5)
        self.assertTrue(sources[0]["truncated"])
        self.assertTrue(any("0099" in e["quote"] for e in excerpts.values()))
        for excerpt in excerpts.values():
            self.assertIn(excerpt["quote"], text)
            self.assertTrue(12 <= len(excerpt["quote"]) <= 240)


if __name__ == "__main__":
    unittest.main()
