import json
import unittest

from analyzer.export import export_report
from analyzer.evidence import source_scope, prepare_evidence
from analyzer.report import AnalysisError
from test_report import snapshot, report


class ExportTests(unittest.TestCase):
    def test_export_preserves_evidence_and_declares_validation_limits(self):
        data = snapshot()
        data.update({"commit": "abc123", "url": "https://github.com/example/project"})
        data["evidence"][0]["url"] = "https://github.com/example/project/blob/abc123/README.md"
        data["evidence"][0]["text"] += "\nUnicode example: café."
        exported = json.loads(export_report(data, report()))
        self.assertEqual(exported["snapshot"]["commit"], "abc123")
        self.assertEqual(exported["snapshot"]["evidence"], data["evidence"])
        self.assertEqual(exported["analysis"]["recommendations"], report()["recommendations"])
        self.assertFalse(exported["validation"]["interpretations_independently_verified"])

    def test_export_excludes_configuration_and_unknown_fields(self):
        data = snapshot()
        data["api_key"] = "secret-marker"
        data["progress"] = {"draft": "private-state"}
        data["evidence"][0]["authorization"] = "secret-marker"
        result = report()
        result["unexpected_configuration"] = "secret-marker"
        with self.assertRaises(AnalysisError):
            export_report(data, result)
        exported = export_report(data, report())
        self.assertNotIn("secret-marker", exported)
        self.assertNotIn("private-state", exported)

    def test_export_rejects_invented_source_quotes(self):
        result = report()
        result["strengths"][0]["evidence_quotes"][0]["quote"] = "This quotation was invented."
        with self.assertRaises(AnalysisError):
            export_report(snapshot(), result)

    def test_example_and_workflow_scope_are_explicit(self):
        self.assertIn("Test/example", source_scope("e2e_playwright/agent_api_flows.py"))
        self.assertIn("Test/example", source_scope("examples/app.py"))
        self.assertIn("visible triggers", source_scope(".github/workflows/close-issues.yml"))
        data = snapshot()
        data["evidence"][0]["path"] = "e2e_playwright/agent_api_flows.py"
        _, evidence = prepare_evidence(data, 2)
        self.assertIn("not proof of library", evidence[0]["scope"])
