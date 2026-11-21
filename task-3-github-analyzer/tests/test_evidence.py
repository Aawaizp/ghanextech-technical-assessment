import unittest
from analyzer.evidence import prepare_evidence


class EvidenceTests(unittest.TestCase):
    def select(self, path, text):
        return prepare_evidence({'evidence': [{'id': 'file-1', 'path': path, 'text': text}]}, 2)

    def test_python_prefers_complete_functions_over_imports_and_tail(self):
        text = ('import typing\n\n' * 25 +
                'def validate(value):\n    return value is not None\n\n' +
                'def normalize(value):\n    return value.strip()\n\n' +
                '# unrelated final comment\n' * 25)
        excerpts, sources = self.select('module.py', text)
        quotes = [e['quote'] for e in excerpts.values()]
        self.assertEqual(len(quotes), 2)
        self.assertTrue(all(q.startswith('def ') and 'return ' in q for q in quotes))
        self.assertTrue(all(q in text for q in quotes))
        self.assertTrue(all(not e['partial'] for e in sources[0]['available_quotes']))

    def test_documentation_skips_logo_markup(self):
        text = '### Sponsors\n\n**Documentation**: [https://example.org](https://example.org)\n\n<p align="center"><img src="logo.png"></p>\n\nA small Python library for parsing configuration.\n\nRun the tests with pytest.'
        excerpts, _ = self.select('README.md', text)
        self.assertTrue(excerpts)
        self.assertTrue(all('<img' not in e['quote'] for e in excerpts.values()))
        self.assertTrue(any('parsing configuration' in e['quote'] for e in excerpts.values()))
        self.assertTrue(all('### Sponsors' not in e['quote'] for e in excerpts.values()))

    def test_logo_only_source_is_not_citable(self):
        excerpts, sources = self.select('README.md', '<img src="logo.png">')
        self.assertFalse(excerpts)
        self.assertIn('do not cite', sources[0]['selection_note'])

    def test_workflow_selects_triggers_and_commands(self):
        text = 'name: Docs\n\non:\n  push:\n    branches: [main]\n\njobs:\n  docs:\n    steps:\n      - run: python scripts/build_docs.py\n'
        excerpts, _ = self.select('.github/workflows/docs.yml', text)
        quotes = [e['quote'] for e in excerpts.values()]
        self.assertTrue(any('on:' in q and 'branches: [main]' in q for q in quotes))
        self.assertTrue(any('run: python scripts/build_docs.py' in q for q in quotes))
        self.assertTrue(all(q in text for q in quotes))

    def test_partial_function_keeps_name_and_marks_missing_context(self):
        text = 'def process(value):\n' + '    value += 1\n' * 50
        excerpts, sources = self.select('module.py', text)
        self.assertTrue(next(iter(excerpts.values()))['quote'].startswith('def process'))
        self.assertTrue(sources[0]['available_quotes'][0]['partial'])
        self.assertTrue(sources[0]['truncated'])

    def test_no_source_execution_and_quote_budget(self):
        text = 'raise RuntimeError("Never execute repository source")\n\ndef inspect(value):\n    return value\n'
        excerpts, _ = self.select('module.py', text)
        self.assertTrue(excerpts)
        for excerpt in excerpts.values():
            self.assertIn(excerpt['quote'], text)
            self.assertTrue(12 <= len(excerpt['quote']) <= 240)

    def test_partial_prose_retains_original_subject_in_context(self):
        text = ('Run the skills installer for your coding agent. ' + 'Installation details. ' * 9 +
                'The coding agent reads the shared skills folder and follows the installed instructions. ' * 4)
        _, sources = self.select('README.md', text)
        for item in sources[0]['available_quotes']:
            if item['partial']:
                self.assertIn(item['text'], item['context'])
                self.assertIn(item['context'], text)
                self.assertLessEqual(len(item['context']), 690)

    def test_dependency_exclusion_is_not_hidden_by_comment(self):
        text = 'dependencies = [\n' + '  # Known compatibility problem; see upstream issue.\n' * 7 + '  "chartlib>=5,!=5.4.0,!=5.4.1",\n]'
        excerpts, sources = self.select('lib/pyproject.toml', text)
        self.assertTrue(any('!=5.4.0,!=5.4.1' in q['quote'] for q in excerpts.values()))
        self.assertTrue(any('Known compatibility' in q.get('context', '') for q in sources[0]['available_quotes']))
