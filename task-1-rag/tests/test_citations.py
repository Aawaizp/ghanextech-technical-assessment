import unittest
from modules.citations import validate_answer


class CitationTests(unittest.TestCase):
    def test_valid_references(self):
        self.assertEqual(validate_answer(" Claim [1][2]. ", 2, "Not found"), "Claim [1][2].")

    def test_refusal(self):
        self.assertEqual(validate_answer("Not found", 2, "Not found"), "Not found")

    def test_invalid_answers(self):
        for answer in ("", "No citation", "Claim [0]", "Claim [3]", "Claim 【3†L1-L4】", "Claim 【unknown】"):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                validate_answer(answer, 2, "Not found")

    def test_normalizes_known_passage_citations(self):
        self.assertEqual(
            validate_answer("Claim【1†L1-L4】【2†L2-L5】.", 2, "Not found"),
            "Claim[1][2].",
        )
        self.assertEqual(validate_answer("Claim 【1】", 1, "Not found"), "Claim [1]")


if __name__ == "__main__":
    unittest.main()
