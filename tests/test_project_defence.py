
import json
import unittest
from unittest.mock import patch

from utils.project_defence import (
    ProjectDefenceError,
    evaluate_project_answer,
)


class ProjectDefenceTests(unittest.TestCase):

    def setUp(self):
        self.repository_analysis = {
            "source_evidence": [
                {
                    "path": "src/app.py",
                    "content": "def calculate_total(items):\n    return sum(items)",
                    "structure": {
                        "functions": ["calculate_total"],
                        "classes": [],
                    },
                }
            ]
        }

        self.project_evaluation = {
            "project_summary": "A Python application.",
            "verified_components": [
                "calculate_total function in src/app.py"
            ],
        }

    @patch("utils.project_defence.generate_text")
    def test_valid_answer_returns_evaluation(self, mock_generate):
        mock_generate.return_value = json.dumps({
            "score": 8,
            "assessment": "The explanation is technically correct.",
            "strengths": ["Correctly explained the function."],
            "knowledge_gaps": [],
            "evidence_references": ["src/app.py"],
            "follow_up_question": None,
            "follow_up_reason": "No further clarification needed.",
            "is_complete": True,
        })

        result = evaluate_project_answer(
            question="What does calculate_total do?",
            answer="It adds the values in the items collection.",
            repository_analysis=self.repository_analysis,
            project_evaluation=self.project_evaluation,
        )

        self.assertEqual(result["score"], 8)
        self.assertIn("technically correct", result["assessment"])
        self.assertEqual(result["evidence_references"], ["src/app.py"])
        self.assertTrue(result["is_complete"])

    @patch("utils.project_defence.generate_text")
    def test_follow_up_question_is_returned(self, mock_generate):
        mock_generate.return_value = json.dumps({
            "score": 5,
            "assessment": "The answer needs more detail.",
            "strengths": ["Identified the function's purpose."],
            "knowledge_gaps": ["Did not explain error handling."],
            "evidence_references": ["src/app.py"],
            "follow_up_question": "How would you handle invalid item values?",
            "follow_up_reason": "Tests error-handling understanding.",
            "is_complete": False,
        })

        result = evaluate_project_answer(
            question="What does calculate_total do?",
            answer="It calculates the total.",
            repository_analysis=self.repository_analysis,
            project_evaluation=self.project_evaluation,
        )

        self.assertEqual(result["score"], 5)
        self.assertEqual(
            result["follow_up_question"],
            "How would you handle invalid item values?",
        )
        self.assertFalse(result["is_complete"])

    @patch("utils.project_defence.generate_text")
    def test_follow_up_is_disabled_after_limit(self, mock_generate):
        mock_generate.return_value = json.dumps({
            "score": 7,
            "assessment": "Acceptable answer.",
            "strengths": ["Explained the main idea."],
            "knowledge_gaps": [],
            "evidence_references": ["src/app.py"],
            "follow_up_question": "This question must not be used.",
            "follow_up_reason": "Extra question.",
            "is_complete": False,
        })

        result = evaluate_project_answer(
            question="Explain the function.",
            answer="It sums the items.",
            repository_analysis=self.repository_analysis,
            project_evaluation=self.project_evaluation,
            follow_up_count=2,
            max_follow_ups=2,
        )

        self.assertIsNone(result["follow_up_question"])
        self.assertTrue(result["is_complete"])

    @patch("utils.project_defence.generate_text")
    def test_invalid_json_is_rejected(self, mock_generate):
        mock_generate.return_value = "This is not valid JSON"

        with self.assertRaises(ProjectDefenceError):
            evaluate_project_answer(
                question="Explain the function.",
                answer="It sums the items.",
                repository_analysis=self.repository_analysis,
                project_evaluation=self.project_evaluation,
            )

    def test_missing_repository_evidence_is_rejected(self):
        with self.assertRaises(ProjectDefenceError):
            evaluate_project_answer(
                question="Explain the function.",
                answer="It sums the items.",
                repository_analysis={"source_evidence": []},
                project_evaluation=self.project_evaluation,
            )

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(ProjectDefenceError):
            evaluate_project_answer(
                question="Explain the function.",
                answer="   ",
                repository_analysis=self.repository_analysis,
                project_evaluation=self.project_evaluation,
            )


if __name__ == "__main__":
    unittest.main()
