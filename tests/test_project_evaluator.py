
import unittest
from unittest.mock import patch

from utils.project_evaluator import (
    ProjectEvaluationError,
    evaluate_project_evidence,
)


SAMPLE_ANALYSIS = {
    "file_count": 2,
    "source_file_count": 2,
    "extensions": [".py"],
    "source_evidence": [
        {
            "path": "app.py",
            "content": "from flask import Flask\napp = Flask(__name__)",
            "structure": {
                "classes": [],
                "functions": [],
                "imports": ["flask"],
            },
        }
    ],
}


VALID_RESPONSE = """
{
    "project_summary": "A Flask application.",
    "detected_technologies": ["Python", "Flask"],
    "verified_components": ["Flask application instance"],
    "unverified_claims": [],
    "architecture_observations": ["A basic application entry point."],
    "strengths": ["Simple structure."],
    "improvements": ["Add error handling."],
    "defence_questions": ["How does Flask handle routing?"],
    "evidence_limitations": ["No database files supplied."],
    "overall_assessment": "A basic Flask project."
}
"""


class ProjectEvaluatorTests(unittest.TestCase):

    @patch(
        "utils.project_evaluator.generate_text",
        return_value=VALID_RESPONSE,
    )
    def test_valid_evaluation_is_returned(self, mock_generate):
        result = evaluate_project_evidence(SAMPLE_ANALYSIS)

        self.assertEqual(
            result["detected_technologies"],
            ["Python", "Flask"],
        )
        self.assertEqual(
            result["verified_components"],
            ["Flask application instance"],
        )
        mock_generate.assert_called_once()

    @patch(
        "utils.project_evaluator.generate_text",
        return_value=VALID_RESPONSE,
    )
    def test_resume_and_job_description_are_accepted(self, mock_generate):
        result = evaluate_project_evidence(
            SAMPLE_ANALYSIS,
            resume_text="Python developer",
            job_description="Looking for Python skills",
        )

        self.assertTrue(result["project_summary"])

    def test_missing_source_evidence_is_rejected(self):
        with self.assertRaises(ProjectEvaluationError):
            evaluate_project_evidence({
                "file_count": 0,
                "source_file_count": 0,
                "extensions": [],
                "source_evidence": [],
            })

    @patch(
        "utils.project_evaluator.generate_text",
        return_value="This is not JSON",
    )
    def test_invalid_json_is_rejected(self, mock_generate):
        with self.assertRaises(ProjectEvaluationError):
            evaluate_project_evidence(SAMPLE_ANALYSIS)

    @patch(
        "utils.project_evaluator.generate_text",
        return_value='{"project_summary": "Incomplete response"}',
    )
    def test_missing_required_fields_are_rejected(self, mock_generate):
        with self.assertRaises(ProjectEvaluationError):
            evaluate_project_evidence(SAMPLE_ANALYSIS)

    @patch(
        "utils.project_evaluator.generate_text",
        return_value=VALID_RESPONSE,
    )
    def test_non_list_fields_are_normalised(self, mock_generate):
        result = evaluate_project_evidence(SAMPLE_ANALYSIS)

        self.assertIsInstance(
            result["detected_technologies"],
            list,
        )
        self.assertIsInstance(
            result["defence_questions"],
            list,
        )


if __name__ == "__main__":
    unittest.main()
