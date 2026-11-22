
import unittest
from io import BytesIO
from zipfile import ZipFile

from utils.repository_analyzer import (
    RepositoryAnalysisError,
    analyze_repository_zip,
)


def make_zip(files):
    """Create an in-memory ZIP from a filename-to-content dictionary."""
    buffer = BytesIO()

    with ZipFile(buffer, "w") as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)

    return BytesIO(buffer.getvalue())


class RepositoryAnalyzerTests(unittest.TestCase):

    def test_valid_repository_is_analysed(self):
        uploaded = make_zip({
            "project/app.py": (
                "import streamlit\n"
                "class InterviewApp:\n"
                "    def start(self):\n"
                "        pass\n"
            ),
            "project/README.md": "# Interview Project",
            "project/requirements.txt": "streamlit\n",
        })

        result = analyze_repository_zip(uploaded)

        self.assertGreaterEqual(result["file_count"], 3)
        self.assertGreaterEqual(result["source_file_count"], 2)
        self.assertIn(".py", result["extensions"])

    def test_python_functions_and_classes_are_detected(self):
        uploaded = make_zip({
            "app.py": (
                "class InterviewCoach:\n"
                "    def evaluate(self):\n"
                "        pass\n"
                "\n"
                "def generate_questions():\n"
                "    pass\n"
            )
        })

        result = analyze_repository_zip(uploaded)
        structure = result["source_evidence"][0]["structure"]

        self.assertIn("InterviewCoach", structure["classes"])
        self.assertIn("evaluate", structure["functions"])
        self.assertIn("generate_questions", structure["functions"])

    def test_environment_secrets_are_ignored(self):
        uploaded = make_zip({
            "app.py": "def main():\n    pass\n",
            ".env": "GEMINI_API_KEY=secret-value",
            "project/.env": "PASSWORD=secret",
        })

        result = analyze_repository_zip(uploaded)
        paths = [item["path"] for item in result["source_evidence"]]

        self.assertNotIn(".env", paths)
        self.assertNotIn("project/.env", paths)

    def test_invalid_zip_is_rejected(self):
        uploaded = BytesIO(b"this is not a zip file")

        with self.assertRaises(RepositoryAnalysisError):
            analyze_repository_zip(uploaded)

    def test_empty_zip_is_rejected(self):
        uploaded = make_zip({})

        with self.assertRaises(RepositoryAnalysisError):
            analyze_repository_zip(uploaded)

    def test_path_traversal_is_rejected(self):
        uploaded = make_zip({
            "../outside.py": "print('unsafe')",
            "app.py": "def main():\n    pass\n",
        })

        with self.assertRaises(RepositoryAnalysisError):
            analyze_repository_zip(uploaded)

    def test_unsupported_files_are_not_treated_as_source(self):
        uploaded = make_zip({
            "app.py": "def main():\n    pass\n",
            "image.png": "not really an image",
            "program.exe": "binary content",
        })

        result = analyze_repository_zip(uploaded)
        paths = [item["path"] for item in result["source_evidence"]]

        self.assertEqual(paths, ["app.py"])


if __name__ == "__main__":
    unittest.main()
