
from __future__ import annotations

import ast
import io
import re
import zipfile
from pathlib import PurePosixPath


MAX_ARCHIVE_SIZE = 20 * 1024 * 1024
MAX_EXTRACTED_SIZE = 40 * 1024 * 1024
MAX_FILE_SIZE = 1 * 1024 * 1024
MAX_FILES = 500
MAX_SOURCE_FILES = 150

IGNORED_DIRS = {
    ".git", ".github", ".venv", "venv", "env",
    "__pycache__", "node_modules", "dist", "build",
    ".idea", ".vscode", ".pytest_cache", "coverage",
}

IGNORED_FILES = {
    ".env", ".env.local", ".env.production",
    "id_rsa", "id_ed25519",
}

IGNORED_SUFFIXES = {
    ".pyc", ".pyo", ".exe", ".dll", ".so", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".mp3", ".mp4", ".pdf", ".zip", ".sqlite",
    ".db", ".class", ".jar", ".lock",
}

SOURCE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".rb",
    ".html", ".css", ".sql", ".sh",
    ".yaml", ".yml", ".toml",
}

IMPORTANT_FILES = {
    "readme.md", "requirements.txt", "pyproject.toml",
    "package.json", "pom.xml", "build.gradle",
    "dockerfile", "makefile", "cargo.toml",
}


class RepositoryAnalysisError(ValueError):
    """Raised when a repository archive is unsafe or unsuitable."""


def _is_ignored(path: PurePosixPath) -> bool:
    parts = {part.lower() for part in path.parts}
    filename = path.name.lower()

    if parts.intersection(IGNORED_DIRS):
        return True

    if filename in IGNORED_FILES:
        return True

    if filename.endswith((".pem", ".key", ".p12", ".pfx")):
        return True

    if filename in {".ds_store", "thumbs.db"}:
        return True

    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True

    return False


def _python_structure(source: str) -> dict:
    """Extract Python functions, classes, and imports without execution."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "classes": [],
            "functions": [],
            "imports": [],
        }

    classes = []
    functions = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)

        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)

        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    return {
        "classes": sorted(set(classes)),
        "functions": sorted(set(functions)),
        "imports": sorted(set(imports)),
    }


def analyze_repository_zip(uploaded_file) -> dict:
    """
    Safely inspect a ZIP archive without extracting files to disk.

    Returns repository metadata, source evidence, and file coverage.
    """
    archive_bytes = uploaded_file.getvalue()

    if not archive_bytes:
        raise RepositoryAnalysisError("The uploaded ZIP file is empty.")

    if len(archive_bytes) > MAX_ARCHIVE_SIZE:
        raise RepositoryAnalysisError("ZIP file exceeds the 20 MB upload limit.")

    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except (zipfile.BadZipFile, OSError):
        raise RepositoryAnalysisError(
            "The uploaded file is not a valid ZIP archive."
        )

    with archive:
        entries = archive.infolist()

        if len(entries) > MAX_FILES:
            raise RepositoryAnalysisError(
                f"Archive contains too many entries. Maximum: {MAX_FILES}."
            )

        total_size = sum(item.file_size for item in entries)

        if total_size > MAX_EXTRACTED_SIZE:
            raise RepositoryAnalysisError(
                "Uncompressed repository exceeds the 40 MB limit."
            )

        if any(item.flag_bits & 0x1 for item in entries):
            raise RepositoryAnalysisError(
                "Password-protected ZIP archives are not supported."
            )

        files = []
        source_evidence = []
        detected_extensions = set()
        total_source_bytes = 0

        coverage = {
            "ignored_files": 0,
            "oversized_files": 0,
            "non_source_files": 0,
            "unreadable_source_files": 0,
            "source_limit_skipped": 0,
        }

        for item in entries:
            if item.is_dir():
                continue

            path = PurePosixPath(item.filename)

            # Reject absolute paths, parent traversal and Windows drive paths.
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or re.match(r"^[A-Za-z]:", item.filename)
            ):
                raise RepositoryAnalysisError(
                    "Archive contains an unsafe file path."
                )

            if _is_ignored(path):
                coverage["ignored_files"] += 1
                continue

            if item.file_size > MAX_FILE_SIZE:
                coverage["oversized_files"] += 1
                continue

            files.append(item.filename)

            suffix = path.suffix.lower()

            if suffix:
                detected_extensions.add(suffix)

            is_source = (
                suffix in SOURCE_SUFFIXES
                or path.name.lower() in IMPORTANT_FILES
            )

            if not is_source:
                coverage["non_source_files"] += 1
                continue

            if len(source_evidence) >= MAX_SOURCE_FILES:
                coverage["source_limit_skipped"] += 1
                continue

            if total_source_bytes + item.file_size > MAX_EXTRACTED_SIZE:
                coverage["source_limit_skipped"] += 1
                continue

            try:
                content = archive.read(item)
                source = content.decode("utf-8")
            except (UnicodeDecodeError, OSError, RuntimeError):
                coverage["unreadable_source_files"] += 1
                continue

            total_source_bytes += len(content)

            record = {
                "path": item.filename,
                "content": source[:MAX_FILE_SIZE],
            }

            if suffix == ".py":
                record["structure"] = _python_structure(source)

            source_evidence.append(record)

    if not files:
        raise RepositoryAnalysisError(
            "No usable project files were found in the ZIP archive."
        )

    if not source_evidence:
        raise RepositoryAnalysisError(
            "No supported source or project configuration files were found."
        )

    return {
        "file_count": len(files),
        "source_file_count": len(source_evidence),
        "extensions": sorted(detected_extensions),
        "files": sorted(files)[:MAX_FILES],
        "source_evidence": source_evidence,
        "coverage": {
            **coverage,
            "included_files": len(files),
            "analysed_source_files": len(source_evidence),
        },
    }
