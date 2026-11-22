# AI-Powered Adaptive Interview & Project Assessment System

An AI-powered interview preparation platform built using Python, Streamlit, and Google Gemini. This project extends an existing AI Interview Coach with repository-aware analysis and an evidence-based Project Defence workflow.

## Key Features

### Resume Analysis
- Extract information from uploaded resumes.
- Generate role-specific interview preparation.
- Match resume details with job requirements.

### AI Interview Practice
- Generate profile-based interview questions.
- Support typed and recorded answers.
- Evaluate responses and provide structured feedback.
- Generate interview progress reports.

### Evidence-Based Project Defence
- Analyze uploaded project ZIP archives.
- Inspect source files and identify implementation evidence.
- Distinguish verified implementation from unverified claims.
- Generate questions based on actual project structure.
- Evaluate answers against repository evidence.
- Identify strengths and knowledge gaps.
- Generate adaptive follow-up questions.

## Technology Stack
- Python
- Streamlit
- Google Gemini API
- Pandas
- Whisper
- Python AST and ZIP analysis
- Pytest

## Run Locally

```bash
git clone https://github.com/Arya831/AI--Interview-Coach.git
cd AI--Interview-Coach
python -m venv .venv