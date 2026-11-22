import html
import uuid

import pandas as pd
import streamlit as st

from utils.communication import analyze_communication
from utils.database import get_history, initialize_database, save_answer
from utils.evaluator import evaluate_answer
from utils.gemini_client import has_api_key
from utils.pdf_reader import extract_text
from utils.repository_analyzer import analyze_repository_zip
from utils.project_evaluator import evaluate_project_evidence
from utils.project_defence import (
    evaluate_project_answer,
    ProjectDefenceError,
)
from utils.question_generator import generate_interview_questions
from utils.report import build_report, report_as_markdown
from utils.speech import transcribe_audio
from utils.styles import (
    inject_global_styles,
    render_accessibility_control,
    render_landing_hero,
    render_feature_section,
    render_product_nav,
    render_workspace_header,
)
from utils.summarizer import summarize_resume


st.set_page_config(
    page_title="AI Interview Coach",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
initialize_database()
inject_global_styles()

SAMPLE_RESUME = """
Jordan Lee
Computer Science Student

Skills: Python, SQL, Git, data visualization, REST APIs, teamwork

Experience:
- Coordinated a four-person capstone team and delivered weekly progress updates.
- Built a Python dashboard that reduced manual reporting time for a student club.
- Presented project findings to faculty and incorporated stakeholder feedback.

Projects:
- Interview preparation assistant using Streamlit and a generative AI API.
- Campus event analytics dashboard using Python, pandas, and SQL.

Education:
BSc Computer Science, expected 2027
""".strip()

SAMPLE_JOB_DESCRIPTION = """
Junior Project Coordinator

Support project planning, track milestones, maintain documentation, and communicate
updates to stakeholders. The ideal candidate is organized, comfortable working with
cross-functional teams, and familiar with Agile practices, risk tracking, Jira, and
data-driven reporting. Strong written and verbal communication is required.
""".strip()

def new_interview_session(keep_summary=True):
    """Clear one interview's answers while optionally keeping the resume summary."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.interview_questions = []
    st.session_state.question_index = 0
    st.session_state.evaluations = {}
    st.session_state.audio_durations = {}
    st.session_state.answer_sources = {}
    if not keep_summary:
        st.session_state.resume_summary = None
        st.session_state.resume_text = None
        st.session_state.job_description_text = None


def load_sample_profile():
    """Populate a judge-friendly candidate and vacancy without making API calls."""
    st.session_state.demo_resume_text = SAMPLE_RESUME
    st.session_state.target_role = "Junior Project Coordinator"
    st.session_state.experience_level = "Entry level"
    st.session_state.job_description_input = SAMPLE_JOB_DESCRIPTION
    new_interview_session(keep_summary=False)


def clear_sample_profile():
    """Return setup to a clean state so the user can upload their own resume."""
    st.session_state.demo_resume_text = None
    st.session_state.target_role = ""
    st.session_state.experience_level = "Student / Intern"
    st.session_state.job_description_input = ""
    new_interview_session(keep_summary=False)


defaults = {
    "resume_summary": None,
    "resume_text": None,
    "job_description_text": None,
    "job_description_input": "",
    "demo_resume_text": None,
    "target_role": "",
    "experience_level": "Student / Intern",
    "interview_questions": [],
    "question_index": 0,
    "evaluations": {},
    "audio_durations": {},
    "answer_sources": {},
    "session_id": str(uuid.uuid4()),
    "save_history": False,
    "accessibility_mode": False,
    "project_analysis": None,
    "project_evaluation": None,
    "project_filename": None,
    "project_defence_questions": [],
    "project_defence_history": [],
    "project_defence_current_question": "",
    "project_defence_base_index": 0,
    "project_defence_follow_up_count": 0,
    "project_defence_last_feedback": None,
    "project_defence_active": False,
    "project_defence_completed": False,
}
for state_key, default_value in defaults.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


plan_ready = bool(st.session_state.resume_summary)
render_product_nav()
render_accessibility_control()

if not plan_ready:
    render_landing_hero()
else:
    render_workspace_header(
        html.escape(st.session_state.target_role),
        html.escape(st.session_state.experience_level),
    )


def render_setup_fields(compact=False):
    """Render setup inline on landing and in a popover inside the workspace."""
    if compact:
        setup_surface = st.popover(
            "Edit interview setup",
            icon=":material/tune:",
            width="stretch",
        )
    else:
        setup_surface = st.container(border=True)

    with setup_surface:
        if not compact:
            st.markdown(
                """
                <div class="setup-heading">
                    <h2>Create your practice plan</h2>
                    <p>Add a resume and choose the interview you want to prepare for.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if compact:
            candidate_area = st.container()
            target_area = st.container()
        else:
            candidate_area, target_area = st.columns(2, gap="large")

        with candidate_area:
            if st.session_state.demo_resume_text:
                uploaded_resume = None
                st.success("Sample resume selected")
                st.caption(
                    "Jordan Lee Â· Computer Science student Â· Project coordination"
                )
                st.button(
                    "Use my own resume instead",
                    icon=":material/upload_file:",
                    width="stretch",
                    on_click=clear_sample_profile,
                    key=f"clear_sample_{'compact' if compact else 'full'}",
                )
            else:
                uploaded_resume = st.file_uploader(
                    "resume",
                    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                    key="resume_file",
                    help=(
                        "Use a PDF, Word document, UTF-8 text file, or clear image "
                        "(maximum 8 MB)."
                    ),
                )
                st.button(
                    "Use a sample resume",
                    icon=":material/person_play:",
                    width="stretch",
                    on_click=load_sample_profile,
                    key=f"sample_profile_{'compact' if compact else 'full'}",
                )

        with target_area:
            st.text_input(
                "Target role",
                key="target_role",
                placeholder="e.g. Junior Data Analyst",
            )
            st.selectbox(
                "Experience level",
                ("Student / Intern", "Entry level", "Mid level", "Senior"),
                key="experience_level",
            )

        with st.expander("Add a job description", expanded=False):
            st.text_area(
                "Paste the job listing",
                key="job_description_input",
                placeholder="Paste responsibilities, skills, and qualificationsâ€¦",
                height=150,
            )
            uploaded_job = st.file_uploader(
                "Or upload the job listing",
                type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                key="job_description_file",
                help="Pasted text takes priority if both are provided.",
            )

        st.toggle(
            "Save score history on this device",
            key="save_history",
            help="When disabled, scores are kept only in the current session.",
        )

        resume_is_available = (
            uploaded_resume is not None
            or bool(st.session_state.demo_resume_text)
        )
        build_clicked = st.button(
            "Build my interview plan" if not compact else "Update interview plan",
            type="primary",
            icon=":material/arrow_forward:",
            width="stretch",
            disabled=(
                not resume_is_available
                or not st.session_state.target_role.strip()
                or not has_api_key()
            ),
            key=f"build_plan_{'compact' if compact else 'full'}",
        )

        if not resume_is_available:
            st.caption("Upload a resume or use the sample to continue.")
        elif not st.session_state.target_role.strip():
            st.caption("Enter a target role to continue.")
        elif not has_api_key():
            st.caption("Gemini is not configured on this server.")

    return uploaded_resume, uploaded_job, build_clicked


questions_ready = bool(st.session_state.interview_questions)
report_ready = bool(
    questions_ready
    and len(st.session_state.evaluations)
    == len(st.session_state.interview_questions)
)
if not plan_ready:
    step_classes = ["active", "", ""]
elif not report_ready:
    step_classes = ["done", "active", ""]
else:
    step_classes = ["done", "done", "active"]
step_current = [
    ' aria-current="step"' if step_class == "active" else ""
    for step_class in step_classes
]

st.markdown(
    f"""
    <div class="workflow" aria-label="Interview practice progress">
      <div class="workflow-step {step_classes[0]}"{step_current[0]}>
        <span class="workflow-marker"><span class="step-number">1</span><span class="step-check">âœ“</span></span>
        <span class="workflow-copy"><small>Step 1 of 3</small><span>Build your plan</span></span>
      </div>
      <div class="workflow-step {step_classes[1]}"{step_current[1]}>
        <span class="workflow-marker"><span class="step-number">2</span><span class="step-check">âœ“</span></span>
        <span class="workflow-copy"><small>Step 2 of 3</small><span>Practise answers</span></span>
      </div>
      <div class="workflow-step {step_classes[2]}"{step_current[2]}>
        <span class="workflow-marker"><span class="step-number">3</span><span class="step-check">âœ“</span></span>
        <span class="workflow-copy"><small>Step 3 of 3</small><span>Review progress</span></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if plan_ready:
    setup_column, restart_column, progress_column = st.columns(
        [1, 1, 1.8],
        vertical_alignment="center",
    )
    with setup_column:
        uploaded_file, job_description_file, analyze_clicked = render_setup_fields(
            compact=True
        )
    with restart_column:
        with st.popover(
            "Start a new attempt",
            icon=":material/refresh:",
            width="stretch",
        ):
            st.markdown("**Clear this attempt and start again?**")
            st.caption(
                "Your answers, feedback, and report will be cleared. "
                "Your resume, target role, and interview questions will stay."
            )
            if st.button(
                "Clear answers and restart",
                type="primary",
                width="stretch",
                key="confirm_new_attempt",
            ):
                new_interview_session()
                st.rerun()
    with progress_column:
        completed_answers = len(st.session_state.evaluations)
        total_answers = len(st.session_state.interview_questions) or 5
        st.caption(
            f"{completed_answers} of {total_answers} answers reviewed in this session"
        )
else:
    uploaded_file, job_description_file, analyze_clicked = render_setup_fields(
        compact=False
    )
    render_feature_section()

if analyze_clicked:
    with st.spinner("Gemini is analysing your resume..."):
        try:
            resume_text = (
                extract_text(uploaded_file)
                if uploaded_file is not None
                else st.session_state.demo_resume_text
            )
            job_description_text = st.session_state.job_description_input.strip()
            if not job_description_text and job_description_file is not None:
                job_description_text = extract_text(
                    job_description_file,
                    document_kind="job description",
                )
            st.session_state.resume_text = resume_text
            st.session_state.job_description_text = job_description_text or None
            st.session_state.resume_summary = summarize_resume(
                resume_text,
                st.session_state.target_role,
                job_description_text,
            )
            new_interview_session()
            st.rerun()
        except Exception as error:
            st.error("We could not analyse this resume.")
            st.caption(f"Technical detail: {type(error).__name__}: {error}")

if not st.session_state.resume_summary:
    st.stop()

overview_tab, practice_tab, report_tab, project_tab = st.tabs(
    [
        ":material/description: Resume overview",
        ":material/record_voice_over: Interview practice",
        ":material/monitoring: Report and progress",
        ":material/code: Project defence",
    ]
)

with overview_tab:
    st.subheader("Candidate and role briefing")
    st.caption(
        f"Built for {st.session_state.target_role} Â· "
        f"{st.session_state.experience_level}"
    )
    st.markdown(st.session_state.resume_summary)
    if st.session_state.job_description_text:
        with st.expander("Job description used for this interview"):
            st.text_area(
                "Job description",
                st.session_state.job_description_text,
                height=260,
                disabled=True,
            )
    with st.expander("Check extracted resume text"):
        st.text_area(
            "Extracted text",
            st.session_state.resume_text,
            height=320,
            disabled=True,
        )

with practice_tab:
    st.subheader("Practice interview")
    st.caption(
        "Answer naturally by text or voice. Your coach will help with content, "
        "structure, and communication."
    )

    if not st.session_state.interview_questions:
        st.markdown(
            """
            <div class="empty-state">
                <h3>Your interview is ready to generate</h3>
                <p>
                    Create five questions grounded in your resume, experience level,
                    target role, and job description.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Generate 5 interview questions",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        ):
            with st.spinner("Gemini is creating your questions..."):
                try:
                    st.session_state.interview_questions = generate_interview_questions(
                        st.session_state.resume_summary,
                        st.session_state.target_role,
                        st.session_state.experience_level,
                        st.session_state.job_description_text or "",
                    )
                    st.session_state.question_index = 0
                    st.session_state.evaluations = {}
                    st.session_state.audio_durations = {}
                    st.session_state.answer_sources = {}
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
                except Exception as error:
                    st.error("Gemini could not generate questions. Please try again.")
                    st.caption(f"Technical detail: {type(error).__name__}: {error}")

    if st.session_state.interview_questions:
        questions = st.session_state.interview_questions
        question_index = st.session_state.question_index
        question = questions[question_index]
        answer_key = f"answer_{st.session_state.session_id}_{question_index}"
        mode_key = f"mode_{st.session_state.session_id}_{question_index}"

        st.progress((question_index + 1) / len(questions))
        st.markdown(
            f"""
            <div class="question-meta">
                <span>Question {question_index + 1} of {len(questions)}</span>
                <span>{len(st.session_state.evaluations)} reviewed</span>
            </div>
            <div class="question-card">{html.escape(question)}</div>
            """,
            unsafe_allow_html=True,
        )

        answer_mode = st.segmented_control(
            "How would you like to answer?",
            ("Type your answer", "Record your answer"),
            default="Type your answer",
            required=True,
            width="stretch",
            key=mode_key,
        )

        if answer_mode == "Record your answer":
            recorded_audio = st.audio_input("Record your answer")
            if recorded_audio:
                st.audio(recorded_audio)
                if st.button(
                    "Transcribe recording",
                    icon=":material/transcribe:",
                    width="stretch",
                ):
                    with st.spinner("Whisper is transcribing your recording..."):
                        try:
                            transcript, duration = transcribe_audio(recorded_audio)
                            st.session_state[answer_key] = transcript
                            st.session_state.audio_durations[question_index] = duration
                            st.session_state.answer_sources[question_index] = "Recorded"
                            st.rerun()
                        except Exception as error:
                            st.error("We could not transcribe this recording.")
                            st.caption(
                                "Whisper downloads its speech model the first time it is used. "
                                f"Technical detail: {type(error).__name__}: {error}"
                            )

        answer = st.text_area(
            "Your answer",
            placeholder="Type your answer, or transcribe a recording above...",
            height=180,
            key=answer_key,
        )

        if st.button(
            "Get AI feedback",
            type="primary",
            icon=":material/psychology:",
            width="stretch",
        ):
            if not answer.strip():
                st.warning("Write or transcribe an answer before requesting feedback.")
            else:
                with st.spinner("Gemini is reviewing your answer..."):
                    try:
                        communication = analyze_communication(
                            answer,
                            st.session_state.audio_durations.get(question_index),
                        )
                        feedback = evaluate_answer(question, answer)
                        record = {
                            "session_id": st.session_state.session_id,
                            "question_index": question_index,
                            "question": question,
                            "answer": answer,
                            "answer_source": st.session_state.answer_sources.get(question_index, "Typed"),
                            "communication": communication,
                            "feedback": feedback,
                        }
                        st.session_state.evaluations[question_index] = record
                        if st.session_state.save_history:
                            save_answer(record)
                        st.rerun()
                    except Exception as error:
                        st.error("Gemini could not evaluate the answer. Please try again.")
                        st.caption(f"Technical detail: {type(error).__name__}: {error}")

        evaluation = st.session_state.evaluations.get(question_index)
        if evaluation:
            feedback = evaluation["feedback"]
            communication = evaluation["communication"]

            score_column, communication_column, fillers_column = st.columns(3)
            score_column.metric("Answer score", f"{feedback['score']}/10")
            communication_column.metric(
                "Communication", f"{communication['communication_score']}/10"
            )
            fillers_column.metric("Filler words", communication["total_fillers"])

            with st.expander("AI Feedback", expanded=True):
                st.markdown("#### What went well")
                st.markdown("\n".join(f"- {item}" for item in feedback["strengths"]))
                st.markdown("#### Improve next time")
                st.markdown("\n".join(f"- {item}" for item in feedback["improvements"]))
                st.markdown("#### Better answer structure")
                st.markdown("\n".join(f"- {item}" for item in feedback["better_structure"]))

            with st.expander("Communication details"):
                st.write(f"Words used: {communication['word_count']}")
                if communication["words_per_minute"]:
                    st.write(f"Speaking speed: {communication['words_per_minute']} words per minute")
                if communication["filler_counts"]:
                    st.write("Filler words found:")
                    st.json(communication["filler_counts"])
                else:
                    st.success("No common filler words were detected.")

        previous_column, next_column, fresh_questions_column = st.columns(3)
        with previous_column:
            if st.button(
                "Previous",
                icon=":material/arrow_back:",
                disabled=question_index == 0,
                width="stretch",
            ):
                st.session_state.question_index -= 1
                st.rerun()
        with next_column:
            if st.button(
                "Next",
                icon=":material/arrow_forward:",
                disabled=question_index == len(questions) - 1,
                width="stretch",
            ):
                st.session_state.question_index += 1
                st.rerun()
        with fresh_questions_column:
            if st.button(
                "New questions",
                icon=":material/refresh:",
                width="stretch",
            ):
                new_interview_session()
                st.rerun()

with report_tab:
    st.subheader("Interview report")
    st.caption(
        "A concise view of your answer quality, communication, and strongest next steps."
    )
    evaluations = st.session_state.evaluations

    if st.session_state.interview_questions and len(evaluations) == len(
        st.session_state.interview_questions
    ):
        report = build_report(evaluations)
        score_column, communication_column, fillers_column = st.columns(3)
        score_column.metric("Average answer score", f"{report['average_score']}/10")
        communication_column.metric(
            "Average communication", f"{report['average_communication']}/10"
        )
        fillers_column.metric("Total filler words", report["total_fillers"])

        st.markdown("#### Best answer")
        st.write(report["best_question"])
        st.markdown("#### Main improvement area")
        st.write(report["improvement_question"])
        st.markdown("#### Recommendations")
        st.markdown("\n".join(f"- {item}" for item in report["recommendations"]))
        st.download_button(
            "Download report",
            data=report_as_markdown(report),
            file_name="interview_report.md",
            mime="text/markdown",
            icon=":material/download:",
        )
    else:
        total_questions = len(st.session_state.interview_questions) or 5
        st.markdown(
            f"""
            <div class="empty-state">
                <h3>Your report is taking shape</h3>
                <p>
                    Review all {total_questions} answers to unlock your complete
                    interview report. You have finished {len(evaluations)} so far.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Current session history")
    history = (
        get_history(st.session_state.session_id)
        if st.session_state.save_history
        else []
    )

    if history:
        history_frame = pd.DataFrame(history)
        st.line_chart(history_frame[["score", "communication_score"]])
        st.dataframe(
            history_frame[["created_at", "score", "communication_score", "filler_count"]],
            hide_index=True,
            width="stretch",
        )
    else:
        if st.session_state.save_history:
            st.caption(
                "Your saved answer scores will appear here after your first evaluation."
            )
        else:
            st.caption(
                "Enable local score history in the interview setup to persist this "
                "session's score trend."
            )


with project_tab:
    st.markdown(
        """
        <style>
        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) {
            padding-top: 1.4rem;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) h3,
        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) h4 {
            letter-spacing: -0.035em;
            font-weight: 700;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(20, 184, 166, 0.10), rgba(20, 184, 166, 0.025));
            border: 1px solid rgba(45, 212, 191, 0.22);
            border-radius: 15px;
            padding: 17px 18px;
            min-height: 108px;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) [data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            font-weight: 600;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 750;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) [data-testid="stFileUploader"] section {
            border: 1px dashed rgba(45, 212, 191, 0.55);
            border-radius: 16px;
            background: rgba(20, 184, 166, 0.045);
            padding: 1.2rem;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) [data-testid="stAlert"] {
            border-radius: 13px;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 15px;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) .stButton button,
        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) .stDownloadButton button {
            border-radius: 11px;
            font-weight: 650;
            min-height: 2.8rem;
        }

        div[data-testid="stTabPanel"]:has([data-testid="stFileUploader"]) hr {
            margin-top: 1.7rem;
            margin-bottom: 1.7rem;
            border-color: rgba(148, 163, 184, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("AI Project Defence Assessment")
    st.caption(
        "Analyse your repository, verify implementation evidence, "
        "and prepare for technical project defence."
    )

    st.info(
        "The analyser inspects supported source files without executing "
        "the uploaded code."
    )

    project_zip = st.file_uploader(
        "Upload your project repository (.zip)",
        type=["zip"],
        key="project_zip_file",
        help="Maximum upload size: 20 MB.",
    )

    analyse_project_clicked = st.button(
        "Analyse my project",
        type="primary",
        icon=":material/analytics:",
        disabled=(project_zip is None or not has_api_key()),
        width="stretch",
    )

    if not has_api_key():
        st.warning("Configure Gemini before analysing a project.")

    if analyse_project_clicked and project_zip is not None:
        st.session_state.project_analysis = None
        st.session_state.project_evaluation = None
        st.session_state.project_filename = project_zip.name
        st.session_state.project_defence_questions = []
        st.session_state.project_defence_history = []
        st.session_state.project_defence_current_question = ""
        st.session_state.project_defence_base_index = 0
        st.session_state.project_defence_follow_up_count = 0
        st.session_state.project_defence_last_feedback = None
        st.session_state.project_defence_active = False
        st.session_state.project_defence_completed = False

        try:
            with st.spinner("Inspecting repository files..."):
                analysis = analyze_repository_zip(project_zip)

            st.session_state.project_analysis = analysis

            with st.spinner("Gemini is evaluating your project evidence..."):
                evaluation = evaluate_project_evidence(
                    analysis,
                    resume_text=st.session_state.resume_text or "",
                    job_description=(
                        st.session_state.job_description_text or ""
                    ),
                )

            st.session_state.project_evaluation = evaluation
            st.success("Project analysis completed successfully.")

        except Exception as error:
            st.error("We could not complete the project analysis.")
            st.caption(
                f"Technical detail: {type(error).__name__}: {error}"
            )

    analysis = st.session_state.project_analysis
    evaluation = st.session_state.project_evaluation

    if analysis:
        st.divider()
        st.markdown("#### Repository overview")

        file_column, source_column = st.columns(2)
        file_column.metric("Files detected", analysis["file_count"])
        source_column.metric(
            "Source files analysed",
            analysis["source_file_count"],
        )

        st.write(
            "**File types detected:**",
            ", ".join(analysis["extensions"]) or "Not detected",
        )

        coverage = analysis.get("coverage", {})

        st.markdown("#### Repository analysis coverage")

        coverage_items = [
            ("Included files", coverage.get("included_files", 0)),
            ("Analysed source files", coverage.get("analysed_source_files", 0)),
            ("Ignored files", coverage.get("ignored_files", 0)),
            ("Oversized files", coverage.get("oversized_files", 0)),
            ("Non-source files", coverage.get("non_source_files", 0)),
            ("Unreadable source files", coverage.get("unreadable_source_files", 0)),
            ("Skipped due to source limit", coverage.get("source_limit_skipped", 0)),
        ]

        coverage_columns = st.columns(3)
        for index, (label, value) in enumerate(coverage_items):
            coverage_columns[index % 3].metric(label, value)

        with st.expander("View repository file list"):
            st.code("\n".join(analysis["files"]))

    if evaluation:
        st.divider()
        st.markdown("#### AI project assessment")

        st.markdown("**Implementation status**")
        status = evaluation.get(
            "implementation_status",
            "Implementation evidence insufficient",
        )
        st.info(status)

        st.markdown("**Project summary**")
        st.write(evaluation["project_summary"])

        st.markdown("**Detected technologies**")
        st.write(
            ", ".join(evaluation["detected_technologies"])
            or "None detected"
        )

        verified_col, unverified_col = st.columns(2, gap="medium")

        with verified_col.container(border=True):
            st.markdown("##### Verified components")
            st.caption("Implementation supported by repository evidence")
            if evaluation["verified_components"]:
                for item in evaluation["verified_components"]:
                    st.markdown(f"- {item}")
            else:
                st.caption("No components were verified.")

        with unverified_col.container(border=True):
            st.markdown("##### Unverified claims")
            st.caption("Claims that need additional implementation evidence")
            if evaluation["unverified_claims"]:
                for item in evaluation["unverified_claims"]:
                    st.markdown(f"- {item}")
            else:
                st.caption("No unverified claims were returned.")

        st.markdown("#### Technical review")

        architecture_col, strengths_col = st.columns(2, gap="medium")

        with architecture_col.container(border=True):
            st.markdown("##### Architecture observations")
            if evaluation["architecture_observations"]:
                for item in evaluation["architecture_observations"]:
                    st.markdown(f"- {item}")
            else:
                st.caption("No architecture observations available.")

        with strengths_col.container(border=True):
            st.markdown("##### Technical strengths")
            if evaluation["strengths"]:
                for item in evaluation["strengths"]:
                    st.markdown(f"- {item}")
            else:
                st.caption("No technical strengths returned.")

        with st.container(border=True):
            st.markdown("##### Recommended improvements")
            st.caption("Suggested next steps based on the repository review")
            if evaluation["improvements"]:
                for index, item in enumerate(evaluation["improvements"], start=1):
                    st.markdown(f"**{index}.** {item}")
            else:
                st.caption("No additional improvements suggested.")


        if evaluation["verified_components"]:
            for item in evaluation["verified_components"]:
                st.markdown(f"- {item}")
        else:
            st.caption("No components were verified.")

        st.markdown("**Unverified claims**")
        if evaluation["unverified_claims"]:
            for item in evaluation["unverified_claims"]:
                st.markdown(f"- {item}")
        else:
            st.caption("No unverified claims were returned.")

        st.markdown("**Architecture observations**")
        for item in evaluation["architecture_observations"]:
            st.markdown(f"- {item}")

        st.markdown("**Technical strengths**")
        for item in evaluation["strengths"]:
            st.markdown(f"- {item}")

        st.markdown("**Suggested improvements**")
        for item in evaluation["improvements"]:
            st.markdown(f"- {item}")

        st.divider()
        st.markdown("#### Repository-grounded defence questions")

        repository_questions = evaluation.get(
            "repository_defence_questions",
            evaluation.get("defence_questions", []),
        )

        if repository_questions:
            for index, question in enumerate(repository_questions, start=1):
                st.markdown(f"**Q{index}.** {question}")
        else:
            st.caption("No repository-grounded questions were generated.")

        st.markdown("#### Resume-based defence questions")

        resume_questions = evaluation.get("resume_based_questions", [])

        if resume_questions:
            for index, question in enumerate(resume_questions, start=1):
                st.markdown(f"**Q{index}.** {question}")
        else:
            st.caption("No additional resume-based questions were generated.")

        st.markdown("**Evidence limitations**")
        for item in evaluation["evidence_limitations"]:
            st.markdown(f"- {item}")

        st.markdown("**Overall assessment**")
        st.write(evaluation["overall_assessment"])

        st.download_button(
            "Download project assessment",
            data=__import__("json").dumps(
                evaluation,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="project_defence_assessment.json",
            mime="application/json",
            icon=":material/download:",
        )

        st.divider()
        st.markdown("### Interactive project defence")
        st.caption(
            "Answer repository-specific questions. The AI will assess "
            "your explanation against the uploaded source code and "
            "adapt its follow-up questions."
        )

        if not st.session_state.project_defence_questions:
            available_questions = (
                evaluation.get("repository_defence_questions", [])
                or evaluation.get("defence_questions", [])
            )

            if available_questions:
                if st.button(
                    "Start interactive project defence",
                    type="primary",
                    icon=":material/play_arrow:",
                    key="start_project_defence",
                ):
                    st.session_state.project_defence_questions = (
                        available_questions[:3]
                    )
                    st.session_state.project_defence_history = []
                    st.session_state.project_defence_base_index = 0
                    st.session_state.project_defence_follow_up_count = 0
                    st.session_state.project_defence_last_feedback = None
                    st.session_state.project_defence_active = True
                    st.session_state.project_defence_completed = False
                    st.session_state.project_defence_current_question = (
                        available_questions[0]
                    )
                    st.rerun()
            else:
                st.warning(
                    "No repository-grounded questions are available "
                    "for an interactive defence."
                )

        if st.session_state.project_defence_active:
            questions = st.session_state.project_defence_questions
            history = st.session_state.project_defence_history
            question_index = st.session_state.project_defence_base_index
            follow_up_count = st.session_state.project_defence_follow_up_count

            progress_value = min(
                (question_index + 1) / max(len(questions), 1),
                1.0,
            )

            progress_col, answered_col = st.columns(2)
            progress_col.metric(
                "Interview progress",
                f"{question_index + 1} / {len(questions)}",
            )
            answered_col.metric("Answers submitted", len(history))

            st.progress(
                progress_value,
                text=f"Question {question_index + 1} of {len(questions)}",
            )

            with st.container(border=True):
                st.caption("LIVE PROJECT DEFENCE")
                if follow_up_count:
                    st.markdown("##### Adaptive follow-up question")
                    st.caption(
                        f"Follow-up {follow_up_count} of 2 · "
                        "Generated from your previous answer"
                    )
                else:
                    st.markdown("##### Current interview question")
                    st.caption("Repository-grounded technical question")

                st.markdown(
                    f"### {st.session_state.project_defence_current_question}"
                )

            st.markdown("#### Your response")
            st.caption(
                "Explain the implementation, technical decisions and "
                "how the repository supports your answer."
            )

            with st.form(
                key=(
                    f"project_defence_answer_"
                    f"{question_index}_"
                    f"{follow_up_count}"
                )
            ):
                candidate_answer = st.text_area(
                    "Write your answer",
                    height=180,
                    placeholder=(
                        "Describe your approach, the code involved, "
                        "why you chose it and any trade-offs..."
                    ),
                    label_visibility="collapsed",
                )

                submitted = st.form_submit_button(
                    "Evaluate my answer",
                    type="primary",
                    width="stretch",
                    icon=":material/psychology:",
                )

            if submitted:
                if not candidate_answer.strip():
                    st.warning("Please write an answer before submitting.")
                else:
                    try:
                        with st.spinner(
                            "Evaluating your answer against repository evidence..."
                        ):
                            feedback = evaluate_project_answer(
                                question=(
                                    st.session_state.project_defence_current_question
                                ),
                                answer=candidate_answer,
                                repository_analysis=analysis,
                                project_evaluation=evaluation,
                                history=history,
                                follow_up_count=(
                                    st.session_state.project_defence_follow_up_count
                                ),
                                max_follow_ups=2,
                            )

                        history.append({
                            "question": (
                                st.session_state.project_defence_current_question
                            ),
                            "answer": candidate_answer.strip(),
                            "score": feedback["score"],
                            "assessment": feedback["assessment"],
                            "strengths": feedback["strengths"],
                            "knowledge_gaps": feedback["knowledge_gaps"],
                            "evidence_references": (
                                feedback["evidence_references"]
                            ),
                        })

                        st.session_state.project_defence_history = history
                        st.session_state.project_defence_last_feedback = feedback
 

                        if feedback["follow_up_question"]:
                            st.session_state.project_defence_current_question = (
                                feedback["follow_up_question"]
                            )
                            st.session_state.project_defence_follow_up_count += 1
                        else:
                            st.session_state.project_defence_follow_up_count = 0
                            next_index = question_index + 1

                            if next_index < len(questions):
                                st.session_state.project_defence_base_index = (
                                    next_index
                                )
                                st.session_state.project_defence_current_question = (
                                    questions[next_index]
                                )
                            else:
                                st.session_state.project_defence_active = False
                                st.session_state.project_defence_completed = True

                        st.rerun()

                    except ProjectDefenceError as error:
                        st.error(str(error))
                    except Exception as error:
                        st.error("Could not evaluate this answer.")
                        st.caption(
                            f"Technical detail: {type(error).__name__}: {error}"
                        )

        feedback = st.session_state.project_defence_last_feedback

        if feedback:
            st.markdown("#### Latest answer feedback")
            st.metric("Answer score", f"{feedback['score']}/10")
            st.write(feedback["assessment"])

            if feedback["strengths"]:
                st.markdown("**Strengths**")
                for item in feedback["strengths"]:
                    st.markdown(f"- {item}")

            if feedback["knowledge_gaps"]:
                st.markdown("**Knowledge gaps**")
                for item in feedback["knowledge_gaps"]:
                    st.markdown(f"- {item}")

            if feedback["evidence_references"]:
                st.markdown("**Repository evidence references**")
                for item in feedback["evidence_references"]:
                    st.code(item)

            if feedback["follow_up_question"]:
                st.caption(
                    "The next question is adapted from your previous answer."
                )

        if st.session_state.project_defence_completed:
            history = st.session_state.project_defence_history

            st.divider()
            st.markdown("### Final project defence results")

            if history:
                average_score = sum(
                    item["score"] for item in history
                ) / len(history)

                score_column, answer_column = st.columns(2)
                score_column.metric(
                    "Average answer score",
                    f"{average_score:.1f}/10",
                )
                answer_column.metric(
                    "Answers evaluated",
                    len(history),
                )

                st.markdown("#### Answer history")

                for index, item in enumerate(history, start=1):
                    with st.expander(
                        f"Answer {index} - {item['score']}/10"
                    ):
                        st.markdown("**Question**")
                        st.write(item["question"])
                        st.markdown("**Your answer**")
                        st.write(item["answer"])
                        st.markdown("**Assessment**")
                        st.write(item["assessment"])

                st.download_button(
                    "Download interactive defence results",
                    data=__import__("json").dumps(
                        history,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    file_name="interactive_project_defence.json",
                    mime="application/json",
                    key="download_interactive_defence",
                )

            if st.button(
                "Restart project defence",
                key="restart_project_defence",
            ):
                st.session_state.project_defence_questions = []
                st.session_state.project_defence_history = []
                st.session_state.project_defence_current_question = ""
                st.session_state.project_defence_base_index = 0
                st.session_state.project_defence_follow_up_count = 0
                st.session_state.project_defence_last_feedback = None
                st.session_state.project_defence_active = False
                st.session_state.project_defence_completed = False
                st.rerun()
