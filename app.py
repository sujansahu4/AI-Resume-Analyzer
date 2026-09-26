import os
import json
import re

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

# This also helps later when we deploy on Streamlit Cloud
if not API_KEY:
    try:
        API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        API_KEY = None

if not API_KEY:
    st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()

client = Groq(api_key=API_KEY)

#MODEL = "llama-3.3-70b-versatile"
MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">AI Resume Analyzer & Job Matcher</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze your resume against a job description using Generative AI.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FUNCTIONS
# ============================================================

def extract_resume_text(uploaded_file):
    """Extract text from uploaded PDF."""

    try:
        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception as e:
        raise Exception(f"Unable to read PDF: {e}")


def clean_text(text):
    """Clean unnecessary whitespace."""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_json_response(content):
    """Safely extract JSON from model response."""

    content = content.strip()

    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"^```\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    try:
        return json.loads(content)

    except json.JSONDecodeError:

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:

            json_text = content[start:end + 1]

            return json.loads(json_text)

        raise ValueError(
            "The AI returned an invalid response. Please try again."
        )


def analyze_resume(resume_text, job_description):
    """Send resume and job description to Groq LLM."""

    prompt = f"""
You are an expert technical recruiter and ATS resume analyzer.

Analyze the candidate resume against the provided job description.

Your response MUST be valid JSON.

Return exactly this structure:

{{
    "ats_score": 0,
    "candidate_summary": "",
    "matched_skills": [],
    "missing_skills": [],
    "partial_match_skills": [],
    "strengths": [],
    "weaknesses": [],
    "experience_match": "",
    "education_match": "",
    "project_match": "",
    "resume_improvements": [],
    "interview_questions": []
}}

Rules:

1. ats_score must be an integer between 0 and 100.
2. matched_skills must contain skills clearly present in both resume and job description.
3. missing_skills must contain important job requirements absent from the resume.
4. partial_match_skills must contain skills that are related but not strongly demonstrated.
5. Do not invent candidate experience.
6. Base the analysis only on the provided resume and job description.
7. interview_questions should contain 10 relevant questions.
8. Keep the answer concise but useful.
9. Return JSON only.

RESUME:

{resume_text}

JOB DESCRIPTION:

{job_description}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a professional ATS resume analyzer. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content

    return parse_json_response(content)


def generate_resume_tips(resume_text):
    """Generate general resume improvement tips."""

    prompt = f"""
Review this resume as a professional technical recruiter.

Provide 8 practical recommendations to improve it.

Focus on:

- ATS compatibility
- Technical skills
- Project descriptions
- Achievement statements
- Keywords
- Formatting
- Quantifiable results
- Professional summary

Resume:

{resume_text}

Return only a numbered list.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert resume coach."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Information")

    st.write(
        """
        This application uses Generative AI to compare a
        candidate's resume with a job description.
        """
    )

    st.divider()

    st.subheader("Technology")

    st.write(
        """
        - Python
        - Streamlit
        - Groq
        - LLM
        - PDF Processing
        - Prompt Engineering
        """
    )

    st.divider()

    st.subheader("Analysis")

    st.write(
        """
        - ATS Score
        - Matched Skills
        - Missing Skills
        - Strengths
        - Weaknesses
        - Interview Questions
        """
    )


# ============================================================
# MAIN INPUTS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("1. Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )


with col2:

    st.subheader("2. Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste the complete job description..."
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze_button = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS
# ============================================================

if analyze_button:

    if uploaded_file is None:

        st.warning("Please upload a resume PDF.")

        st.stop()

    if not job_description.strip():

        st.warning("Please enter a job description.")

        st.stop()

    with st.spinner("Reading and analyzing resume..."):

        try:

            resume_text = extract_resume_text(uploaded_file)

            resume_text = clean_text(resume_text)

            if len(resume_text) < 100:

                st.error(
                    "Very little text was extracted from the PDF. "
                    "Please upload a text-based PDF."
                )

                st.stop()

            result = analyze_resume(
                resume_text,
                job_description
            )

            st.session_state["analysis"] = result
            st.session_state["resume_text"] = resume_text

        except Exception as e:

            st.error(f"Error: {e}")

            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    st.divider()

    st.header("Resume Analysis Report")


    # ATS SCORE

    score = result.get("ats_score", 0)

    score_col1, score_col2, score_col3 = st.columns(3)


    with score_col1:

        st.metric(
            "ATS Match Score",
            f"{score}%"
        )


    with score_col2:

        matched_count = len(
            result.get("matched_skills", [])
        )

        st.metric(
            "Matched Skills",
            matched_count
        )


    with score_col3:

        missing_count = len(
            result.get("missing_skills", [])
        )

        st.metric(
            "Missing Skills",
            missing_count
        )


    st.progress(
        min(max(score, 0), 100) / 100
    )


    # SUMMARY

    st.subheader("Candidate Summary")

    st.write(
        result.get(
            "candidate_summary",
            "No summary available."
        )
    )


    # SKILLS

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Matched Skills")

        matched = result.get(
            "matched_skills",
            []
        )

        if matched:

            for skill in matched:
                st.write(f"✅ {skill}")

        else:

            st.write("No strong matches identified.")


    with col2:

        st.subheader("Missing Skills")

        missing = result.get(
            "missing_skills",
            []
        )

        if missing:

            for skill in missing:
                st.write(f"• {skill}")

        else:

            st.write("No major missing skills identified.")


    # PARTIAL MATCH

    st.subheader("Partial Match Skills")

    partial = result.get(
        "partial_match_skills",
        []
    )

    if partial:

        for skill in partial:
            st.write(f"~ {skill}")

    else:

        st.write("No partial matches identified.")


    # STRENGTHS AND WEAKNESSES

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Strengths")

        strengths = result.get(
            "strengths",
            []
        )

        for item in strengths:
            st.write(f"✅ {item}")


    with col2:

        st.subheader("Weaknesses")

        weaknesses = result.get(
            "weaknesses",
            []
        )

        for item in weaknesses:
            st.write(f"• {item}")


    # EXPERIENCE / EDUCATION / PROJECT

    st.subheader("Experience Match")

    st.write(
        result.get(
            "experience_match",
            "Not available."
        )
    )


    st.subheader("Education Match")

    st.write(
        result.get(
            "education_match",
            "Not available."
        )
    )


    st.subheader("Project Match")

    st.write(
        result.get(
            "project_match",
            "Not available."
        )
    )


    # IMPROVEMENTS

    st.subheader("Resume Improvement Recommendations")

    improvements = result.get(
        "resume_improvements",
        []
    )

    for index, item in enumerate(
        improvements,
        start=1
    ):

        st.write(
            f"{index}. {item}"
        )


    # INTERVIEW QUESTIONS

    st.subheader("AI-Generated Interview Questions")

    questions = result.get(
        "interview_questions",
        []
    )

    for index, question in enumerate(
        questions,
        start=1
    ):

        st.write(
            f"{index}. {question}"
        )


    # GENERAL RESUME TIPS

    with st.expander(
        "Generate General Resume Improvement Tips"
    ):

        if st.button(
            "Generate Tips"
        ):

            with st.spinner(
                "Generating recommendations..."
            ):

                try:

                    tips = generate_resume_tips(
                        st.session_state["resume_text"]
                    )

                    st.write(tips)

                except Exception as e:

                    st.error(
                        f"Unable to generate tips: {e}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume Analyzer | Python + Streamlit + Groq"
)