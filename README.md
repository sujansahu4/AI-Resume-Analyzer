# AI Resume Analyzer & Job Matcher

An AI-powered resume analysis application built using Python, Streamlit, and Groq.

## Features

- Upload a PDF resume
- Paste a job description
- Generate ATS match score
- Identify matched skills
- Identify missing skills
- Show partial skill matches
- Analyze strengths and weaknesses
- Check experience, education, and project match
- Generate resume improvement suggestions
- Generate AI-based interview questions

## Technologies Used

- Python
- Streamlit
- Groq API
- Large Language Models
- PyPDF
- Prompt Engineering

## How It Works

1. User uploads a PDF resume.
2. User enters a job description.
3. Resume text is extracted using PyPDF.
4. Resume and job description are sent to the Groq LLM.
5. The AI returns structured analysis.
6. Streamlit displays the ATS score and recommendations.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt