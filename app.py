import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

def match_score(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def keyword_gap(resume_text, job_text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([resume_text, job_text])

    resume_words = set(vectorizer.build_analyzer()(resume_text))
    job_words = set(vectorizer.build_analyzer()(job_text))

    missing = job_words - resume_words
    return list(missing)[:top_n]

st.title("AI Resume Screener")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_text = st.text_area("Paste Job Description")

if st.button("Analyze") and resume_file and job_text:
    resume_text = extract_text_from_pdf(resume_file)

    score = match_score(resume_text, job_text)
    gaps = keyword_gap(resume_text, job_text)

    st.subheader(f"Match Score: {score}%")

    if gaps:
        st.write("Missing / Weak Keywords:")
        st.write(gaps)
    else:
        st.write("Strong keyword alignment.")
