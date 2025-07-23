import streamlit as st
from transformers import pipeline
import PyPDF2

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ✅ Cleaned-up version of the PDF text extractor
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        raw_text = page.extract_text()
        if raw_text:
            # Replace multiple newlines and split words with clean space
            cleaned = " ".join(raw_text.split())
            text += cleaned + " "
    return text.strip()

def summarize_text(text, max_chunk=1000):
    summary = ""
    text = text.replace("\n", " ")
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    for chunk in chunks:
        summarized = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += summarized[0]['summary_text'] + " "
    return summary

# UI
st.set_page_config(page_title="📄 Document Summarization Tool", layout="centered")
st.title("📄 Document Summarization Tool")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        full_text = extract_text_from_pdf(uploaded_file)
        st.subheader("📃 Extracted Text Preview")
        st.text_area("Document Content", full_text[:3000] + ("..." if len(full_text) > 3000 else ""), height=300)

    if st.button("📝 Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = summarize_text(full_text)
            st.subheader("🧾 Summary")
            st.write(summary)
