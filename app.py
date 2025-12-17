"""
Document Comparison Streamlit App (Single File)

- Semantic similarity (YES / NO)
- Precise text-level change detection
- Supports PDF / DOCX / TXT
- OCR for scanned PDFs
"""

import difflib
from io import BytesIO
from pathlib import Path

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import docx
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

# -----------------------
# Config
# -----------------------
CHUNK_SIZE = 80
SIMILARITY_THRESHOLD = 0.75
SIMILARITY_RATIO_FOR_SIMILAR = 0.25

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------
# Text Helpers
# -----------------------
def clean_text(text):
    if not text:
        return ""
    text = text.replace("\r", " ")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

# -----------------------
# File Extraction
# -----------------------
def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
        else:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(BytesIO(pix.tobytes("png")))
            try:
                ocr_text = pytesseract.image_to_string(img)
            except:
                ocr_text = ""
            pages.append(ocr_text)

    return clean_text("\n".join(pages))

def extract_text(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(uploaded_file.read())

    if ext == ".docx":
        doc = docx.Document(BytesIO(uploaded_file.read()))
        return clean_text("\n".join(p.text for p in doc.paragraphs))

    if ext == ".txt":
        return clean_text(uploaded_file.read().decode("utf-8", errors="ignore"))

    return ""

# -----------------------
# Chunking & Similarity
# -----------------------
def split_into_chunks(text):
    words = text.split()
    chunks, buffer = [], []

    for w in words:
        buffer.append(w)
        if len(buffer) >= CHUNK_SIZE:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks

def semantic_similarity(text_a, text_b):
    chunks_a = split_into_chunks(text_a)
    chunks_b = split_into_chunks(text_b)

    if not chunks_a or not chunks_b:
        return "NO"

    emb_a = model.encode(chunks_a, normalize_embeddings=True)
    emb_b = model.encode(chunks_b, normalize_embeddings=True)

    changed = 0
    for ea in emb_a:
        sims = np.dot(emb_b, ea)
        if np.max(sims) < SIMILARITY_THRESHOLD:
            changed += 1

    ratio = changed / max(len(chunks_a), 1)
    return "YES" if ratio < SIMILARITY_RATIO_FOR_SIMILAR else "NO"

# -----------------------
# Precise Change Detection
# -----------------------
def detect_precise_changes(text_a, text_b):
    a_lines = text_a.splitlines()
    b_lines = text_b.splitlines()

    diff = difflib.ndiff(a_lines, b_lines)

    changes = []
    for line in diff:
        if line.startswith("- "):
            changes.append(("Removed", line[2:]))
        elif line.startswith("+ "):
            changes.append(("Added", line[2:]))

    return changes

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Document Comparison", layout="wide")
st.title("Document Comparison")

col1, col2 = st.columns(2)
with col1:
    doc_a = st.file_uploader("Upload Document A", type=["pdf", "docx", "txt"])
with col2:
    doc_b = st.file_uploader("Upload Document B", type=["pdf", "docx", "txt"])

if st.button("Compare Documents"):
    if not doc_a or not doc_b:
        st.error("Please upload both documents.")
    else:
        with st.spinner("Comparing documents..."):
            text_a = extract_text(doc_a)
            doc_b.seek(0)
            text_b = extract_text(doc_b)

            similarity = semantic_similarity(text_a, text_b)
            changes = detect_precise_changes(text_a, text_b)

        st.success("Comparison complete")

        st.subheader("Similarity Result")
        st.write(f"Documents Similar? **{similarity}**")

        st.subheader("Detected Changes")

        if not changes:
            st.info("No exact text changes detected.")
        else:
            for i, (ctype, text) in enumerate(changes, 1):
                st.write(f"{i}. **{ctype}**")
                st.write(text)
                st.markdown("---")
