import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from io import BytesIO
import docx
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd

# ---------------- CONFIG ----------------
CHUNK_SIZE = 120
SIM_THRESHOLD = 0.78
SIM_RATIO_THRESHOLD = 0.25

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- HELPERS ----------------
def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())

def split_chunks(text, size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- EXTRACTION ----------------
def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = []
    image_count = 0

    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text.append(page_text)
        else:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(BytesIO(pix.tobytes("png")))
            image_count += 1
            text.append(pytesseract.image_to_string(img))

        image_count += len(page.get_images(full=True))

    return {
        "text": clean_text(" ".join(text)),
        "pages": doc.page_count,
        "images": image_count
    }

def extract_docx(file):
    doc = docx.Document(file)
    return {
        "text": clean_text(" ".join(p.text for p in doc.paragraphs)),
        "pages": 1,
        "images": 0
    }

def extract_txt(file):
    return {
        "text": clean_text(file.read().decode("utf-8", errors="ignore")),
        "pages": 1,
        "images": 0
    }

def extract(upload):
    ext = Path(upload.name).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(upload.read())
    if ext == ".docx":
        return extract_docx(upload)
    if ext == ".txt":
        return extract_txt(upload)
    return {"text": "", "pages": 0, "images": 0}

# ---------------- SEMANTIC MATCH ----------------
def semantic_diff(text_a, text_b):
    chunks_a = split_chunks(text_a)
    chunks_b = split_chunks(text_b)

    if not chunks_a or not chunks_b:
        return [], 1.0

    emb_a = model.encode(chunks_a, normalize_embeddings=True)
    emb_b = model.encode(chunks_b, normalize_embeddings=True)

    changes = []
    changed = 0

    for i, vec in enumerate(emb_a):
        sims = emb_b @ vec
        best = float(np.max(sims))
        if best < SIM_THRESHOLD:
            changed += 1
            changes.append({
                "Type": "Content Modified",
                "Description": f"Section {i+1} rewritten or changed"
            })

    ratio = changed / max(len(chunks_a), 1)
    return changes, ratio

# ---------------- STREAMLIT UI ----------------
st.set_page_config("Document Comparison", layout="wide")
st.title("📄 Document Comparison")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Document A", type=["pdf", "docx", "txt"])
with col2:
    file_b = st.file_uploader("Upload Document B", type=["pdf", "docx", "txt"])

if st.button("Compare Documents"):
    if not file_a or not file_b:
        st.error("Please upload both documents.")
    else:
        with st.spinner("Comparing documents..."):
            doc_a = extract(file_a)
            file_b.seek(0)
            doc_b = extract(file_b)

            detected_changes = []

            # Page count
            if doc_a["pages"] != doc_b["pages"]:
                detected_changes.append({
                    "Type": "Structural",
                    "Description": f"Page count changed from {doc_a['pages']} to {doc_b['pages']}"
                })

            # Image / signature detection
            if doc_a["images"] != doc_b["images"]:
                detected_changes.append({
                    "Type": "Visual",
                    "Description": "Visual elements changed (signature, stamp, or image added/removed)"
                })

            # Semantic text comparison
            semantic_changes, change_ratio = semantic_diff(doc_a["text"], doc_b["text"])
            detected_changes.extend(semantic_changes)

            similar = "YES" if change_ratio < SIM_RATIO_THRESHOLD and not detected_changes else "NO"

        # ---------------- OUTPUT ----------------
        st.subheader("Result")
        st.metric("Documents Similar?", similar)

        if detected_changes:
            st.subheader("Detected Changes")
            df = pd.DataFrame(detected_changes)
            st.table(df)
        else:
            st.success("No meaningful changes detected.")

        st.caption("Comparison considers structure, visuals, and semantic meaning.")

