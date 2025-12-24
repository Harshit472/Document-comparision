import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import docx
import difflib

# ---------------- CONFIG ----------------
SIM_THRESHOLD = 0.80
MAX_DIFF_LINES = 3

# ---------------- MODEL -----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- UTILS -----------------
def clean_text(text):
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def extract_text_diff(a, b):
    """Extract readable changed text snippets"""
    a_lines = a.split(". ")
    b_lines = b.split(". ")

    diff = list(difflib.ndiff(a_lines, b_lines))

    removed = [l[2:] for l in diff if l.startswith("- ")][:MAX_DIFF_LINES]
    added = [l[2:] for l in diff if l.startswith("+ ")][:MAX_DIFF_LINES]

    return (
        " | ".join(removed) if removed else "No clear removed text",
        " | ".join(added) if added else "No clear added text"
    )

# ---------------- EXTRACTION ----------------
def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        images_count = len(page.get_images(full=True))

        # Only use real text layer, ignore scanned images
        if not text:
            # Image-only page, mark as no text
            text = ""

        pages.append({
            "page": page_num,
            "text": clean_text(text),
            "images": images_count
        })

    return pages, doc.page_count

def extract_docx(file):
    doc = docx.Document(file)
    text = clean_text(" ".join(p.text for p in doc.paragraphs))
    return [{"page": 1, "text": text, "images": 0}], 1

def extract_txt(file):
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except:
        text = raw.decode("latin-1")
    return [{"page": 1, "text": clean_text(text), "images": 0}], 1

def extract(upload):
    ext = Path(upload.name).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(upload.read())
    if ext == ".docx":
        return extract_docx(upload)
    if ext == ".txt":
        return extract_txt(upload)
    return [], 0

# ---------------- COMPARISON ----------------
def compare_pages(pages_a, pages_b):
    changes = []
    similarities = []

    max_pages = max(len(pages_a), len(pages_b))

    for i in range(max_pages):
        page_no = i + 1

        if i >= len(pages_a):
            changes.append({
                "Page": page_no,
                "Change Type": "Page Added",
                "Before": "-",
                "After": "New page added"
            })
            continue

        if i >= len(pages_b):
            changes.append({
                "Page": page_no,
                "Change Type": "Page Removed",
                "Before": "Page existed",
                "After": "-"
            })
            continue

        a = pages_a[i]
        b = pages_b[i]

        # Visual changes
        if b["images"] > a["images"]:
            changes.append({
                "Page": page_no,
                "Change Type": "Visual Added",
                "Before": f"{a['images']} image(s)",
                "After": f"{b['images']} image(s)"
            })
        elif b["images"] < a["images"]:
            changes.append({
                "Page": page_no,
                "Change Type": "Visual Removed",
                "Before": f"{a['images']} image(s)",
                "After": f"{b['images']} image(s)"
            })

        # Text changes (only if both pages have real text)
        if a["text"] and b["text"]:
            emb = model.encode([a["text"], b["text"]], normalize_embeddings=True)
            sim = cosine_similarity(emb[0], emb[1])
            similarities.append(sim)

            if sim < SIM_THRESHOLD:
                removed, added = extract_text_diff(a["text"], b["text"])
                changes.append({
                    "Page": page_no,
                    "Change Type": "Content Modified",
                    "Before": removed,
                    "After": added
                })

        # If page has no real text, skip text comparison to avoid junk

    avg_similarity = sum(similarities) / max(len(similarities), 1)
    return changes, avg_similarity

# ---------------- STREAMLIT UI ----------------
st.set_page_config("Document Comparison", layout="wide")
st.title("📄 Document Comparison")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Document A (Original)", type=["pdf", "docx", "txt"])
with col2:
    file_b = st.file_uploader("Upload Document B (Updated)", type=["pdf", "docx", "txt"])

if st.button("Compare Documents"):
    if not file_a or not file_b:
        st.error("Please upload both documents.")
    else:
        with st.spinner("Comparing documents..."):
            pages_a, count_a = extract(file_a)
            file_b.seek(0)
            pages_b, count_b = extract(file_b)

            detected_changes, similarity = compare_pages(pages_a, pages_b)
            documents_similar = "YES" if not detected_changes and count_a == count_b else "NO"

        st.subheader("Result")
        st.metric("Documents Similar?", documents_similar)

        st.subheader("Page Count")
        st.write(f"Document A: {count_a} pages")
        st.write(f"Document B: {count_b} pages")

        st.subheader("Detected Changes (Page-wise)")
        if detected_changes:
            st.dataframe(pd.DataFrame(detected_changes), use_container_width=True)
        else:
            st.success("No significant changes detected.")

