import os
import re
import heapq
from pathlib import Path

import streamlit as st
from docx import Document
import ollama


# -------------------------
# Concept: Tokenization
# We normalize text so matching works better (lowercase, remove punctuation, remove common words).
# -------------------------
def tokenize(text: str):
    stopwords = {
        "a","an","the","and","or","but","if","then","else","when","while",
        "is","am","are","was","were","be","been","being",
        "this","that","these","those","which","what","who","whom","where","why","how",
        "of","to","in","on","for","with","as","by","at","from","into","about",
        "it","its","we","you","your","our","they","their"
    }
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [w for w in text.split() if len(w) > 1 and w not in stopwords]
    return tokens


# -------------------------
# Concept: Chunking (streaming)
# Instead of loading the entire doc into memory, we produce chunks one by one.
# This is why it runs on low-RAM machines.
# -------------------------
def stream_chunks_from_docx(path: str, chunk_words: int = 220, overlap_words: int = 40):
    doc = Document(path)
    words = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        words.extend(t.split())

        while len(words) >= chunk_words:
            chunk = " ".join(words[:chunk_words])
            yield chunk
            words = words[chunk_words - overlap_words :]

    if words:
        yield " ".join(words)


# -------------------------
# Concept: Retrieval (lightweight search)
# We score each chunk by overlap with question words and keep only top_k.
# -------------------------
def retrieve_top_chunks(docs_folder: str, question: str, top_k: int = 4):
    q_tokens = tokenize(question)

    # Helpful hints for university questions
    if "university" in question.lower() or "institution" in question.lower():
        q_tokens += ["university", "school", "department", "faculty", "campus", "msc"]

    q_set = set(q_tokens)
    top = []  # min-heap of (score, chunk_text, file)

    for file in os.listdir(docs_folder):
        if not file.lower().endswith(".docx"):
            continue
        # Skip Microsoft Word temp/lock files
        if file.startswith("~$"):
            continue

        path = os.path.join(docs_folder, file)

        for chunk in stream_chunks_from_docx(path):
            c_tokens = tokenize(chunk)
            score = sum(1 for w in c_tokens if w in q_set)
            if score <= 0:
                continue

            item = (score, chunk, file)
            if len(top) < top_k:
                heapq.heappush(top, item)
            else:
                if score > top[0][0]:
                    heapq.heapreplace(top, item)

    return sorted(top, key=lambda x: x[0], reverse=True)


def ask_phi3(question: str, retrieved, mode: str):
    if not retrieved:
        return "UNKNOWN" if mode == "extract_university" else "I don't know based on the provided documents."

    sources_text = []
    for i, (score, chunk, file) in enumerate(retrieved, start=1):
        sources_text.append(f"[Source {i}] File: {file}\n{chunk}")

    if mode == "extract_university":
        prompt = f"""You are an information extraction system.

TASK:
Extract ONLY the name of the university from the sources.

STRICT OUTPUT RULES:
- Output ONLY the university name.
- Do NOT add any extra words.
- Do NOT add punctuation.
- Do NOT add explanations.
- If no university name is explicitly mentioned, output exactly: UNKNOWN

SOURCES:
{chr(10).join(sources_text)}

QUESTION:
{question}

FINAL OUTPUT:
"""
    else:
        prompt = f"""You are a helpful assistant.

Rules:
- Answer ONLY the user's question.
- Use ONLY the sources below.
- If not explicitly stated in the sources, say: I don't know based on the provided documents.
- Keep the answer short (max 4 sentences).
- If you use a source, cite it like [Source 1].

SOURCES:
{chr(10).join(sources_text)}

QUESTION:
{question}

ANSWER:
"""

    resp = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"].strip()


# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="Local Doc Chat (phi3)", layout="wide")
st.title("Local Document Q&A (Runs on your Mac)")

BASE = Path(__file__).parent
DOCS_DIR = BASE / "data" / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("1) Upload documents")
    uploaded = st.file_uploader("Upload .docx files", type=["docx"], accept_multiple_files=True)

    if uploaded:
        saved = 0
        for f in uploaded:
            # Save into data/docs
            out_path = DOCS_DIR / f.name
            out_path.write_bytes(f.getbuffer())
            saved += 1
        st.success(f"Saved {saved} file(s) to {DOCS_DIR}")

    st.header("2) Settings")
    mode_label = st.radio(
        "Answer Mode",
        ["Normal Q&A", "University only"],
        index=0
    )
    show_sources = st.checkbox("Show evidence sources", value=True)
    top_k = st.slider("How many source chunks to use", 2, 6, 4)

st.divider()

# Show current docs
docs = [p.name for p in DOCS_DIR.glob("*.docx") if not p.name.startswith("~$")]
st.subheader("Documents loaded")
if docs:
    st.write(docs)
else:
    st.info("Upload at least one .docx file using the sidebar.")

st.subheader("Ask a question")
question = st.text_input("Type your question here")

if st.button("Answer", type="primary", disabled=not question.strip() or not docs):
    mode = "extract_university" if mode_label == "University only" else "qa"

    with st.spinner("Searching your documents and asking phi3..."):
        retrieved = retrieve_top_chunks(str(DOCS_DIR), question, top_k=top_k)
        answer = ask_phi3(question, retrieved, mode=mode)

    st.markdown("### Answer")
    st.write(answer)

    if show_sources:
        st.markdown("### Evidence (what the AI read)")
        if not retrieved:
            st.write("No relevant chunks found.")
        else:
            for i, (score, chunk, file) in enumerate(retrieved, start=1):
                st.markdown(f"**Source {i}** — `{file}` (score={score})")
                st.write(chunk)
                st.divider()
