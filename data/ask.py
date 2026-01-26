import os
import re
from docx import Document
from rank_bm25 import BM25Okapi
import ollama


# --------- 1) Load DOCX files ----------
def load_docx_files(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".docx"):
            path = os.path.join(folder_path, file)
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            docs.append({"file": file, "text": text})
    return docs


# --------- 2) Chunking ----------
def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap_words
        if start < 0:
            start = 0
        if start >= len(words):
            break
    return chunks


# --------- 3) Simple tokenizer ----------
def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [w for w in s.split() if len(w) > 1]


# --------- 4) Build BM25 “search index” ----------
def build_bm25_index(docs_folder: str):
    docs = load_docx_files(docs_folder)
    if not docs:
        raise RuntimeError("No .docx files found in data/docs")

    chunks = []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunks.append({"file": d["file"], "text": c})

    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


# --------- 5) Ask a question using retrieved chunks ----------
def answer_from_docs(question: str, bm25, chunks, top_k: int = 4):
    q_tokens = tokenize(question)
    scores = bm25.get_scores(q_tokens)

    # get top_k chunk indices
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    context_blocks = []
    citations = []

    for rank, i in enumerate(top_idx, start=1):
        c = chunks[i]
        context_blocks.append(f"[Source {rank}] File: {c['file']}\n{c['text']}")
        citations.append((rank, c["file"]))

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant.
Answer the user's question using ONLY the sources below.
If the answer is not in the sources, say: "I don't know based on the provided documents."

SOURCES:
{context}

QUESTION:
{question}

ANSWER (include citations like [Source 1], [Source 2] when you use them):
"""

    resp = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"], citations


if __name__ == "__main__":
    bm25, chunks = build_bm25_index("data/docs")

    print("✅ Ready. Ask a question about your documents.")
    while True:
        q = input("\nYour question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        answer, cites = answer_from_docs(q, bm25, chunks, top_k=4)
        print("\n--- Answer ---")
        print(answer)
