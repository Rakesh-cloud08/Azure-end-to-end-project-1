import os
import re
import heapq
from docx import Document
import ollama


# -------------------------
# Concept: Tokenization
# Computers compare words better when we normalize them:
# - lowercase
# - remove punctuation
# -------------------------
def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]


# -------------------------
# Concept: Chunking (streaming)
# Instead of loading ALL chunks into memory,
# we generate chunks one by one (memory-friendly).
# -------------------------
def stream_chunks_from_docx(path: str, chunk_words: int = 220, overlap_words: int = 40):
    doc = Document(path)
    words = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        words.extend(t.split())

        # emit chunks whenever we have enough words
        while len(words) >= chunk_words:
            chunk = " ".join(words[:chunk_words])
            yield chunk
            # keep overlap (so context isn't lost at boundaries)
            words = words[chunk_words - overlap_words :]

    # last leftover
    if words:
        yield " ".join(words)


# -------------------------
# Concept: Retrieval (lightweight)
# We score each chunk by "how many question words appear in it".
# Keep only top_k chunks using a heap (efficient).
# -------------------------
def retrieve_top_chunks(docs_folder: str, question: str, top_k: int = 4):
    q_tokens = tokenize(question)
    if not q_tokens:
        return []

    q_set = set(q_tokens)
    top = []  # min-heap of (score, chunk_text, file)

    for file in os.listdir(docs_folder):
        if not file.lower().endswith(".docx"):
            continue
        path = os.path.join(docs_folder, file)

        for chunk in stream_chunks_from_docx(path):
            c_tokens = tokenize(chunk)

            # simple relevance score: count overlaps with question words
            score = sum(1 for w in c_tokens if w in q_set)

            if score <= 0:
                continue

            item = (score, chunk, file)
            if len(top) < top_k:
                heapq.heappush(top, item)
            else:
                # keep only best scores
                if score > top[0][0]:
                    heapq.heapreplace(top, item)

    # sort best first
    top_sorted = sorted(top, key=lambda x: x[0], reverse=True)
    return top_sorted


def ask_phi3_with_sources(question: str, retrieved, mode: str = "qa"):
    if not retrieved:
        if mode == "extract_university":
            return "UNKNOWN"
        return "I don't know based on the provided documents."

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


if __name__ == "__main__":
    docs_folder = "data/docs"
    if not os.path.isdir(docs_folder):
        print("❌ Folder not found:", docs_folder)
        print("Make sure your docx files are in ~/doc_chat_app/data/docs")
        raise SystemExit(1)

    print("✅ Ready (light mode). Ask a question about your documents.")
    while True:
        q = input("\nYour question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        retrieved = retrieve_top_chunks(docs_folder, q, top_k=4)

        print("\n--- Top sources picked (debug) ---")
        for i, (score, _chunk, file) in enumerate(retrieved, start=1):
            print(f"[Source {i}] score={score} file={file}")

        answer = ask_phi3_with_sources(q, retrieved)
        print("\n--- Answer ---")
        print(answer)
