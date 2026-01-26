import os
from docx import Document

from sentence_transformers import SentenceTransformer
import chromadb


# ---------- 1) Load DOCX files ----------
def load_docx_files(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".docx"):
            path = os.path.join(folder_path, file)
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            docs.append({"file": file, "text": text})
    return docs


# ---------- 2) Chunking ----------
def chunk_text(text: str, chunk_words: int = 350, overlap_words: int = 50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start = end - overlap_words
        if start < 0:
            start = 0
        if start >= len(words):
            break

    return chunks


# ---------- 3) Build the vector database ----------
def build_index(docs_folder: str, db_folder: str = "data/chroma"):
    docs = load_docx_files(docs_folder)

    if not docs:
        print("❌ No .docx files found in data/docs. Please copy a docx file there.")
        return

    print(f"📄 Found {len(docs)} docx file(s).")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=db_folder)

    try:
        client.delete_collection("my_docs")
    except Exception:
        pass

    collection = client.create_collection(name="my_docs")

    all_ids = []
    all_texts = []
    all_metas = []

    chunk_count = 0
    for d in docs:
        chunks = chunk_text(d["text"])
        for c in chunks:
            cid = f"{d['file']}::chunk_{chunk_count}"
            all_ids.append(cid)
            all_texts.append(c)
            all_metas.append({"file": d["file"]})
            chunk_count += 1

    print(f"🧩 Created {len(all_texts)} chunks. Storing embeddings...")

    collection.add(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metas
    )

    print("✅ Index built successfully!")
    print("📁 Stored at:", db_folder)


if __name__ == "__main__":
    build_index("data/docs")
