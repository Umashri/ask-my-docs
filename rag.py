"""
ask-my-docs: a minimal RAG (Retrieval-Augmented Generation) system built from scratch.

Pipeline: load documents -> chunk -> embed -> retrieve relevant chunks -> generate a grounded answer.
No frameworks (no LangChain/vector DB) - raw numpy similarity search, to keep the mechanism visible.
"""

import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "models/gemini-3.6-flash"
CACHE_FILE = "vectors.json"

# Retrieval tuning: take all chunks scoring above THRESHOLD,
# but always return at least MIN_CHUNKS so the model is never starved of context.
THRESHOLD = 0.65
MIN_CHUNKS = 2


def load_chunks(path):
    """Read a document and split it into chunks on blank lines."""
    with open(path) as f:
        text = f.read()
    return text.split("\n\n")


def embed_text(text):
    """Turn a piece of text into an embedding vector."""
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return result["embedding"]


def embed_chunks(chunks):
    """Embed all chunks, caching to disk so we only call the API once."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)

    vectors = [embed_text(chunk) for chunk in chunks]
    with open(CACHE_FILE, "w") as f:
        json.dump(vectors, f)
    return vectors


def cosine_similarity(a, b):
    """Similarity between two vectors, measured as the angle between them (1 = identical meaning)."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question, chunks, chunk_vectors):
    """Find the chunks most relevant to the question.

    Uses a threshold with a top-k floor: keep everything above THRESHOLD,
    but if fewer than MIN_CHUNKS clear the bar, fall back to the top MIN_CHUNKS.
    This trims noise when scores are strong, and guarantees context when they're weak.
    """
    question_vector = embed_text(question)
    scores = [cosine_similarity(question_vector, v) for v in chunk_vectors]

    ranked = list(np.argsort(scores)[::-1])  # indices, highest score first
    above = [i for i in ranked if scores[i] >= THRESHOLD]
    indices = above if len(above) >= MIN_CHUNKS else ranked[:MIN_CHUNKS]

    return [chunks[i] for i in indices], [(i, scores[i]) for i in indices]


def generate_answer(question, retrieved_chunks):
    """Ask the model to answer using only the retrieved context."""
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}"""

    model = genai.GenerativeModel(CHAT_MODEL)
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"(Could not generate an answer: {e})"


def main():
    chunks = load_chunks("planets.txt")
    chunk_vectors = embed_chunks(chunks)
    print(f"Ready. {len(chunks)} chunks loaded.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        retrieved_chunks, scored = retrieve(question, chunks, chunk_vectors)

        print("\nRetrieved:")
        for i, score in scored:
            print(f"  Chunk {i} (score {score:.3f})")

        answer = generate_answer(question, retrieved_chunks)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()