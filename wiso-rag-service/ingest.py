"""
FAQ Ingestion Script — WiSo Chatbot
Reads faq.docx, chunks by question/answer pairs (using --> delimiter),
enriches with LLM-generated keywords, embeds with OpenAI, stores in ChromaDB.
"""

import os
import re
import time
import chromadb
from docx import Document
from openai import OpenAI

# --- Config ---
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
KEYWORD_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# --- Chunking ---
def load_and_chunk(path="faq.docx"):
    """
    Parse faq.docx into question/answer chunks.
    Format: question line(s) followed by --> answer line(s).
    Each chunk = "question --> answer(s)"
    """
    doc = Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # Skip header lines (Fragensammlung, Beispiel, etc.)
    lines = full_text.split("\n")

    chunks = []
    current_question = None
    current_answers = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("-->"):
            # This is an answer line
            answer = line.lstrip("->").strip()
            if answer:
                current_answers.append(answer)
        elif "?" in line or line.endswith(":"):
            # New question — save previous chunk first
            if current_question and current_answers:
                chunk_text = f"{current_question}\n--> " + "\n--> ".join(current_answers)
                chunks.append(chunk_text)

            current_question = line
            current_answers = []
        else:
            # Could be a continuation or header — skip headers
            if line in ("Fragensammlung", "Beispiel:", "Mögliche Frage",
                        "Mögliche Antwort", "Frage 1", "Frage 2", "Antwort"):
                continue
            # If we have a question context, treat as continuation of answer
            if current_question and current_answers:
                current_answers.append(line)
            elif current_question:
                # Might be a multi-line question
                current_question += " " + line

    # Don't forget the last chunk
    if current_question and current_answers:
        chunk_text = f"{current_question}\n--> " + "\n--> ".join(current_answers)
        chunks.append(chunk_text)

    return chunks

# --- Keyword enrichment ---
def generate_keywords(chunk_text: str) -> str:
    prompt = f"""Analysiere diese FAQ-Antwort für Studierende und erstelle 3-5 deutsche Suchbegriffe/Synonyme,
die Studierende wahrscheinlich eingeben würden, um diese Information zu finden.

FAQ-Text:
{chunk_text}

Antworte NUR mit den Suchbegriffen, kommagetrennt. Keine Erklärung."""

    try:
        response = openai_client.chat.completions.create(
            model=KEYWORD_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        keywords = response.choices[0].message.content.strip()
        return keywords
    except Exception as e:
        print(f"  Keyword generation failed: {e}")
        return ""

# --- Embedding ---
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

# --- Main ---
def ingest():
    chunks = load_and_chunk()
    print(f"Loaded {len(chunks)} chunks from faq.docx\n")

    if len(chunks) == 0:
        print("ERROR: No chunks found! Check faq.docx format.")
        return

    # Preview first 3 chunks
    for i, c in enumerate(chunks[:3]):
        print(f"  Preview chunk {i}: {c[:100]}...")
    print()

    # Reset collection
    try:
        chroma_client.delete_collection("faq")
    except:
        pass
    collection = chroma_client.create_collection(
        "faq",
        metadata={"hnsw:space": "cosine"}
    )

    total_start = time.time()

    for i, chunk in enumerate(chunks):
        start = time.time()

        # Generate keywords
        print(f"[{i+1}/{len(chunks)}] Generating keywords...", end=" ")
        keywords = generate_keywords(chunk)
        print(f"→ {keywords[:80]}{'...' if len(keywords) > 80 else ''}")

        # Build enriched text
        enriched = f"[TAGS: {keywords}]\n{chunk}" if keywords else chunk

        # Embed
        embedding = get_embedding(enriched)

        # Store
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[enriched],
            embeddings=[embedding],
            metadatas=[{
                "original_text": chunk,
                "keywords": keywords,
                "chunk_index": i
            }]
        )

        elapsed = time.time() - start
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

    total = time.time() - total_start
    print(f"\nDone! {len(chunks)} chunks stored in {total:.1f}s")
    print(f"Average: {total/len(chunks):.1f}s per chunk")

if __name__ == "__main__":
    ingest()