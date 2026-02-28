from docx import Document
import re
import chromadb
import ollama
import time
import requests
import os


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = OLLAMA_HOST + "/api/chat"
ollama_client = ollama.Client(host=OLLAMA_HOST)
MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"

def load_chunks(filepath):
    doc = Document(filepath)
    chunks = []
    current_chunk = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if text in ["Fragensammlung", "Beispiel:"]:
            continue

        is_new_question = re.match(r'^\d+[\.\)]\s?', text) or (
            text.endswith("?") and not text.startswith("-->") and not text.startswith("-")
        )

        if is_new_question:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [text]
        else:
            if current_chunk:
                current_chunk.append(text)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

# Replace the generate_keywords function in ingest.py with this:

def generate_keywords(chunk: str) -> str:
    """Use LLM to generate search keywords/synonyms for a chunk."""
    prompt = f"""Aufgabe: Gib exakt 5 einzelne deutsche Suchbegriffe aus, die ein Student tippen würde um diesen FAQ-Eintrag zu finden.

Regeln:
- Exakt 5 Begriffe, kommagetrennt
- Nur einzelne Wörter oder kurze Begriffe (max 2 Wörter)
- Umgangssprachlich, wie Studenten suchen würden
- KEINE Sätze, KEINE Nummerierung, KEINE Erklärungen
- NUR die Begriffe, sonst nichts

FAQ-Eintrag:
{chunk}

Begriffe:"""

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=body)
        data = response.json()
        raw = data["message"]["content"].strip()
        
        # Clean up: take only the first line, remove numbering/bullets
        first_line = raw.split('\n')[0].strip()
        # Remove any numbering like "1." or "- "
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', first_line)
        cleaned = re.sub(r'^[-•]\s*', '', cleaned)
        
        # Split by comma, clean each keyword, take max 5
        keywords = [k.strip().strip('.').strip() for k in cleaned.split(',')]
        keywords = [k for k in keywords if k and len(k) < 30][:5]
        
        return ', '.join(keywords)
    except Exception as e:
        print(f"  ⚠️  Keyword generation failed: {e}")
        return ""

def embed_and_store(chunks):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection("faq")
    except:
        pass
    
    collection = client.create_collection(
        "faq",
        metadata={"hnsw:space": "cosine"}
    )

    total_start = time.time()

    for i, chunk in enumerate(chunks):
        # Generate keywords
        print(f"[{i+1}/{len(chunks)}] Generating keywords...", end=" ")
        keywords = generate_keywords(chunk)
        print(f"→ {keywords[:80]}{'...' if len(keywords) > 80 else ''}")

        # Create enriched version for embedding/search
        enriched = f"{chunk}\nTags: {keywords}" if keywords else chunk

        # Embed the enriched text
        response = ollama_client.embeddings(model=EMBED_MODEL, prompt=enriched)
        embedding = response["embedding"]
        
        # Store with original text in metadata, enriched text as document
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[enriched],
            metadatas=[{
                "original_text": chunk,
                "keywords": keywords
            }]
        )

    elapsed = time.time() - total_start
    print(f"\nDone! {len(chunks)} chunks stored in {elapsed:.1f}s")
    print(f"Average: {elapsed/len(chunks):.1f}s per chunk")

if __name__ == "__main__":
    chunks = load_chunks("faq.docx")
    print(f"Loaded {len(chunks)} chunks from faq.docx\n")
    embed_and_store(chunks)