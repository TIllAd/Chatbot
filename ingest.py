"""
Unified Ingestion Pipeline — WiSo Chatbot
Parses any supported file format (PDF, DOCX, PPTX, XLSX, HTML) via Docling,
cleans and chunks into semantic sections, enriches with keywords, embeds, stores in ChromaDB.
"""

import os
import re
import time
import chromadb
from pathlib import Path
from openai import OpenAI
from docling.document_converter import DocumentConverter

# --- Config ---
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
KEYWORD_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATA_DIR = os.getenv("DATA_DIR", "./data")
MAX_CHUNK_CHARS = 1500  # ~375 tokens
MIN_CHUNK_CHARS = 100   # skip tiny/junk fragments

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
converter = DocumentConverter()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md"}


# ─── Parsing ────────────────────────────────────────────────

def parse_file(file_path: str) -> str:
    """Convert any supported file to Markdown via Docling."""
    print(f"  Parsing with Docling...")
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


# ─── Cleaning ───────────────────────────────────────────────

def clean_markdown(markdown: str) -> str:
    """
    Clean Docling's raw Markdown output.
    Removes junk lines common in PPTX/PDF exports.
    """
    lines = markdown.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Skip image placeholders
        if "<!-- image -->" in stripped or "←!—" in stripped:
            continue

        # Skip lines that are just numbers (page numbers, slide numbers)
        if re.match(r'^\d{1,3}$', stripped):
            continue

        # Skip very short non-heading lines (likely footer/header junk)
        if stripped and not stripped.startswith("#") and len(stripped) < 4:
            continue

        # Skip common PPTX metadata patterns
        if re.match(r'^\d{1,2}\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*\d{4}$', stripped):
            # Keep dates but only if they're part of content, not standalone
            # For now skip standalone dates as they're usually slide footers
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def is_junk_chunk(text: str) -> bool:
    """
    Detect chunks that are just slide decoration / metadata with no real content.
    Returns True if the chunk should be skipped.
    """
    # Strip markdown formatting for analysis
    plain = re.sub(r'#+ ', '', text).strip()

    # Count "real" words (not just short tokens)
    words = [w for w in plain.split() if len(w) > 2]

    # Too few meaningful words
    if len(words) < 8:
        return True

    # Mostly just a heading with nothing else
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    non_heading_lines = [l for l in lines if not l.startswith("#")]
    if len(non_heading_lines) < 1:
        return True

    # Check if it's just a list of short fragments (common in PPTX title slides)
    if all(len(l) < 30 for l in non_heading_lines) and len(non_heading_lines) < 3:
        return True

    return False


# ─── Chunking ───────────────────────────────────────────────

def chunk_markdown(markdown: str, source_file: str) -> list[dict]:
    """
    Split Markdown into chunks by headings or double-newlines.
    Each chunk is a dict with 'text', 'source_file', 'section'.
    """
    # Clean first
    markdown = clean_markdown(markdown)

    chunks = []

    # Split on headings (##, ###, etc.)
    sections = re.split(r'\n(?=#{1,4}\s)', markdown)

    for section in sections:
        section = section.strip()
        if not section or len(section) < MIN_CHUNK_CHARS:
            continue

        # Extract heading as section title
        heading = ""
        lines = section.split("\n")
        if lines[0].startswith("#"):
            heading = lines[0].lstrip("#").strip()

        # Skip junk chunks
        if is_junk_chunk(section):
            continue

        # If section is too long, split further on double newlines
        if len(section) > MAX_CHUNK_CHARS:
            sub_parts = section.split("\n\n")
            current = ""
            for part in sub_parts:
                if len(current) + len(part) > MAX_CHUNK_CHARS and current:
                    if not is_junk_chunk(current):
                        chunks.append({
                            "text": current.strip(),
                            "source_file": source_file,
                            "section": heading
                        })
                    current = part
                else:
                    current = current + "\n\n" + part if current else part

            if current.strip() and len(current.strip()) >= MIN_CHUNK_CHARS and not is_junk_chunk(current):
                chunks.append({
                    "text": current.strip(),
                    "source_file": source_file,
                    "section": heading
                })
        else:
            chunks.append({
                "text": section,
                "source_file": source_file,
                "section": heading
            })

    return chunks


def chunk_faq_legacy(file_path: str) -> list[dict]:
    """
    Legacy chunker for the existing faq.docx format (question + --> answers).
    Falls back to this if a .docx file contains the --> pattern.
    """
    from docx import Document
    doc = Document(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # Check if this is our FAQ format
    if "-->" not in full_text:
        return []  # Not FAQ format, use Docling instead

    lines = full_text.split("\n")
    chunks = []
    current_question = None
    current_answers = []

    skip_headers = {"Fragensammlung", "Beispiel:", "Mögliche Frage",
                    "Mögliche Antwort", "Frage 1", "Frage 2", "Antwort"}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("-->"):
            answer = line.lstrip("->").strip()
            if answer:
                current_answers.append(answer)
        elif "?" in line or line.endswith(":"):
            if current_question and current_answers:
                chunk_text = f"{current_question}\n--> " + "\n--> ".join(current_answers)
                chunks.append({
                    "text": chunk_text,
                    "source_file": Path(file_path).name,
                    "section": current_question
                })
            current_question = line
            current_answers = []
        else:
            if line in skip_headers:
                continue
            if current_question and current_answers:
                current_answers.append(line)
            elif current_question:
                current_question += " " + line

    if current_question and current_answers:
        chunk_text = f"{current_question}\n--> " + "\n--> ".join(current_answers)
        chunks.append({
            "text": chunk_text,
            "source_file": Path(file_path).name,
            "section": current_question
        })

    return chunks


# ─── LLM-based chunk merging for PPTX ──────────────────────

def merge_slide_fragments(raw_chunks: list[dict], source_file: str) -> list[dict]:
    """
    PPTX slides often produce many tiny chunks that belong together.
    Merge consecutive small chunks from the same file into larger ones.
    """
    if not raw_chunks:
        return []

    merged = []
    current = raw_chunks[0].copy()

    for chunk in raw_chunks[1:]:
        # If both are small and from the same source, merge
        if (len(current["text"]) + len(chunk["text"]) < MAX_CHUNK_CHARS
                and chunk["source_file"] == current["source_file"]):
            current["text"] += "\n\n" + chunk["text"]
            # Keep the first heading, append the second if different
            if chunk["section"] and chunk["section"] != current["section"]:
                current["section"] += " / " + chunk["section"]
        else:
            if not is_junk_chunk(current["text"]):
                merged.append(current)
            current = chunk.copy()

    if not is_junk_chunk(current["text"]):
        merged.append(current)

    return merged


# ─── Enrichment ─────────────────────────────────────────────

def generate_keywords(chunk_text: str) -> str:
    """Generate search keywords for a chunk via LLM."""
    prompt = f"""Analysiere diesen Text für Studierende und erstelle 3-5 deutsche Suchbegriffe/Synonyme,
die Studierende wahrscheinlich eingeben würden, um diese Information zu finden.

Text:
{chunk_text}

Antworte NUR mit den Suchbegriffen, kommagetrennt. Keine Erklärung."""

    try:
        response = openai_client.chat.completions.create(
            model=KEYWORD_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Keyword generation failed: {e}")
        return ""


def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


# ─── Main ───────────────────────────────────────────────────

def ingest():
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        print(f"ERROR: Data directory '{DATA_DIR}' does not exist.")
        print(f"Create it and add your documents (PDF, DOCX, PPTX, etc.)")
        return

    # Collect all supported files
    files = [f for f in data_dir.iterdir()
             if f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")]

    if not files:
        print(f"No supported files found in {DATA_DIR}/")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"Found {len(files)} file(s) in {DATA_DIR}/:")
    for f in files:
        print(f"  • {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    print()

    # Parse and chunk all files
    all_chunks = []
    for file_path in sorted(files):
        print(f"📄 Processing: {file_path.name}")

        # Try legacy FAQ format first for .docx files
        if file_path.suffix.lower() == ".docx":
            faq_chunks = chunk_faq_legacy(str(file_path))
            if faq_chunks:
                print(f"  → {len(faq_chunks)} FAQ chunks (legacy format)")
                all_chunks.extend(faq_chunks)
                continue

        # Default: Docling → Markdown → clean → chunk
        try:
            markdown = parse_file(str(file_path))

            # Debug: show raw markdown preview
            print(f"  Raw markdown: {len(markdown)} chars")

            chunks = chunk_markdown(markdown, file_path.name)

            # For PPTX: merge small slide fragments
            if file_path.suffix.lower() == ".pptx":
                before = len(chunks)
                chunks = merge_slide_fragments(chunks, file_path.name)
                print(f"  Merged: {before} → {len(chunks)} chunks (slide fragment merging)")

            print(f"  → {len(chunks)} chunks")

            # Preview chunks for this file
            for i, c in enumerate(chunks[:2]):
                preview = c['text'][:120].replace('\n', ' ')
                print(f"    [{i}] {preview}...")

            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠ Failed to process: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Total: {len(all_chunks)} chunks from {len(files)} file(s)")
    print(f"{'='*60}\n")

    if not all_chunks:
        print("ERROR: No chunks extracted!")
        return

    # Reset collection
    try:
        chroma_client.delete_collection("faq")
    except:
        pass
    collection = chroma_client.create_collection("faq", metadata={"hnsw:space": "cosine"})

    # Enrich, embed, store
    total_start = time.time()

    for i, chunk in enumerate(all_chunks):
        start = time.time()

        print(f"[{i+1}/{len(all_chunks)}] {chunk['source_file']} | Keywords...", end=" ")
        keywords = generate_keywords(chunk["text"])
        print(f"→ {keywords[:70]}{'...' if len(keywords) > 70 else ''}")

        enriched = f"[TAGS: {keywords}]\n{chunk['text']}" if keywords else chunk["text"]
        embedding = get_embedding(enriched)

        collection.add(
            ids=[f"chunk_{i}"],
            documents=[enriched],
            embeddings=[embedding],
            metadatas=[{
                "original_text": chunk["text"],
                "keywords": keywords,
                "source_file": chunk["source_file"],
                "section": chunk["section"],
                "chunk_index": i
            }]
        )

        elapsed = time.time() - start
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

    total = time.time() - total_start
    print(f"\nDone! {len(all_chunks)} chunks stored in {total:.1f}s")
    print(f"Average: {total/len(all_chunks):.1f}s per chunk")


if __name__ == "__main__":
    ingest()