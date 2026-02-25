from docx import Document
import re
import chromadb
import ollama

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

        is_new_question = re.match(r'^\d+[\.\)]\s', text) or (
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

    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        embedding = response["embedding"]
        
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
        print(f"Stored chunk {i+1}/{len(chunks)}")

if __name__ == "__main__":
    chunks = load_chunks("faq.docx")
    embed_and_store(chunks)
    print("Done! FAQ stored in ChromaDB.")