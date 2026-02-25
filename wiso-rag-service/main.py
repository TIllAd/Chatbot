from fastapi import FastAPI
from pydantic import BaseModel
import requests
import chromadb
import ollama

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"

# Load ChromaDB collection on startup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("faq")

class ChatRequest(BaseModel):
    message: str

def retrieve_context(question: str, n_results: int = 3) -> str:
    # Embed the question
    response = ollama.embeddings(model=EMBED_MODEL, prompt=question)
    embedding = response["embedding"]
    
    # Find most similar chunks
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    
    chunks = results["documents"][0]
    return "\n\n".join(chunks)

@app.post("/chat")
def chat(req: ChatRequest):
    # Get relevant FAQ chunks
    context = retrieve_context(req.message)
    
    # Build prompt with context
    system_prompt = f"""Du bist der WiSo-Chatbot der FAU Erlangen-Nürnberg. 
Beantworte Fragen von Studierenden kurz und hilfreich auf Deutsch.
Nutze folgende Informationen aus der FAQ um die Frage zu beantworten:

{context}

Falls die Antwort nicht in der FAQ steht, sage das ehrlich."""

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=body)
    data = response.json()
    return {"reply": data["message"]["content"]}