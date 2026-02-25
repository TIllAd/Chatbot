from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Du bist der WiSo-Chatbot. Antworte kurz und hilfreich auf Deutsch."},
            {"role": "user", "content": req.message}
        ],
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=body)
    data = response.json()
    return {"reply": data["message"]["content"]}