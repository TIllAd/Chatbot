from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
import requests
import chromadb
import ollama
import os
import time
import re
from rank_bm25 import BM25Okapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from oauthlib.oauth1 import SignatureOnlyEndpoint, RequestValidator
from urllib.parse import urlencode


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.get("/lti/launch")
def lti_launch_get():
    return FileResponse("index.html")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = OLLAMA_HOST + "/api/chat"
MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

ollama_client = ollama.Client(host=OLLAMA_HOST)
# Hybrid search config
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = 1 - VECTOR_WEIGHT
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))

# Thresholds
HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.65

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("faq")



# --- BM25 Index Setup ---
GERMAN_STOPWORDS = {        # we dont want to overvalue finding filler words to confuse the retrieval
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "sein",
    "und", "oder", "aber", "wenn", "weil", "dass", "als", "wie", "was",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
    "einem", "einer", "ist", "sind", "war", "wird", "werden", "wurde",
    "hat", "haben", "hatte", "kann", "können", "muss", "müssen", "soll",
    "nicht", "auch", "noch", "schon", "sehr", "mehr", "nur", "von",
    "mit", "für", "auf", "an", "in", "zu", "zum", "zur", "bei", "nach",
    "über", "unter", "vor", "hinter", "zwischen", "durch", "aus", "bis",
    "im", "am", "vom", "beim", "ins", "ans", "es", "man", "sich",
    "hier", "dort", "da", "so", "dann", "denn", "mal", "doch", "ja",
    "nein", "kein", "keine", "keinen", "einem", "dieses", "dieser",
    "diese", "jeder", "jede", "jedes", "alle", "alles", "mich", "mir",
    "dir", "ihm", "uns", "euch", "ihnen", "wo", "wer", "wann",
    "warum", "welche", "welcher", "welches", "ob", "immer", "wieder",
    "gibt", "gibt", "sollte", "sollten", "würde", "würden", "könnte",
}


# LTI 1.1 Config
LTI_CONSUMER_KEY = os.getenv("LTI_CONSUMER_KEY", "wiso-chatbot")
LTI_SHARED_SECRET = os.getenv("LTI_SHARED_SECRET", "change-me-secret")

class LTIRequestValidator(RequestValidator):
    @property
    def client_key_length(self):
        return (3, 50)
    
    @property
    def nonce_length(self):
        return (20, 50)
    
    def validate_client_key(self, client_key, request):
        return client_key == LTI_CONSUMER_KEY
    
    def get_client_secret(self, client_key, request):
        return LTI_SHARED_SECRET
    
    def validate_timestamp_and_nonce(self, client_key, timestamp, nonce, 
                                      request_token=None, access_token=None, request=None):
        return True  # For simplicity; production should check for replay attacks
    
lti_validator = LTIRequestValidator()
lti_endpoint = SignatureOnlyEndpoint(lti_validator)


@app.post("/lti/launch")
async def lti_launch(request: Request):
    form = await request.form()
    body = dict(form)
    
    # Check consumer key matches
    if body.get("oauth_consumer_key") != LTI_CONSUMER_KEY:
        return HTMLResponse("<h1>LTI Authentication Failed</h1>", status_code=403)
    
    # Extract user info
    user_name = body.get("lis_person_name_full", body.get("lis_person_name_given", "Student"))
    user_role = body.get("roles", "Student")
    
    # Serve the chatbot UI
    return FileResponse("index.html")



def tokenize(text: str) -> list[str]:
    """German-friendly tokenizer with stopword removal."""
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in GERMAN_STOPWORDS]






def build_bm25_index():
    """Load all chunks from ChromaDB and build a BM25 index at startup."""
    all_docs = collection.get(include=["documents", "metadatas"])
    doc_ids = all_docs["ids"]
    doc_texts = all_docs["documents"]  # enriched (with tags) — used for BM25
    doc_originals = [
        m.get("original_text", doc_texts[i]) 
        for i, m in enumerate(all_docs["metadatas"])
    ]  # original (no tags) — used for LLM context
    tokenized = [tokenize(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, doc_ids, doc_texts, doc_originals

try:
    bm25_index, all_ids, all_texts, all_originals = build_bm25_index()
except Exception as e:
    print("BM25 build failed:", e)
    bm25_index, all_ids, all_texts, all_originals = None, [], [], []
print(f"BM25 index built with {len(all_ids)} chunks")

class ChatRequest(BaseModel):
    message: str

def retrieve_context(question: str, n_results: int = 25):
    start = time.time()

    # --- Vector search (get larger candidate pool) ---
    response = ollama_client.embeddings(model=EMBED_MODEL, prompt=question)
    embedding = response["embedding"]

    vector_results = collection.query(
        query_embeddings=[embedding],
        n_results=CANDIDATE_POOL,
        include=["documents", "distances"]
    )

    vector_ids = vector_results["ids"][0]
    vector_docs = vector_results["documents"][0]
    vector_distances = vector_results["distances"][0]
    vector_scores = {
        vid: round(1 - (d / 2), 4)
        for vid, d in zip(vector_ids, vector_distances)
    }

    # --- BM25 search ---
    query_tokens = tokenize(question)
    bm25_raw_scores = bm25_index.get_scores(query_tokens)

    # Normalize BM25 scores to 0-1
    max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
    bm25_scores = {
        all_ids[i]: round(bm25_raw_scores[i] / max_bm25, 4)
        for i in range(len(all_ids))
    }

    # --- Combine scores for all candidates ---
    candidate_ids = set(vector_ids) | {
        all_ids[i] for i, s in enumerate(bm25_raw_scores) if s > 0
    }

    combined = []
    for cid in candidate_ids:
        v_score = vector_scores.get(cid, 0)
        b_score = bm25_scores.get(cid, 0)
        final = round(VECTOR_WEIGHT * v_score + BM25_WEIGHT * b_score, 4)

        idx = all_ids.index(cid)
        doc = all_texts[idx]

        combined.append({
            "id": cid,
            "vector_score": v_score,
            "bm25_score": b_score, 
            "combined_score": final,
            "document": doc,
            "original": all_originals[idx]
        })

    # Sort by combined score, take top n
    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    top = combined[:n_results]

    elapsed = int((time.time() - start) * 1000)

    debug_chunks = [
        {
            "rank": i + 1,
            "id": item["id"],
            "combined_score": item["combined_score"],
            "vector_score": item["vector_score"],
            "bm25_score": item["bm25_score"],
            "preview": item["document"][:200] + "..." if len(item["document"]) > 200 else item["document"]
        }
        for i, item in enumerate(top)
    ]

    context = "\n\n".join(f"[{item['id']}] {item['original']}" for item in top)
    return context, debug_chunks, elapsed
# --- Add these endpoints to your main.py ---

@app.get("/inspect/chunks")
def inspect_chunks():
    """Return all chunks in the collection for browsing."""
    all_docs = collection.get(include=["documents", "metadatas"])
    chunks = [
        {
            "id": all_docs["ids"][i],
            "text": all_docs["metadatas"][i].get("original_text", all_docs["documents"][i]),
            "keywords": all_docs["metadatas"][i].get("keywords", ""),
            "enriched": all_docs["documents"][i]
        }
        for i in range(len(all_docs["ids"]))
    ]
    return {"chunks": chunks}

class InspectSearchRequest(BaseModel):
    query: str
    n_results: int = 10

@app.post("/inspect/search")
def inspect_search(req: InspectSearchRequest):
    """Run hybrid search and return detailed scoring for inspection."""
    _, debug_chunks, elapsed = retrieve_context(req.query, n_results=req.n_results)
    
    # Enrich with full document text
    for chunk in debug_chunks:
        idx = all_ids.index(chunk["id"])
        chunk["document"] = all_texts[idx]
    
    return {"results": debug_chunks, "elapsed_ms": elapsed, "query": req.query}

# Replace the entire chat endpoint and add the helper function above it:

def build_debug(debug_chunks, top_score, verdict, retrieval_ms, llm_ms=0):
    """Build debug info dict — keeps chat endpoint clean."""
    return {
        "retrieved_chunks": debug_chunks[:5],  # only top 5 for UI
        "top_score": top_score,
        "verdict": verdict,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "search_mode": "hybrid",
        "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "mode": "N/A"
    }

@app.post("/chat")
def chat(req: ChatRequest, debug: bool = Query(default=False)):
    context, debug_chunks, retrieval_ms = retrieve_context(req.message)
    top_score = debug_chunks[0]["combined_score"] if debug_chunks else 0
    show_debug = debug or DEBUG_MODE

    # --- Below threshold → skip LLM ---
    if top_score < LOW_CONFIDENCE:
        reply = (
            "Ich konnte leider keine passende Antwort in meiner FAQ finden. 😕\n"
            "Versuche es mit einer anderen Formulierung oder einem anderen Stichwort, "
            "oder schau direkt in den Quellen nach."
        )
        result = {"reply": reply}
        if show_debug:
            result["debug"] = build_debug(debug_chunks, top_score, "below threshold - LLM skipped", retrieval_ms)
            result["debug"]["mode"] = "REJECT"
        return result

    # --- Determine mode ---
    if top_score >= HIGH_CONFIDENCE:
        mode = "ANSWER"
    else:
        mode = "ANSWER_WITH_CAUTION"

    # --- Build prompt ---
    system_prompt = f"""Du bist der WiSo-Chatbot der FAU Erlangen-Nürnberg. Du hilfst Erstsemester-Studierenden, sich im Studium zurechtzufinden.

    MODUS: {mode}

    MODUS-REGELN:
    - ANSWER_WITH_CAUTION: Antworte kurz und füge eine Rückfrage hinzu, ob das die richtige Frage war.
    - ANSWER: Antworte kurz und hilfreich.

    DEIN WICHTIGSTES ZIEL:
    Hilf Studierenden, die Info SELBST zu finden. Nenne immer die konkrete Quelle oder Anlaufstelle (z.B. "Homepage des Prüfungsamtes", "Campo", "StudOn", "MHB", "RRZE Website"). Nenne NIEMALS Chunk-IDs.

    ANTWORT-REGELN:
    - Antworte NUR mit Informationen aus den QUELLEN unten.
    - Wenn die QUELLEN keine Antwort auf die Frage enthalten, sage IMMER: "Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen 😊"
    - Das gilt auch für Witze, Smalltalk, persönliche Fragen, oder alles was nicht direkt mit dem WiSo-Studium zu tun hat.
    - Erfinde NICHTS dazu. Keine eigenen Informationen, keine Vermutungen.
    - Antworte auf Deutsch, kurz und freundlich (du-Form).

    FORMAT:
    1) Kurze Antwort (2-3 Sätze max)
    2) 📍 Wo du das findest: [konkrete Quelle]

    Beispiel:
    Frage: "Wie melde ich mich für Prüfungen an?"
    Antwort: Die Anmeldung läuft über Campo und muss im Anmeldezeitraum erfolgen. Den genauen Zeitraum bekommst du per Mail.
    📍 Wo du das findest: Campo (campo.fau.de) → Angemeldete Prüfungen

    QUELLEN:
    {context}""".strip()

    # --- Call LLM ---
    llm_start = time.time()
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
    llm_ms = int((time.time() - llm_start) * 1000)
    reply = data["message"]["content"]

    result = {"reply": reply}
    if show_debug:
        result["debug"] = build_debug(
            debug_chunks, top_score,
            "high confidence" if top_score >= HIGH_CONFIDENCE else "borderline",
            retrieval_ms, llm_ms
        )
        result["debug"]["mode"] = mode

    return result