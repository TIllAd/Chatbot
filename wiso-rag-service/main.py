from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
import chromadb
import os
import time
import re
import json
from datetime import datetime, timezone
from rank_bm25 import BM25Okapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from oauthlib.oauth1 import SignatureOnlyEndpoint, RequestValidator
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- OpenAI Config ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# --- Hybrid search config ---
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = 1 - VECTOR_WEIGHT
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))

# --- Thresholds ---
HIGH_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.55

# --- Logging ---
LOG_FILE = os.getenv("LOG_FILE", "./logs/chat_log.jsonl")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_interaction(entry: dict):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Logging failed: {e}")

# --- ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("faq")

# --- Routes ---
@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.get("/inspector")
def serve_inspector():
    return FileResponse("inspector.html")

@app.get("/analytics")
def serve_analytics():
    return FileResponse("analytics.html")

@app.get("/lti/launch")
def lti_launch_get():
    return FileResponse("index.html")

# --- LTI 1.1 Config ---
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
        return True

lti_validator = LTIRequestValidator()
lti_endpoint = SignatureOnlyEndpoint(lti_validator)

@app.post("/lti/launch")
async def lti_launch(request: Request):
    form = await request.form()
    body = dict(form)
    if body.get("oauth_consumer_key") != LTI_CONSUMER_KEY:
        return HTMLResponse("<h1>LTI Authentication Failed</h1>", status_code=403)
    return FileResponse("index.html")

# --- BM25 Index Setup ---
GERMAN_STOPWORDS = {
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

def tokenize(text: str) -> list[str]:
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in GERMAN_STOPWORDS]

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

def build_bm25_index():
    all_docs = collection.get(include=["documents", "metadatas"])
    doc_ids = all_docs["ids"]
    doc_texts = all_docs["documents"]
    doc_originals = [
        m.get("original_text", doc_texts[i])
        for i, m in enumerate(all_docs["metadatas"])
    ]
    tokenized = [tokenize(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, doc_ids, doc_texts, doc_originals

try:
    bm25_index, all_ids, all_texts, all_originals = build_bm25_index()
except Exception as e:
    print("BM25 build failed:", e)
    bm25_index, all_ids, all_texts, all_originals = None, [], [], []
print(f"BM25 index built with {len(all_ids)} chunks")

# --- Retrieval ---
class ChatRequest(BaseModel):
    message: str

def retrieve_context(question: str, n_results: int = 25):
    start = time.time()
    embedding = get_embedding(question)

    vector_results = collection.query(
        query_embeddings=[embedding],
        n_results=CANDIDATE_POOL,
        include=["documents", "distances"]
    )

    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_scores = {
        vid: round(1 - (d / 2), 4)
        for vid, d in zip(vector_ids, vector_distances)
    }

    query_tokens = tokenize(question)
    bm25_raw_scores = bm25_index.get_scores(query_tokens)

    max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
    bm25_scores = {
        all_ids[i]: round(bm25_raw_scores[i] / max_bm25, 4)
        for i in range(len(all_ids))
    }

    candidate_ids = set(vector_ids) | {
        all_ids[i] for i, s in enumerate(bm25_raw_scores) if s > 0
    }

    combined = []
    for cid in candidate_ids:
        v_score = vector_scores.get(cid, 0)
        b_score = bm25_scores.get(cid, 0)
        final = round(VECTOR_WEIGHT * v_score + BM25_WEIGHT * b_score, 4)
        idx = all_ids.index(cid)
        combined.append({
            "id": cid, "vector_score": v_score, "bm25_score": b_score,
            "combined_score": final, "document": all_texts[idx], "original": all_originals[idx]
        })

    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    top = combined[:n_results]
    elapsed = int((time.time() - start) * 1000)

    debug_chunks = [
        {
            "rank": i + 1, "id": item["id"],
            "combined_score": item["combined_score"],
            "vector_score": item["vector_score"],
            "bm25_score": item["bm25_score"],
            "preview": item["document"][:200] + "..." if len(item["document"]) > 200 else item["document"]
        }
        for i, item in enumerate(top)
    ]

    context = "\n\n".join(f"[{item['id']}] {item['original']}" for item in top)
    return context, debug_chunks, elapsed

# --- LLM refusal detection ---
LLM_REJECT_PHRASES = [
    "ich kann dir nur bei fragen rund ums studium",
    "nur bei fragen rund ums studium an der wiso",
    "kann dir nur bei fragen",
]

def detect_llm_reject(reply: str) -> bool:
    """Check if the LLM refused to answer (off-topic detected by LLM itself)."""
    lower = reply.lower()
    return any(phrase in lower for phrase in LLM_REJECT_PHRASES)

# --- System prompt builder ---
def build_system_prompt(mode: str, context: str) -> str:
    return f"""Du bist der WiSo-Chatbot der FAU Erlangen-Nürnberg. Du hilfst Erstsemester-Studierenden, sich im Studium zurechtzufinden.

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

QUELLEN:
{context}""".strip()

# --- Inspect endpoints ---
@app.get("/inspect/chunks")
def inspect_chunks():
    all_docs = collection.get(include=["documents", "metadatas"])
    return {"chunks": [
        {
            "id": all_docs["ids"][i],
            "text": all_docs["metadatas"][i].get("original_text", all_docs["documents"][i]),
            "keywords": all_docs["metadatas"][i].get("keywords", ""),
            "source_file": all_docs["metadatas"][i].get("source_file", ""),
            "section": all_docs["metadatas"][i].get("section", ""),
            "enriched": all_docs["documents"][i]
        }
        for i in range(len(all_docs["ids"]))
    ]}

class InspectSearchRequest(BaseModel):
    query: str
    n_results: int = 10

@app.post("/inspect/search")
def inspect_search(req: InspectSearchRequest):
    _, debug_chunks, elapsed = retrieve_context(req.query, n_results=req.n_results)
    for chunk in debug_chunks:
        idx = all_ids.index(chunk["id"])
        chunk["document"] = all_texts[idx]
    return {"results": debug_chunks, "elapsed_ms": elapsed, "query": req.query}

# --- Logs endpoints ---
@app.get("/logs")
def get_logs(limit: int = Query(default=100), mode: str = Query(default=None)):
    try:
        if not os.path.exists(LOG_FILE):
            return {"logs": [], "total": 0}
        logs = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
        if mode:
            logs = [l for l in logs if l.get("mode") == mode]
        total = len(logs)
        logs = list(reversed(logs))[:limit]
        return {"logs": logs, "total": total}
    except Exception as e:
        return {"error": str(e)}

@app.get("/logs/stats")
def get_log_stats():
    try:
        if not os.path.exists(LOG_FILE):
            return {"total": 0}
        logs = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
        if not logs:
            return {"total": 0}
        modes = {}
        scores = []
        retrieval_times = []
        llm_times = []
        for log in logs:
            m = log.get("mode", "unknown")
            modes[m] = modes.get(m, 0) + 1
            if log.get("top_score"):
                scores.append(log["top_score"])
            if log.get("retrieval_ms"):
                retrieval_times.append(log["retrieval_ms"])
            if log.get("llm_ms"):
                llm_times.append(log["llm_ms"])
        return {
            "total": len(logs),
            "modes": modes,
            "avg_top_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "avg_retrieval_ms": round(sum(retrieval_times) / len(retrieval_times)) if retrieval_times else 0,
            "avg_llm_ms": round(sum(llm_times) / len(llm_times)) if llm_times else 0,
            "first_log": logs[0].get("timestamp"),
            "last_log": logs[-1].get("timestamp"),
        }
    except Exception as e:
        return {"error": str(e)}

# --- Chat (non-streaming, kept for eval/debug) ---
def build_debug(debug_chunks, top_score, verdict, retrieval_ms, llm_ms=0):
    return {
        "retrieved_chunks": debug_chunks[:5],
        "top_score": top_score, "verdict": verdict,
        "retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
        "model": MODEL, "embed_model": EMBED_MODEL,
        "search_mode": "hybrid", "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT, "mode": "N/A"
    }

@app.post("/chat")
def chat(req: ChatRequest, debug: bool = Query(default=False)):
    context, debug_chunks, retrieval_ms = retrieve_context(req.message)
    top_score = debug_chunks[0]["combined_score"] if debug_chunks else 0
    show_debug = debug or DEBUG_MODE

    if top_score < LOW_CONFIDENCE:
        reply = (
            "Ich konnte leider keine passende Antwort in meiner FAQ finden. 😕\n"
            "Versuche es mit einer anderen Formulierung oder einem anderen Stichwort, "
            "oder schau direkt in den Quellen nach."
        )
        log_interaction({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": req.message, "reply": reply, "mode": "REJECT",
            "top_score": top_score,
            "top_chunk_id": debug_chunks[0]["id"] if debug_chunks else None,
            "retrieval_ms": retrieval_ms, "llm_ms": 0,
        })
        result = {"reply": reply}
        if show_debug:
            result["debug"] = build_debug(debug_chunks, top_score, "below threshold - LLM skipped", retrieval_ms)
            result["debug"]["mode"] = "REJECT"
        return result

    mode = "ANSWER" if top_score >= HIGH_CONFIDENCE else "ANSWER_WITH_CAUTION"
    system_prompt = build_system_prompt(mode, context)

    llm_start = time.time()
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        max_tokens=300, temperature=0.3
    )
    llm_ms = int((time.time() - llm_start) * 1000)
    reply = response.choices[0].message.content

    # Detect if LLM refused (off-topic despite high retrieval score)
    actual_mode = "LLM_REJECT" if detect_llm_reject(reply) else mode

    log_interaction({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": req.message, "reply": reply, "mode": actual_mode,
        "top_score": top_score,
        "top_chunk_id": debug_chunks[0]["id"] if debug_chunks else None,
        "retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
    })

    result = {"reply": reply}
    if show_debug:
        result["debug"] = build_debug(
            debug_chunks, top_score,
            "high confidence" if top_score >= HIGH_CONFIDENCE else "borderline",
            retrieval_ms, llm_ms
        )
        result["debug"]["mode"] = actual_mode
    return result

# --- Streaming Chat (SSE) ---
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    context, debug_chunks, retrieval_ms = retrieve_context(req.message)
    top_score = debug_chunks[0]["combined_score"] if debug_chunks else 0

    def generate():
        # Send retrieval metadata first
        meta = {
            "type": "meta",
            "top_score": top_score,
            "retrieval_ms": retrieval_ms,
            "top_chunk_id": debug_chunks[0]["id"] if debug_chunks else None,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Below threshold → reject without calling LLM
        if top_score < LOW_CONFIDENCE:
            reply = (
                "Ich konnte leider keine passende Antwort in meiner FAQ finden. 😕\n"
                "Versuche es mit einer anderen Formulierung oder einem anderen Stichwort, "
                "oder schau direkt in den Quellen nach."
            )
            yield f"data: {json.dumps({'type': 'token', 'content': reply})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'mode': 'REJECT'})}\n\n"

            log_interaction({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": req.message, "reply": reply, "mode": "REJECT",
                "top_score": top_score,
                "top_chunk_id": debug_chunks[0]["id"] if debug_chunks else None,
                "retrieval_ms": retrieval_ms, "llm_ms": 0,
            })
            return

        mode = "ANSWER" if top_score >= HIGH_CONFIDENCE else "ANSWER_WITH_CAUTION"
        system_prompt = build_system_prompt(mode, context)

        # Stream from OpenAI
        llm_start = time.time()
        full_reply = ""

        stream = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message}
            ],
            max_tokens=300, temperature=0.3,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_reply += delta.content
                yield f"data: {json.dumps({'type': 'token', 'content': delta.content})}\n\n"

        llm_ms = int((time.time() - llm_start) * 1000)

        # Detect if LLM refused (off-topic despite high retrieval score)
        actual_mode = "LLM_REJECT" if detect_llm_reject(full_reply) else mode

        # Send done signal with timing
        yield f"data: {json.dumps({'type': 'done', 'mode': actual_mode, 'llm_ms': llm_ms})}\n\n"

        # Log the full interaction
        log_interaction({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": req.message, "reply": full_reply, "mode": actual_mode,
            "top_score": top_score,
            "top_chunk_id": debug_chunks[0]["id"] if debug_chunks else None,
            "retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
        })

    return StreamingResponse(generate(), media_type="text/event-stream")