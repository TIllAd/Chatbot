"""
Pure utility functions for the WiSo Chatbot.
No external service dependencies (no ChromaDB, no OpenAI) — safe to import anywhere.
"""

import re
import time
import os
from collections import defaultdict

# --- Config (from env, with defaults) ---
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

HIGH_CONFIDENCE = float(os.getenv("HIGH_CONFIDENCE", "0.75"))
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.55"))

RATE_LIMIT_REPLY = (
    "Du hast zu viele Nachrichten in kurzer Zeit geschickt. "
    "Bitte warte einen Moment und versuche es dann erneut."
)

# --- German Stopwords ---
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
    "gibt", "sollte", "sollten", "würde", "würden", "könnte",
}


# --- Tokenizer ---
def tokenize(text: str) -> list[str]:
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in GERMAN_STOPWORDS]


# --- LLM Refusal Detection ---
LLM_REJECT_PHRASES = [
    "ich kann dir nur bei fragen rund ums studium",
    "nur bei fragen rund ums studium an der wiso",
    "kann dir nur bei fragen",
]

LLM_MISSING_INFO_PHRASES = [
    "dazu habe ich leider keine info",
    "keine info in meinen quellen",
    "schau am besten auf der wiso-website",
]

def detect_llm_reject(reply: str) -> str | None:
    """Returns 'LLM_REJECT' for off-topic, 'LLM_MISSING_INFO' for no-data, or None for normal answers."""
    lower = reply.lower()
    if any(phrase in lower for phrase in LLM_REJECT_PHRASES):
        return "LLM_REJECT"
    if any(phrase in lower for phrase in LLM_MISSING_INFO_PHRASES):
        return "LLM_MISSING_INFO"
    return None


# --- Rate Limiter ---
class RateLimiter:
    """Simple in-memory IP-based rate limiter."""
    def __init__(self):
        self.requests = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        self.requests[ip] = [t for t in self.requests[ip] if now - t < RATE_LIMIT_WINDOW]
        if len(self.requests[ip]) >= RATE_LIMIT_MAX:
            return False
        self.requests[ip].append(now)
        return True

    def remaining(self, ip: str) -> int:
        now = time.time()
        self.requests[ip] = [t for t in self.requests[ip] if now - t < RATE_LIMIT_WINDOW]
        return max(0, RATE_LIMIT_MAX - len(self.requests[ip]))


# --- Message ID ---
def generate_message_id() -> str:
    return f"msg_{int(time.time() * 1000)}"


# --- System Prompt Builder ---
def build_system_prompt(mode: str, context: str, history: list[dict] = None) -> str:
    history_block = ""
    if history:
        recent = history[-6:]
        lines = []
        for msg in recent:
            role = "Student" if msg.get("role") == "user" else "Bot"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\n\nGESPRACHSVERLAUF (fuer Kontext):\n" + "\n".join(lines)

    prompt = (
        f"Du bist der WiSo-Chatbot der FAU Erlangen-Nuernberg. Du hilfst Erstsemester-Studierenden, sich im Studium zurechtzufinden.\n\n"
        f"MODUS: {mode}\n\n"
        "MODUS-REGELN:\n"
        "- ANSWER_WITH_CAUTION: Antworte kurz und fuege eine Rueckfrage hinzu, ob das die richtige Frage war.\n"
        "- ANSWER: Antworte kurz und hilfreich.\n\n"
        "DEIN WICHTIGSTES ZIEL:\n"
        'Hilf Studierenden, die Info SELBST zu finden. Nenne immer die konkrete Quelle oder Anlaufstelle (z.B. "Homepage des Pruefungsamtes", "Campo", "StudOn", "MHB", "RRZE Website"). Nenne NIEMALS Chunk-IDs.\n\n'
        "ANTWORT-REGELN:\n"
        "- Antworte NUR mit Informationen aus den QUELLEN unten.\n"
        "- Erfinde NICHTS dazu. Keine eigenen Informationen, keine Vermutungen.\n"
        "- Antworte auf Deutsch, kurz und freundlich (du-Form).\n"
        "- Beachte den GESPRAECHSVERLAUF unten, um Rueckfragen und Bezuege richtig zu verstehen.\n\n"
        "WENN DIE QUELLEN NICHT AUSREICHEN:\n"
        "- Wenn die Frage zum Studium gehoert aber die QUELLEN keine Antwort enthalten: "
        'Sage: "Dazu habe ich leider keine Info in meinen Quellen. Schau am besten auf der WiSo-Website oder frag die Studienberatung."\n'
        "- Wenn die Frage NICHTS mit dem Studium zu tun hat (Witze, Wetter, Politik, Smalltalk, persoenliche Fragen): "
        '"Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen"\n\n'
        "STUDIEN-RELEVANTE THEMEN (auch ohne Quellen als Studienfrage erkennen):\n"
        "Semesterzeiten, Vorlesungsbeginn, Wintersemester, Sommersemester, Semesterbeitrag, Semesterticket, "
        "Bibliothek, Mensa, Wohnheim, Auslandssemester, Erasmus, Praktikum, Werkstudent, BAfoeg, "
        "Studienberatung, Pruefungsamt, Campo, StudOn, RRZE, FAU, WiSo\n\n"
        "FORMAT:\n"
        "1) Kurze Antwort (2-3 Saetze max)\n"
        "2) Wo du das findest: [konkrete Quelle]"
        f"{history_block}\n\n"
        f"QUELLEN:\n{context}"
    )
    return prompt.strip()


# --- Query Rewrite Trigger Check ---
NEEDS_CONTEXT_INDICATORS = [
    "dafür", "damit", "davon", "dazu", "darüber", "darum",
    "das", "dies", "diese", "diesem", "diesen",
    "er", "sie", "es", "ihm", "ihr",
    "dort", "da", "hier",
    "auch", "noch", "mehr", "weiter",
    "und", "aber",
]

def needs_rewrite(message: str, history: list[dict]) -> bool:
    """Check if a message likely needs query rewriting based on context indicators."""
    if not history:
        return False
    msg_lower = message.lower().strip()
    words = msg_lower.split()
    if len(words) > 8 and not any(w in NEEDS_CONTEXT_INDICATORS for w in words):
        return False
    if len(words) > 4 and not any(w in NEEDS_CONTEXT_INDICATORS for w in words):
        return False
    return True