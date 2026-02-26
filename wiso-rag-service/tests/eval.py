import json
import time
import os
import re
import chromadb
import ollama
from rank_bm25 import BM25Okapi
from datetime import datetime

# --- Config (mirror main.py settings) ---
EMBED_MODEL = "nomic-embed-text"
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = 1 - VECTOR_WEIGHT
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))
LOW_CONFIDENCE = 0.65

# --- Setup ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("faq")

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
}

def tokenize(text: str) -> list[str]:
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in GERMAN_STOPWORDS]

# Build BM25 index
all_docs = collection.get(include=["documents"])
all_ids = all_docs["ids"]
all_texts = all_docs["documents"]
tokenized = [tokenize(doc) for doc in all_texts]
bm25_index = BM25Okapi(tokenized)

def retrieve(question: str, n_results: int = 10):
    """Hybrid retrieval — mirrors main.py logic."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=question)
    embedding = response["embedding"]

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
        combined.append({"id": cid, "combined_score": final})

    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    return combined[:n_results]

def run_eval(dataset_path: str = "tests/eval_dataset.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    hit1 = hit3 = hit5 = 0
    mrr_sum = 0.0
    off_topic_correct = 0
    off_topic_total = 0
    failures = []
    category_stats = {}

    print(f"\n{'='*60}")
    print(f"  WiSo Chatbot Retrieval Eval")
    print(f"  {len(test_cases)} test cases | vector_weight={VECTOR_WEIGHT} | bm25_weight={BM25_WEIGHT}")
    print(f"{'='*60}\n")

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        expected = tc["expected_chunk_id"]
        category = tc["category"]

        # Init category stats
        if category not in category_stats:
            category_stats[category] = {"total": 0, "hit1": 0, "hit3": 0, "hit5": 0}
        category_stats[category]["total"] += 1

        retrieved = retrieve(query, n_results=10)
        retrieved_ids = [r["id"] for r in retrieved]
        top_score = retrieved[0]["combined_score"] if retrieved else 0

        # Off-topic handling
        if expected is None:
            off_topic_total += 1
            is_correct = top_score < LOW_CONFIDENCE
            if is_correct:
                off_topic_correct += 1
                status = "✅"
            else:
                status = "❌"
                failures.append(
                    f"  {status} \"{query}\" → off-topic but top_score={top_score:.4f} (threshold={LOW_CONFIDENCE})"
                )
            print(f"  [{i+1:2d}/{len(test_cases)}] {status} \"{query}\" [off-topic] score={top_score:.4f}")
            continue

        # Find rank of expected chunk
        rank = None
        for j, rid in enumerate(retrieved_ids):
            if rid == expected:
                rank = j + 1
                break

        if rank == 1:
            hit1 += 1
            category_stats[category]["hit1"] += 1
        if rank and rank <= 3:
            hit3 += 1
            category_stats[category]["hit3"] += 1
        if rank and rank <= 5:
            hit5 += 1
            category_stats[category]["hit5"] += 1
        if rank:
            mrr_sum += 1.0 / rank

        # Status display
        if rank == 1:
            status = "✅"
        elif rank and rank <= 3:
            status = "🟡"
        elif rank and rank <= 5:
            status = "🟠"
        else:
            status = "❌"

        rank_str = f"rank #{rank}" if rank else "NOT FOUND"
        got_str = retrieved_ids[0] if retrieved_ids else "none"
        print(f"  [{i+1:2d}/{len(test_cases)}] {status} \"{query}\" → expected {expected}, got {got_str} ({rank_str})")

        if not rank or rank > 3:
            failures.append(
                f"  ❌ \"{query}\" → expected {expected}, got {got_str} ({rank_str})"
            )

    # --- Calculate metrics ---
    n_ranked = len(test_cases) - off_topic_total
    metrics = {
        "hit_at_1": hit1 / n_ranked if n_ranked else 0,
        "hit_at_3": hit3 / n_ranked if n_ranked else 0,
        "hit_at_5": hit5 / n_ranked if n_ranked else 0,
        "mrr": mrr_sum / n_ranked if n_ranked else 0,
        "off_topic_precision": off_topic_correct / off_topic_total if off_topic_total else 0,
    }

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Hit@1:  {hit1}/{n_ranked} ({metrics['hit_at_1']*100:.1f}%)")
    print(f"  Hit@3:  {hit3}/{n_ranked} ({metrics['hit_at_3']*100:.1f}%)")
    print(f"  Hit@5:  {hit5}/{n_ranked} ({metrics['hit_at_5']*100:.1f}%)")
    print(f"  MRR:    {metrics['mrr']:.3f}")
    if off_topic_total:
        print(f"  Off-topic precision: {off_topic_correct}/{off_topic_total} ({metrics['off_topic_precision']*100:.1f}%)")

    print(f"\n  By category:")
    for cat, stats in sorted(category_stats.items()):
        h1 = stats['hit1']
        total = stats['total']
        if cat == "off_topic":
            print(f"    {cat:20s}  precision {off_topic_correct}/{off_topic_total}")
        else:
            print(f"    {cat:20s}  Hit@1 {h1}/{total} ({h1/total*100:.0f}%)")

    if failures:
        print(f"\n  FAILURES:")
        for f in failures:
            print(f)

    # --- Save results ---
    os.makedirs("eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"eval_results/eval_{timestamp}.json"

    save_data = {
        "timestamp": timestamp,
        "config": {
            "vector_weight": VECTOR_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "candidate_pool": CANDIDATE_POOL,
            "low_confidence": LOW_CONFIDENCE,
            "embed_model": EMBED_MODEL,
        },
        "metrics": metrics,
        "category_stats": category_stats,
        "failures": failures,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {result_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_eval()