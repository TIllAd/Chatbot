"""
Evaluation Pipeline — WiSo Chatbot
Re-runs test questions against the API and generates a comparison report.

Usage:
  python eval.py                              # Use questions from logs
  python eval.py --file test_questions.json   # Use custom test file
  python eval.py --url http://localhost:8000  # Against local server

Output: eval_results/eval_YYYY-MM-DD_HH-MM.json
"""

import json
import time
import argparse
import os
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    os.system("pip install requests --break-system-packages -q")
    import requests


API_BASE = os.getenv("EVAL_API_URL", "https://chatbot-wiso.de")
RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_questions_from_logs(log_file="logs/chat_log.jsonl"):
    """Extract unique questions from chat logs."""
    questions = []
    seen = set()
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                q = entry["question"].strip().lower()
                if q not in seen and len(q) > 3:
                    seen.add(q)
                    questions.append({
                        "question": entry["question"],
                        "expected_mode": entry.get("mode"),
                        "previous_score": entry.get("top_score"),
                        "previous_reply": entry.get("reply"),
                    })
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
    return questions


def load_questions_from_file(filepath):
    """Load test questions from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for item in data:
        if isinstance(item, str):
            questions.append({"question": item})
        else:
            questions.append(item)
    return questions


def run_question(question: str, api_base: str) -> dict:
    """Send a question to the chat API with debug mode."""
    try:
        res = requests.post(
            f"{api_base}/chat?debug=true",
            json={"message": question},
            timeout=30
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}


def evaluate(questions: list, api_base: str) -> dict:
    """Run all questions and collect results."""
    results = []
    total_start = time.time()

    print(f"\n{'='*70}")
    print(f"  WiSo Chatbot Evaluation — {len(questions)} questions")
    print(f"  API: {api_base}")
    print(f"{'='*70}\n")

    for i, q_data in enumerate(questions):
        question = q_data["question"]
        print(f"[{i+1}/{len(questions)}] {question[:60]}...", end=" ")

        response = run_question(question, api_base)

        if "error" in response:
            print(f"❌ {response['error']}")
            results.append({"question": question, "error": response["error"], **q_data})
            continue

        debug = response.get("debug", {})
        mode = debug.get("mode", "unknown")
        top_score = debug.get("top_score", 0)
        retrieval_ms = debug.get("retrieval_ms", 0)
        llm_ms = debug.get("llm_ms", 0)
        reply = response.get("reply", "")

        score_delta = None
        if q_data.get("previous_score") is not None:
            score_delta = round(top_score - q_data["previous_score"], 4)

        mode_changed = None
        if q_data.get("expected_mode") is not None:
            mode_changed = mode != q_data["expected_mode"]

        emoji = "🚫" if mode == "REJECT" else ("✅" if mode == "ANSWER" else "⚠️")

        delta_str = ""
        if score_delta is not None:
            if score_delta > 0.01:
                delta_str = f" ↑{score_delta:+.3f}"
            elif score_delta < -0.01:
                delta_str = f" ↓{score_delta:+.3f}"
            else:
                delta_str = " ≈"

        print(f"{emoji} score={top_score:.3f}{delta_str} ({retrieval_ms}ms+{llm_ms}ms)")

        results.append({
            "question": question,
            "reply": reply,
            "mode": mode,
            "top_score": top_score,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "top_chunk_id": debug.get("retrieved_chunks", [{}])[0].get("id") if debug.get("retrieved_chunks") else None,
            "score_delta": score_delta,
            "mode_changed": mode_changed,
            "previous_score": q_data.get("previous_score"),
            "previous_mode": q_data.get("expected_mode"),
            "previous_reply": q_data.get("previous_reply"),
        })

        time.sleep(0.3)

    total_time = time.time() - total_start

    valid = [r for r in results if "error" not in r]
    scores = [r["top_score"] for r in valid]
    modes = {}
    for r in valid:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1

    improved = [r for r in valid if r.get("score_delta") and r["score_delta"] > 0.01]
    regressed = [r for r in valid if r.get("score_delta") and r["score_delta"] < -0.01]
    mode_changes = [r for r in valid if r.get("mode_changed")]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "api_base": api_base,
        "total_questions": len(questions),
        "errors": len(questions) - len(valid),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "min_score": round(min(scores), 4) if scores else 0,
        "max_score": round(max(scores), 4) if scores else 0,
        "modes": modes,
        "avg_retrieval_ms": round(sum(r["retrieval_ms"] for r in valid) / len(valid)) if valid else 0,
        "avg_llm_ms": round(sum(r["llm_ms"] for r in valid) / len(valid)) if valid else 0,
        "total_time_s": round(total_time, 1),
        "improved": len(improved),
        "regressed": len(regressed),
        "mode_changes": len(mode_changes),
    }

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Questions:    {summary['total_questions']}")
    print(f"  Avg Score:    {summary['avg_score']:.3f} ({summary['avg_score']*100:.0f}%)")
    print(f"  Min Score:    {summary['min_score']:.3f}")
    print(f"  Max Score:    {summary['max_score']:.3f}")
    print(f"  Modes:        {json.dumps(modes)}")
    print(f"  Avg Speed:    {summary['avg_retrieval_ms']}ms retrieval + {summary['avg_llm_ms']}ms LLM")
    print(f"  Total Time:   {summary['total_time_s']}s")

    if improved or regressed:
        print(f"\n  📈 Improved:   {len(improved)} questions")
        print(f"  📉 Regressed:  {len(regressed)} questions")
        print(f"  🔄 Mode changes: {len(mode_changes)}")

        if regressed:
            print(f"\n  ⚠️  REGRESSIONS:")
            for r in regressed:
                print(f"    • \"{r['question'][:50]}\" {r['previous_score']:.3f} → {r['top_score']:.3f} ({r['score_delta']:+.3f})")

    print(f"{'='*70}\n")

    return {"summary": summary, "results": results}


def save_results(eval_data: dict):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filepath = RESULTS_DIR / f"eval_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="WiSo Chatbot Evaluation Pipeline")
    parser.add_argument("--file", help="JSON file with test questions")
    parser.add_argument("--logs", default="logs/chat_log.jsonl", help="Log file to extract questions from")
    parser.add_argument("--url", default=API_BASE, help="API base URL")
    args = parser.parse_args()

    if args.file:
        questions = load_questions_from_file(args.file)
        print(f"Loaded {len(questions)} questions from {args.file}")
    else:
        questions = load_questions_from_logs(args.logs)
        print(f"Loaded {len(questions)} questions from logs")

    if not questions:
        print("No questions found! Provide --file or ensure logs exist.")
        return

    eval_data = evaluate(questions, args.url)
    save_results(eval_data)


if __name__ == "__main__":
    main()