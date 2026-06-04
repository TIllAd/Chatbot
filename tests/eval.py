"""
E2E Evaluation — WiSo Chatbot
Tests the live API against a dataset of questions and expected answers.
Run via CI (needs OPENAI_API_KEY) or manually against the live server.

Usage:
    python eval.py
    python eval.py --url http://localhost:8000
    python eval.py --verbose
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

# --- Config ---
DEFAULT_URL = "https://chatbot-wiso.de"
EVAL_DATASET = Path("eval_dataset.json")
PASS_THRESHOLD = 0.65  # minimum combined_score to count as retrieved
TIMEOUT = 30


# --- Dataset ---
BUILTIN_DATASET = [
    {
        "question": "Wie melde ich mich zu Prüfungen an?",
        "expected_keywords": ["campo", "anmeld", "prüfung"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Was ist StudOn?",
        "expected_keywords": ["studon", "lern", "plattform"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Wo finde ich das Prüfungsamt?",
        "expected_keywords": ["prüfungsamt", "wiso"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Was ist die GOP?",
        "expected_keywords": ["grundlagen", "orientierungsprüfung", "ects"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Wie beantrage ich BAföG?",
        "expected_keywords": ["bafög", "studierendenwerk"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Was ist Campo?",
        "expected_keywords": ["campo", "portal", "prüfung"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Wie viele Semester hat das Bachelorstudium?",
        "expected_keywords": ["semester", "bachelor"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Wie verbinde ich mich mit dem Uni-WLAN?",
        "expected_keywords": ["eduroam", "wlan", "rrze"],
        "expected_mode": ["ANSWER", "ANSWER_WITH_CAUTION"],
    },
    {
        "question": "Was ist das Wetter morgen in Nürnberg?",
        "expected_keywords": [],
        "expected_mode": ["REJECT", "LLM_REJECT"],
    },
    {
        "question": "Kannst du mir einen Witz erzählen?",
        "expected_keywords": [],
        "expected_mode": ["REJECT", "LLM_REJECT"],
    },
]


def load_dataset() -> list[dict]:
    """Load eval dataset from file if it exists, otherwise use builtin."""
    if EVAL_DATASET.exists():
        print(f"Loading dataset from {EVAL_DATASET}")
        with open(EVAL_DATASET, encoding="utf-8") as f:
            return json.load(f)
    print(f"No {EVAL_DATASET} found — using builtin dataset ({len(BUILTIN_DATASET)} questions)")
    return BUILTIN_DATASET


def ask(url: str, question: str) -> dict:
    """Send a question to the chatbot API and return the response."""
    try:
        resp = requests.post(
            f"{url}/chat",
            json={"message": question, "history": []},
            params={"debug": "true"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {url}"}
    except requests.exceptions.Timeout:
        return {"error": f"Timeout after {TIMEOUT}s"}
    except Exception as e:
        return {"error": str(e)}


def evaluate_response(result: dict, case: dict, verbose: bool = False) -> dict:
    """Evaluate a single response against expected output."""
    if "error" in result:
        return {"passed": False, "reason": result["error"], "score": 0}

    reply = result.get("reply", "").lower()
    mode = result.get("debug", {}).get("mode", result.get("mode", "UNKNOWN"))
    top_score = result.get("debug", {}).get("top_score", 0)

    expected_modes = case.get("expected_mode", [])
    expected_keywords = case.get("expected_keywords", [])

    # Check mode
    mode_ok = mode in expected_modes

    # Check keywords in reply
    keywords_ok = True
    missing_keywords = []
    if expected_keywords:
        for kw in expected_keywords:
            if kw.lower() not in reply:
                keywords_ok = False
                missing_keywords.append(kw)

    passed = mode_ok and keywords_ok

    eval_result = {
        "passed": passed,
        "mode": mode,
        "mode_ok": mode_ok,
        "top_score": top_score,
        "keywords_ok": keywords_ok,
        "missing_keywords": missing_keywords,
        "reply_preview": result.get("reply", "")[:100],
    }

    if verbose:
        status = "✓" if passed else "✗"
        print(f"  {status} mode={mode} score={top_score:.3f} keywords_ok={keywords_ok}")
        if missing_keywords:
            print(f"    Missing keywords: {missing_keywords}")
        print(f"    Reply: {eval_result['reply_preview']}...")

    return eval_result


def run_eval(url: str, verbose: bool = False) -> dict:
    """Run full evaluation against the live API."""
    # Health check
    try:
        resp = requests.get(f"{url}/", timeout=5)
        print(f"✓ Server reachable: {url} (status {resp.status_code})")
    except Exception as e:
        print(f"✗ Server not reachable: {url} — {e}")
        return {"error": "Server not reachable", "passed": 0, "total": 0}

    dataset = load_dataset()
    results = []
    passed = 0

    print(f"\nRunning {len(dataset)} eval cases...\n{'=' * 60}")

    for i, case in enumerate(dataset):
        question = case["question"]
        print(f"[{i + 1}/{len(dataset)}] {question[:70]}")

        start = time.time()
        result = ask(url, question)
        elapsed = int((time.time() - start) * 1000)

        eval_result = evaluate_response(result, case, verbose=verbose)
        eval_result["question"] = question
        eval_result["elapsed_ms"] = elapsed
        results.append(eval_result)

        if eval_result["passed"]:
            passed += 1

        # Small delay to avoid hammering the server
        time.sleep(0.5)

    total = len(dataset)
    pass_rate = passed / total if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed ({pass_rate:.1%})")

    # Summary of failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for r in failures:
            print(f"  ✗ {r['question'][:60]}")
            print(f"    mode={r['mode']} (ok={r['mode_ok']}) score={r['top_score']:.3f}")
            if r["missing_keywords"]:
                print(f"    missing: {r['missing_keywords']}")

    # Avg scores
    scores = [r["top_score"] for r in results if r["top_score"] > 0]
    if scores:
        print(f"\nAvg top_score: {sum(scores) / len(scores):.3f}")

    summary = {
        "passed": passed,
        "total": total,
        "pass_rate": round(pass_rate, 4),
        "avg_top_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "failures": len(failures),
        "results": results,
    }

    # CI exit code
    if pass_rate < 0.7:
        print(f"\n✗ EVAL FAILED — pass rate {pass_rate:.1%} below 70% threshold")
    else:
        print(f"\n✓ EVAL PASSED — pass rate {pass_rate:.1%}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E2E Evaluation for WiSo Chatbot")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"API base URL (default: {DEFAULT_URL})")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per case")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    summary = run_eval(url=args.url, verbose=args.verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

    # Exit with error code if eval failed
    if summary.get("pass_rate", 0) < 0.7:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
