"""
Rate Limit Test — sends rapid requests to verify the limiter works.

Usage:
  python test_rate_limit.py                          # Against live
  python test_rate_limit.py --url http://localhost:8000  # Against local
"""

import argparse

try:
    import requests
except ImportError:
    import os
    os.system("pip install requests --break-system-packages -q")
    import requests


def test_rate_limit(api_base, total=25, expected_limit=20):
    print(f"\nRate Limit Test: sending {total} requests to {api_base}/chat")
    print(f"Expected: first ~{expected_limit} OK, rest 429\n")

    results = {"ok": 0, "limited": 0, "error": 0}

    for i in range(1, total + 1):
        try:
            res = requests.post(
                f"{api_base}/chat",
                json={"message": "test"},
                timeout=10
            )
            code = res.status_code
            if code == 200:
                results["ok"] += 1
                status = "OK"
            elif code == 429:
                results["limited"] += 1
                status = "RATE LIMITED"
            else:
                results["error"] += 1
                status = f"ERROR {code}"

            print(f"  [{i:2d}/{total}] {status}")

        except Exception as e:
            results["error"] += 1
            print(f"  [{i:2d}/{total}] FAILED: {e}")

    print("\nResults:")
    print(f"  OK (200):       {results['ok']}")
    print(f"  Limited (429):  {results['limited']}")
    print(f"  Errors:         {results['error']}")

    # Verify
    passed = True
    if results["limited"] == 0:
        print("\n  FAIL: No requests were rate limited!")
        passed = False
    if results["ok"] == 0:
        print("\n  FAIL: All requests failed!")
        passed = False
    if results["ok"] > expected_limit + 2:
        print(f"\n  FAIL: Too many OK requests ({results['ok']}), expected ~{expected_limit}")
        passed = False

    if passed:
        print("\n  PASS: Rate limiter works correctly")
    else:
        print("\n  FAIL: Rate limiter not working as expected")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://chatbot-wiso.de")
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    test_rate_limit(args.url, args.requests, args.limit)
