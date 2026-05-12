"""
Kategorie 2a: Unit Tests
Tests individual functions without any server or API calls.
"""

import time

# ─── Tokenizer Tests ────────────────────────────────────────

class TestTokenize:
    def test_basic_tokenization(self):
        from main import tokenize

        tokens = tokenize("Wie melde ich mich für Prüfungen an")
        assert "prüfungen" in tokens
        assert "melde" in tokens
        # Stopwords should be removed
        assert "ich" not in tokens
        assert "mich" not in tokens
        assert "für" not in tokens
        assert "wie" not in tokens

    def test_empty_string(self):
        from main import tokenize

        assert tokenize("") == []

    def test_removes_german_stopwords(self):
        from main import tokenize

        tokens = tokenize("der die das ein eine")
        assert tokens == []

    def test_lowercases(self):
        from main import tokenize

        tokens = tokenize("BAföG CAMPO StudOn")
        assert "bafög" in tokens
        assert "campo" in tokens
        assert "studon" in tokens

    def test_handles_special_characters(self):
        from main import tokenize

        tokens = tokenize("Prüfungs-Anmeldung über Campo!")
        assert "prüfungs" in tokens
        assert "anmeldung" in tokens
        assert "campo" in tokens


# ─── LLM Reject Detection Tests ────────────────────────────

class TestDetectLlmReject:
    def test_detects_standard_reject(self):
        from main import detect_llm_reject

        assert detect_llm_reject("Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen") is True

    def test_detects_reject_with_emoji(self):
        from main import detect_llm_reject

        assert detect_llm_reject("Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen 😊") is True

    def test_does_not_reject_normal_answer(self):
        from main import detect_llm_reject

        assert detect_llm_reject("Du kannst BAföG beim Studierendenwerk beantragen.") is False

    def test_does_not_reject_empty(self):
        from main import detect_llm_reject

        assert detect_llm_reject("") is False

    def test_case_insensitive(self):
        from main import detect_llm_reject

        assert detect_llm_reject("ICH KANN DIR NUR BEI FRAGEN RUND UMS STUDIUM an der WiSo helfen") is True


# ─── Rate Limiter Tests ─────────────────────────────────────

class TestRateLimiter:
    def test_allows_under_limit(self):
        from main import RateLimiter

        limiter = RateLimiter()
        for _ in range(19):
            assert limiter.is_allowed("1.2.3.4") is True

    def test_blocks_over_limit(self):
        import utils
        from main import RateLimiter

        # Temporarily lower the limit for testing
        original = utils.RATE_LIMIT_MAX
        utils.RATE_LIMIT_MAX = 5
        try:
            limiter = RateLimiter()
            for _ in range(5):
                limiter.is_allowed("1.2.3.4")
            assert limiter.is_allowed("1.2.3.4") is False
        finally:
            utils.RATE_LIMIT_MAX = original

    def test_different_ips_independent(self):
        import utils
        from main import RateLimiter

        original = utils.RATE_LIMIT_MAX
        utils.RATE_LIMIT_MAX = 3
        try:
            limiter = RateLimiter()
            for _ in range(3):
                limiter.is_allowed("1.1.1.1")
            # IP 1 is exhausted
            assert limiter.is_allowed("1.1.1.1") is False
            # IP 2 is fresh
            assert limiter.is_allowed("2.2.2.2") is True
        finally:
            utils.RATE_LIMIT_MAX = original

    def test_remaining_count(self):
        import utils
        from main import RateLimiter

        original = utils.RATE_LIMIT_MAX
        utils.RATE_LIMIT_MAX = 10
        try:
            limiter = RateLimiter()
            assert limiter.remaining("5.5.5.5") == 10
            for _ in range(3):
                limiter.is_allowed("5.5.5.5")
            assert limiter.remaining("5.5.5.5") == 7
        finally:
            utils.RATE_LIMIT_MAX = original

    def test_window_expires(self):
        import utils
        from main import RateLimiter

        original_max = utils.RATE_LIMIT_MAX
        original_window = utils.RATE_LIMIT_WINDOW
        utils.RATE_LIMIT_MAX = 2
        utils.RATE_LIMIT_WINDOW = 1  # 1 second window
        try:
            limiter = RateLimiter()
            limiter.is_allowed("9.9.9.9")
            limiter.is_allowed("9.9.9.9")
            assert limiter.is_allowed("9.9.9.9") is False
            # Wait for window to expire
            time.sleep(1.1)
            assert limiter.is_allowed("9.9.9.9") is True
        finally:
            utils.RATE_LIMIT_MAX = original_max
            utils.RATE_LIMIT_WINDOW = original_window


# ─── Message ID Tests ───────────────────────────────────────

class TestMessageId:
    def test_format(self):
        from main import generate_message_id

        mid = generate_message_id()
        assert mid.startswith("msg_")
        assert len(mid) > 10

    def test_unique(self):
        from main import generate_message_id

        ids = set()
        for _ in range(10):
            ids.add(generate_message_id())
            time.sleep(0.002)
        assert len(ids) >= 9

# ─── Query Rewriting Trigger Tests ──────────────────────────
# These test whether rewriting WOULD be triggered, not the actual rewrite
# (that needs OpenAI API)

class TestRewriteTrigger:
    def test_no_rewrite_without_history(self):
        from main import rewrite_query

        result = rewrite_query("Wie melde ich mich für Prüfungen an", [])
        assert result == "Wie melde ich mich für Prüfungen an"

    def test_no_rewrite_long_standalone_question(self):
        from main import rewrite_query

        result = rewrite_query(
            "Wie kann ich mich für die Prüfungen im Wintersemester anmelden bitte",
            [{"role": "user", "content": "hallo"}]
        )
        # Long question without context indicators → returned unchanged
        assert result == "Wie kann ich mich für die Prüfungen im Wintersemester anmelden bitte"

    def test_short_question_without_indicators_skipped(self):
        from utils import needs_rewrite
        # 5+ words without indicators = no rewrite triggered
        assert needs_rewrite("Wann beginnt die Vorlesungszeit genau", [{"role": "user", "content": "hallo"}]) is False


# ─── Threshold Logic Tests ──────────────────────────────────

class TestThresholds:
    def test_low_confidence_value(self):
        from main import LOW_CONFIDENCE

        assert 0 < LOW_CONFIDENCE < 1
        assert LOW_CONFIDENCE <= 0.7  # shouldn't be too high

    def test_high_confidence_value(self):
        from main import HIGH_CONFIDENCE

        assert HIGH_CONFIDENCE > 0.5
        assert HIGH_CONFIDENCE <= 0.9

    def test_high_above_low(self):
        from main import HIGH_CONFIDENCE, LOW_CONFIDENCE

        assert HIGH_CONFIDENCE > LOW_CONFIDENCE
