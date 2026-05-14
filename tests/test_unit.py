"""
Kategorie 2a: Unit Tests
Tests individual functions from utils.py — no ChromaDB or OpenAI needed.
"""

import time
import pytest
from utils import (
    tokenize, detect_llm_reject, RateLimiter, generate_message_id,
    build_system_prompt, needs_rewrite,
    HIGH_CONFIDENCE, LOW_CONFIDENCE, RATE_LIMIT_REPLY,
    RATE_LIMIT_MAX, RATE_LIMIT_WINDOW,
)
import utils


# ─── Tokenizer Tests ────────────────────────────────────────

class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("Wie melde ich mich für Prüfungen an")
        assert "prüfungen" in tokens
        assert "melde" in tokens
        assert "ich" not in tokens
        assert "mich" not in tokens
        assert "für" not in tokens
        assert "wie" not in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_removes_german_stopwords(self):
        assert tokenize("der die das ein eine") == []

    def test_lowercases(self):
        tokens = tokenize("BAföG CAMPO StudOn")
        assert "bafög" in tokens
        assert "campo" in tokens
        assert "studon" in tokens

    def test_handles_special_characters(self):
        tokens = tokenize("Prüfungs-Anmeldung über Campo!")
        assert "prüfungs" in tokens
        assert "anmeldung" in tokens
        assert "campo" in tokens


# ─── LLM Reject Detection Tests ────────────────────────────

class TestDetectLlmReject:
    def test_detects_standard_reject(self):
        assert detect_llm_reject("Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen") == "LLM_REJECT"

    def test_detects_reject_with_emoji(self):
        assert detect_llm_reject("Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen 😊") == "LLM_REJECT"

    def test_does_not_reject_normal_answer(self):
        assert detect_llm_reject("Du kannst BAföG beim Studierendenwerk beantragen.") is None

    def test_does_not_reject_empty(self):
        assert detect_llm_reject("") is None

    def test_case_insensitive(self):
        assert detect_llm_reject("ICH KANN DIR NUR BEI FRAGEN RUND UMS STUDIUM an der WiSo helfen") == "LLM_REJECT"

    def test_detects_missing_info(self):
        assert detect_llm_reject("Dazu habe ich leider keine Info in meinen Quellen. Schau am besten auf der WiSo-Website.") == "LLM_MISSING_INFO"


# ─── Rate Limiter Tests ─────────────────────────────────────

class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = RateLimiter()
        for _ in range(19):
            assert limiter.is_allowed("1.2.3.4") is True

    def test_blocks_over_limit(self):
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
        original = utils.RATE_LIMIT_MAX
        utils.RATE_LIMIT_MAX = 3
        try:
            limiter = RateLimiter()
            for _ in range(3):
                limiter.is_allowed("1.1.1.1")
            assert limiter.is_allowed("1.1.1.1") is False
            assert limiter.is_allowed("2.2.2.2") is True
        finally:
            utils.RATE_LIMIT_MAX = original

    def test_remaining_count(self):
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
        original_max = utils.RATE_LIMIT_MAX
        original_window = utils.RATE_LIMIT_WINDOW
        utils.RATE_LIMIT_MAX = 2
        utils.RATE_LIMIT_WINDOW = 1
        try:
            limiter = RateLimiter()
            limiter.is_allowed("9.9.9.9")
            limiter.is_allowed("9.9.9.9")
            assert limiter.is_allowed("9.9.9.9") is False
            time.sleep(1.1)
            assert limiter.is_allowed("9.9.9.9") is True
        finally:
            utils.RATE_LIMIT_MAX = original_max
            utils.RATE_LIMIT_WINDOW = original_window


# ─── Message ID Tests ───────────────────────────────────────

class TestMessageId:
    def test_format(self):
        mid = generate_message_id()
        assert mid.startswith("msg_")
        assert len(mid) > 10

    def test_unique(self):
        ids = set()
        for _ in range(10):
            ids.add(generate_message_id())
            time.sleep(0.002)  # ensure different timestamps
        assert len(ids) >= 9


# ─── Query Rewriting Trigger Tests ──────────────────────────

class TestNeedsRewrite:
    def test_no_rewrite_without_history(self):
        assert needs_rewrite("Wie melde ich mich an", []) is False

    def test_no_rewrite_long_standalone(self):
        history = [{"role": "user", "content": "hallo"}]
        assert needs_rewrite("Wie kann ich mich für die Prüfungen im Wintersemester anmelden bitte", history) is False

    def test_short_without_indicators_no_rewrite(self):
        # 5+ words without indicators = no rewrite
        history = [{"role": "user", "content": "hallo"}]
        assert needs_rewrite("Wann beginnt die Vorlesungszeit genau", history) is False

    def test_triggers_with_pronoun(self):
        history = [{"role": "user", "content": "was ist bafög"}, {"role": "assistant", "content": "BAföG ist..."}]
        assert needs_rewrite("wann ist die frist dafür", history) is True

    def test_triggers_with_damit(self):
        history = [{"role": "user", "content": "campo"}]
        assert needs_rewrite("was mache ich damit", history) is True


# ─── Threshold Logic Tests ──────────────────────────────────

class TestThresholds:
    def test_low_confidence_value(self):
        assert 0 < LOW_CONFIDENCE < 1
        assert LOW_CONFIDENCE <= 0.6

    def test_high_confidence_value(self):
        assert HIGH_CONFIDENCE > 0.5
        assert HIGH_CONFIDENCE <= 0.9

    def test_high_above_low(self):
        assert HIGH_CONFIDENCE > LOW_CONFIDENCE

    def test_mode_selection_logic(self):
        test_cases = [
            (0.40, "REJECT"),
            (0.54, "REJECT"),
            (0.55, "ANSWER_WITH_CAUTION"),
            (0.60, "ANSWER_WITH_CAUTION"),
            (0.74, "ANSWER_WITH_CAUTION"),
            (0.80, "ANSWER"),
            (0.90, "ANSWER"),
        ]
        for score, expected_mode in test_cases:
            if score < LOW_CONFIDENCE:
                mode = "REJECT"
            elif score >= HIGH_CONFIDENCE:
                mode = "ANSWER"
            else:
                mode = "ANSWER_WITH_CAUTION"
            assert mode == expected_mode, f"Score {score}: expected {expected_mode}, got {mode}"


# ─── System Prompt Tests ────────────────────────────────────

class TestSystemPrompt:
    def test_contains_mode(self):
        prompt = build_system_prompt("ANSWER", "context")
        assert "MODUS: ANSWER" in prompt

    def test_contains_context(self):
        prompt = build_system_prompt("ANSWER", "BAföG beim Studierendenwerk")
        assert "BAföG beim Studierendenwerk" in prompt

    def test_contains_caution_rules(self):
        prompt = build_system_prompt("ANSWER_WITH_CAUTION", "context")
        assert "ANSWER_WITH_CAUTION" in prompt

    def test_includes_history(self):
        history = [
            {"role": "user", "content": "was ist bafög"},
            {"role": "assistant", "content": "BAföG ist Studienfinanzierung."},
        ]
        prompt = build_system_prompt("ANSWER", "context", history)
        assert "bafög" in prompt.lower()

    def test_without_history(self):
        prompt = build_system_prompt("ANSWER", "context", None)
        assert "GESPRACHSVERLAUF" not in prompt

    def test_no_chunk_ids_instruction(self):
        prompt = build_system_prompt("ANSWER", "context")
        assert "Chunk-ID" in prompt


# ─── Rate Limit Reply Tests ─────────────────────────────────

class TestRateLimitReply:
    def test_is_german(self):
        assert "warte" in RATE_LIMIT_REPLY or "Nachricht" in RATE_LIMIT_REPLY

    def test_not_empty(self):
        assert len(RATE_LIMIT_REPLY) > 20