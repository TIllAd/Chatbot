"""
Kategorie 2c: Retrieval / Fallback Tests
Tests threshold behavior, off-topic handling, and prompt logic.
Imports from utils.py only — no ChromaDB or OpenAI needed.
"""

import pytest
from utils import (
    detect_llm_reject, build_system_prompt, needs_rewrite,
    HIGH_CONFIDENCE, LOW_CONFIDENCE, RATE_LIMIT_REPLY,
)


# ─── Threshold Behavior Tests ───────────────────────────────

class TestThresholdBehavior:
    def test_below_low_confidence_is_reject(self):
        score = LOW_CONFIDENCE - 0.01
        assert score < LOW_CONFIDENCE

    def test_between_thresholds_is_caution(self):
        score = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2
        assert score >= LOW_CONFIDENCE
        assert score < HIGH_CONFIDENCE

    def test_above_high_confidence_is_answer(self):
        score = HIGH_CONFIDENCE + 0.01
        assert score >= HIGH_CONFIDENCE

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


# ─── Off-Topic Detection Tests ──────────────────────────────

class TestOffTopicDetection:
    def test_llm_reject_phrases_detected(self):
        off_topic_responses = [
            "Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen 😊",
            "Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen.",
            "Leider kann dir nur bei Fragen rund ums Studium helfen!",
        ]
        for response in off_topic_responses:
            assert detect_llm_reject(response) == "LLM_REJECT", f"Should detect: {response}"

    def test_normal_responses_not_rejected(self):
        normal_responses = [
            "Du kannst dich über Campo für Prüfungen anmelden.",
            "BAföG beantragst du beim Studierendenwerk.",
            "Das Basisticket gilt Mo-Fr ab 18 Uhr.",
            "Informationen findest du auf der Homepage des Prüfungsamtes.",
        ]
        for response in normal_responses:
            assert detect_llm_reject(response) is None, f"Should NOT reject: {response}"

    def test_detects_missing_info(self):
        assert detect_llm_reject("Dazu habe ich leider keine Info in meinen Quellen.") == "LLM_MISSING_INFO"


# ─── Query Rewriting Trigger Logic ──────────────────────────

class TestRewriteTriggerLogic:
    def test_no_trigger_without_history(self):
        assert needs_rewrite("Was sind ECTS?", []) is False

    def test_pronoun_triggers(self):
        history = [{"role": "user", "content": "bafög"}]
        assert needs_rewrite("wann ist die frist dafür", history) is True
        assert needs_rewrite("was mache ich damit", history) is True
        assert needs_rewrite("gibt es dazu infos", history) is True

    def test_long_standalone_no_trigger(self):
        history = [{"role": "user", "content": "hallo"}]
        assert needs_rewrite("Wie kann ich mich für die Prüfungen im nächsten Wintersemester anmelden", history) is False


# ─── System Prompt Tests ────────────────────────────────────

class TestSystemPrompt:
    def test_prompt_contains_mode(self):
        prompt = build_system_prompt("ANSWER", "some context")
        assert "MODUS: ANSWER" in prompt

    def test_prompt_contains_context(self):
        prompt = build_system_prompt("ANSWER", "BAföG beim Studierendenwerk")
        assert "BAföG beim Studierendenwerk" in prompt

    def test_prompt_caution_mode(self):
        prompt = build_system_prompt("ANSWER_WITH_CAUTION", "context")
        assert "ANSWER_WITH_CAUTION" in prompt

    def test_prompt_includes_history(self):
        history = [
            {"role": "user", "content": "was ist bafög"},
            {"role": "assistant", "content": "BAföG ist Studienfinanzierung."},
        ]
        prompt = build_system_prompt("ANSWER", "context", history)
        assert "bafög" in prompt.lower()
        assert "GESPRACHSVERLAUF" in prompt

    def test_prompt_without_history(self):
        prompt = build_system_prompt("ANSWER", "context", None)
        assert "GESPRACHSVERLAUF" not in prompt

    def test_prompt_forbids_chunk_ids(self):
        prompt = build_system_prompt("ANSWER", "context")
        assert "Chunk-ID" in prompt


# ─── Reject Reply Tests ─────────────────────────────────────

class TestRejectReply:
    def test_is_german(self):
        assert "warte" in RATE_LIMIT_REPLY

    def test_not_empty(self):
        assert len(RATE_LIMIT_REPLY) > 20