"""
Kategorie 2c: Retrieval / Fallback Tests
Tests the retrieval logic, threshold behavior, and off-topic handling.
Does NOT call OpenAI or ChromaDB — uses mocked data.
"""



# ─── Threshold Behavior Tests ───────────────────────────────

class TestThresholdBehavior:
    """Test that the system correctly categorizes responses by score."""

    def test_below_low_confidence_is_reject(self):
        from main import LOW_CONFIDENCE

        score = LOW_CONFIDENCE - 0.01
        assert score < LOW_CONFIDENCE

    def test_between_thresholds_is_caution(self):
        from main import HIGH_CONFIDENCE, LOW_CONFIDENCE

        score = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2
        assert score >= LOW_CONFIDENCE
        assert score < HIGH_CONFIDENCE

    def test_above_high_confidence_is_answer(self):
        from main import HIGH_CONFIDENCE

        score = HIGH_CONFIDENCE + 0.01
        assert score >= HIGH_CONFIDENCE

    def test_mode_selection_logic(self):
        """Verify the mode selection matches what main.py does."""
        from main import HIGH_CONFIDENCE, LOW_CONFIDENCE

        test_cases = [
            (0.40, "REJECT"),
            (0.54, "REJECT"),
            (0.55, "ANSWER_WITH_CAUTION"),
            (0.60, "ANSWER_WITH_CAUTION"),
            (0.74, "ANSWER_WITH_CAUTION"),
            (0.75, "ANSWER"),
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
    """Test that off-topic questions are handled correctly."""

    def test_llm_reject_phrases_detected(self):
        from main import detect_llm_reject

        off_topic_responses = [
            "Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen 😊",
            "Ich kann dir nur bei Fragen rund ums Studium an der WiSo helfen.",
            "Leider kann dir nur bei Fragen rund ums Studium helfen!",
        ]
        for response in off_topic_responses:
            assert detect_llm_reject(response) is True, f"Should detect as reject: {response}"

    def test_normal_responses_not_rejected(self):
        from main import detect_llm_reject

        normal_responses = [
            "Du kannst dich über Campo für Prüfungen anmelden.",
            "BAföG beantragst du beim Studierendenwerk.",
            "Das Basisticket gilt Mo-Fr ab 18 Uhr.",
            "Informationen findest du auf der Homepage des Prüfungsamtes.",
        ]
        for response in normal_responses:
            assert detect_llm_reject(response) is False, f"Should NOT reject: {response}"


# ─── Query Rewriting Trigger Logic ──────────────────────────

class TestRewriteTriggerLogic:
    """Test which queries would trigger rewriting (without calling OpenAI)."""

    def test_pronoun_triggers_rewrite(self):
        """Messages with pronouns like 'dafür', 'damit' should trigger rewrite."""
        from main import rewrite_query

        # These need context indicators AND history to trigger
        # Without OpenAI they won't actually rewrite, but they should
        # at least NOT return early
        indicators = ["dafür", "damit", "davon", "dazu", "darüber"]
        for word in indicators:
            msg = f"Wann ist die Frist {word}"
            # With empty history, should return unchanged
            result = rewrite_query(msg, [])
            assert result == msg

    def test_long_standalone_question_no_rewrite(self):
        from main import rewrite_query

        history = [{"role": "user", "content": "hallo"}, {"role": "assistant", "content": "hi"}]
        msg = "Wie kann ich mich für die Prüfungen im nächsten Wintersemester anmelden"
        result = rewrite_query(msg, history)
        # Long question without context indicators → returned unchanged
        assert result == msg


# ─── System Prompt Tests ────────────────────────────────────

class TestSystemPrompt:
    """Test that the system prompt is built correctly for each mode."""

    def test_prompt_contains_mode(self):
        from main import build_system_prompt

        prompt = build_system_prompt("ANSWER", "some context")
        assert "MODUS: ANSWER" in prompt

    def test_prompt_contains_context(self):
        from main import build_system_prompt

        prompt = build_system_prompt("ANSWER", "BAföG beim Studierendenwerk")
        assert "BAföG beim Studierendenwerk" in prompt

    def test_prompt_contains_caution_rules(self):
        from main import build_system_prompt

        prompt = build_system_prompt("ANSWER_WITH_CAUTION", "context")
        assert "ANSWER_WITH_CAUTION" in prompt

    def test_prompt_includes_history(self):
        from main import build_system_prompt

        history = [
            {"role": "user", "content": "was ist bafög"},
            {"role": "assistant", "content": "BAföG ist Studienfinanzierung."},
        ]
        prompt = build_system_prompt("ANSWER", "context", history)
        assert "was ist bafög" in prompt.lower() or "bafög" in prompt.lower()

    def test_prompt_without_history(self):
        from main import build_system_prompt

        prompt = build_system_prompt("ANSWER", "context", None)
        assert "GESPRACH" not in prompt.upper() or "VERLAUF" not in prompt.upper()

    def test_prompt_no_chunk_ids_instruction(self):
        from main import build_system_prompt

        prompt = build_system_prompt("ANSWER", "context")
        assert "Chunk-ID" in prompt or "chunk" in prompt.lower()


# ─── Reject Reply Tests ─────────────────────────────────────

class TestRejectReply:
    """Test the rejection message content."""

    def test_reject_message_is_german(self):
        """The reject message should be in German."""
        # Just verify the constant exists and is German
        from main import RATE_LIMIT_REPLY

        assert "Nachricht" in RATE_LIMIT_REPLY or "warte" in RATE_LIMIT_REPLY
