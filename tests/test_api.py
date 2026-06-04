"""
Kategorie 2b: API Tests
Tests FastAPI endpoints using TestClient (no running server needed).
Note: Chat endpoints need OpenAI + ChromaDB, so we test structure only.
"""

from unittest.mock import MagicMock, patch

import pytest


def make_client(log_file=None):
    """Helper: create TestClient with BM25 mocked, optional log file override."""
    import main

    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [0.5]

    main.bm25_index = mock_bm25
    main.all_ids = ["chunk_0"]
    main.all_texts = ["test doc"]
    main.all_originals = ["test doc"]

    if log_file:
        main.LOG_FILE = str(log_file)

    from fastapi.testclient import TestClient

    return TestClient(main.app)


# ─── Static Endpoint Tests ──────────────────────────────────


class TestStaticEndpoints:
    """Tests for endpoints that don't need ChromaDB or OpenAI."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = make_client()

    def test_root_returns_html(self):
        res = self.client.get("/")
        assert res.status_code == 200

    def test_inspector_returns_html(self):
        res = self.client.get("/inspector")
        assert res.status_code == 200

    def test_analytics_returns_html(self):
        res = self.client.get("/analytics")
        assert res.status_code == 200

    def test_lti_get_returns_html(self):
        res = self.client.get("/lti/launch")
        assert res.status_code == 200

    def test_lti_post_wrong_key(self):
        res = self.client.post("/lti/launch", data={"oauth_consumer_key": "wrong"})
        assert res.status_code == 403


# ─── Logs Endpoint Tests ────────────────────────────────────


class TestLogsEndpoints:
    @pytest.fixture(autouse=True)
    def setup_client(self, tmp_path):
        self.log_file = tmp_path / "test_log.jsonl"
        self.log_file.write_text(
            '{"timestamp":"2026-01-01T00:00:00","question":"test","reply":"answer","mode":"ANSWER","top_score":0.85,"retrieval_ms":100,"llm_ms":500}\n'
            '{"timestamp":"2026-01-01T00:01:00","question":"joke","reply":"rejected","mode":"REJECT","top_score":0.3,"retrieval_ms":50,"llm_ms":0}\n'
            '{"timestamp":"2026-01-01T00:02:00","type":"feedback","message_id":"msg_123","rating":"up"}\n'
        )
        self.client = make_client(log_file=self.log_file)

    def test_logs_returns_list(self):
        res = self.client.get("/logs")
        assert res.status_code == 200
        data = res.json()
        assert "logs" in data
        assert "total" in data

    def test_logs_excludes_feedback(self):
        res = self.client.get("/logs")
        data = res.json()
        for log in data["logs"]:
            assert log.get("type") != "feedback"

    def test_logs_filter_by_mode(self):
        res = self.client.get("/logs?mode=REJECT")
        data = res.json()
        for log in data["logs"]:
            assert log["mode"] == "REJECT"

    def test_stats_returns_metrics(self):
        res = self.client.get("/logs/stats")
        assert res.status_code == 200
        data = res.json()
        assert "total" in data
        assert "modes" in data
        assert "avg_top_score" in data
        assert "feedback_up" in data
        assert "feedback_down" in data
        assert "satisfaction_rate" in data

    def test_stats_counts_feedback(self):
        res = self.client.get("/logs/stats")
        data = res.json()
        assert data["feedback_up"] == 1
        assert data["feedback_down"] == 0


# ─── Feedback Endpoint Tests ────────────────────────────────


class TestFeedbackEndpoint:
    @pytest.fixture(autouse=True)
    def setup_client(self, tmp_path):
        self.log_file = tmp_path / "test_log.jsonl"
        self.log_file.write_text("")
        self.client = make_client(log_file=self.log_file)

    def test_feedback_up(self):
        res = self.client.post("/feedback", json={"message_id": "msg_123", "rating": "up"})
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_feedback_down(self):
        res = self.client.post("/feedback", json={"message_id": "msg_456", "rating": "down"})
        assert res.status_code == 200

    def test_feedback_written_to_log(self):
        self.client.post("/feedback", json={"message_id": "msg_789", "rating": "up"})
        content = self.log_file.read_text()
        assert "msg_789" in content
        assert '"rating": "up"' in content or '"rating":"up"' in content


# ─── Chat Endpoint Structure Tests ──────────────────────────


class TestChatEndpointStructure:
    """Test that chat endpoints accept the right input format."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = make_client()

    def test_chat_rejects_empty_body(self):
        res = self.client.post("/chat", json={})
        assert res.status_code == 422

    def test_chat_stream_rejects_empty_body(self):
        res = self.client.post("/chat/stream", json={})
        assert res.status_code == 422