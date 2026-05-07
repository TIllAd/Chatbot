"""
Kategorie 2b: API Tests
Tests FastAPI endpoints using TestClient (no running server needed).
Note: Chat endpoints need OpenAI + ChromaDB, so we test structure only.
"""

from unittest.mock import MagicMock, patch

import pytest

# ─── Static Endpoint Tests ──────────────────────────────────

class TestStaticEndpoints:
    """Tests for endpoints that don't need ChromaDB or OpenAI."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Create TestClient with mocked ChromaDB."""
        with patch("chromadb.PersistentClient") as mock_chroma:
            # Mock the collection
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "ids": ["chunk_0"],
                "documents": ["test doc"],
                "metadatas": [{"original_text": "test doc", "keywords": "test"}],
            }
            mock_chroma.return_value.get_collection.return_value = mock_collection

            # Mock BM25 to avoid import-time errors
            with patch("main.build_bm25_index", return_value=(MagicMock(), ["chunk_0"], ["test doc"], ["test doc"])):
                from fastapi.testclient import TestClient

                from main import app

                self.client = TestClient(app)

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
        """Create TestClient with a temp log file."""
        log_file = tmp_path / "test_log.jsonl"
        log_file.write_text(
            '{"timestamp":"2026-01-01T00:00:00","question":"test","reply":"answer","mode":"ANSWER","top_score":0.85,"retrieval_ms":100,"llm_ms":500}\n'
            '{"timestamp":"2026-01-01T00:01:00","question":"joke","reply":"rejected","mode":"REJECT","top_score":0.3,"retrieval_ms":50,"llm_ms":0}\n'
            '{"timestamp":"2026-01-01T00:02:00","type":"feedback","message_id":"msg_123","rating":"up"}\n'
        )

        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "ids": ["chunk_0"],
                "documents": ["test doc"],
                "metadatas": [{"original_text": "test doc", "keywords": "test"}],
            }
            mock_chroma.return_value.get_collection.return_value = mock_collection

            with patch("main.build_bm25_index", return_value=(MagicMock(), ["chunk_0"], ["test doc"], ["test doc"])):
                with patch("main.LOG_FILE", str(log_file)):
                    from fastapi.testclient import TestClient

                    from main import app

                    self.client = TestClient(app)

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
        log_file = tmp_path / "test_log.jsonl"
        log_file.write_text("")

        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "ids": ["chunk_0"],
                "documents": ["test doc"],
                "metadatas": [{"original_text": "test doc", "keywords": "test"}],
            }
            mock_chroma.return_value.get_collection.return_value = mock_collection

            with patch("main.build_bm25_index", return_value=(MagicMock(), ["chunk_0"], ["test doc"], ["test doc"])):
                with patch("main.LOG_FILE", str(log_file)):
                    from fastapi.testclient import TestClient

                    from main import app

                    self.client = TestClient(app)
                    self.log_file = log_file

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
    """Test that chat endpoints accept the right input format.
    Actual responses need OpenAI, so we just test input validation."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "ids": ["chunk_0"],
                "documents": ["test doc"],
                "metadatas": [{"original_text": "test doc", "keywords": "test"}],
            }
            mock_chroma.return_value.get_collection.return_value = mock_collection

            with patch("main.build_bm25_index", return_value=(MagicMock(), ["chunk_0"], ["test doc"], ["test doc"])):
                from fastapi.testclient import TestClient

                from main import app

                self.client = TestClient(app)

    def test_chat_rejects_empty_body(self):
        res = self.client.post("/chat", json={})
        assert res.status_code == 422  # Validation error

    def test_chat_stream_rejects_empty_body(self):
        res = self.client.post("/chat/stream", json={})
        assert res.status_code == 422
