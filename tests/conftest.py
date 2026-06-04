"""Shared pytest fixtures for all test categories."""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Add project root to path so we can import main, ingest, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Session-level ChromaDB mock ────────────────────────────
# Patches chromadb in sys.modules before main.py is ever imported,
# preventing 'Collection faq does not exist' errors at import time.


@pytest.fixture(scope="session", autouse=True)
def patch_chromadb_globally():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["chunk_0"],
        "documents": ["test doc"],
        "metadatas": [{"original_text": "test doc", "keywords": "test"}],
    }
    mock_collection.query.return_value = {
        "ids": [["chunk_0"]],
        "documents": [["test doc"]],
        "distances": [[0.1]],
        "metadatas": [[{"original_text": "test doc", "keywords": "test"}]],
    }

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_chroma_client

    sys.modules["chromadb"] = mock_chromadb
    sys.modules["chromadb.api"] = MagicMock()
    sys.modules["chromadb.errors"] = MagicMock()

    yield mock_chromadb

    for key in list(sys.modules.keys()):
        if key.startswith("chromadb"):
            del sys.modules[key]


# ─── Shared fixtures ─────────────────────────────────────────


@pytest.fixture
def sample_history():
    """Sample conversation history for query rewriting tests."""
    return [
        {"role": "user", "content": "wie melde ich mich für bafög an"},
        {"role": "assistant", "content": "Du kannst BAföG beim Studierendenwerk beantragen."},
    ]


@pytest.fixture
def sample_chunks():
    """Sample debug chunks for testing."""
    return [
        {
            "id": "chunk_0",
            "combined_score": 0.85,
            "vector_score": 0.8,
            "bm25_score": 0.95,
            "preview": "BAföG beim Studierendenwerk...",
        },
        {
            "id": "chunk_1",
            "combined_score": 0.72,
            "vector_score": 0.7,
            "bm25_score": 0.75,
            "preview": "Prüfungsanmeldung über Campo...",
        },
        {
            "id": "chunk_2",
            "combined_score": 0.45,
            "vector_score": 0.5,
            "bm25_score": 0.3,
            "preview": "Mensa Öffnungszeiten...",
        },
    ]
