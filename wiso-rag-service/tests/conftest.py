"""Shared pytest fixtures for all test categories."""

import os
import sys

import pytest

# Add project root to path so we can import main, ingest, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        {"id": "chunk_0", "combined_score": 0.85, "vector_score": 0.8, "bm25_score": 0.95, "preview": "BAföG beim Studierendenwerk..."},
        {"id": "chunk_1", "combined_score": 0.72, "vector_score": 0.7, "bm25_score": 0.75, "preview": "Prüfungsanmeldung über Campo..."},
        {"id": "chunk_2", "combined_score": 0.45, "vector_score": 0.5, "bm25_score": 0.3, "preview": "Mensa Öffnungszeiten..."},
    ]
