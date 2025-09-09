import sys
import os
import pytest

# Ensure package root is importable when tests run from this folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raglab.wips.vd_facade_1.interface import VectorSearchEngine


def test_add_and_search_simple():
    engine = VectorSearchEngine()
    engine.clear()
    docs = [
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Vectors represent semantic meaning",
    ]
    engine.add_documents(docs)
    assert engine.size > 0
    results = engine.search("programming", k=2)
    assert len(results) <= 2
    # ensure some result returned
    assert any("Python" in s or "programming" in s for s, _ in results)
