import os
import sys
import pytest

# Ensure package parent is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raglab.wips.vd_facade_1 import langchain_integration

LANGCHAIN_AVAILABLE = getattr(langchain_integration, 'LANGCHAIN_AVAILABLE', False)
QDRANT_AVAILABLE = getattr(langchain_integration, 'QDRANT_AVAILABLE', False)


@pytest.mark.skipif(
    not (LANGCHAIN_AVAILABLE and QDRANT_AVAILABLE),
    reason="LangChain or Qdrant not available",
)
def test_qdrant_store_smoke():
    """Integration smoke test for Qdrant-backed vector store.

    This test will:
    - create a VectorSearchEngine using langchain_recursive, langchain_hf, and langchain_qdrant
    - add a small document mapping
    - perform a search and assert results are returned

    The Qdrant connection parameters are read from environment variables prefixed
    with `VD_FACADE_QDRANT_*` (host, port, collection, api key).
    """
    from raglab.wips.vd_facade_1.interface import VectorSearchEngine

    # Use env overrides if present
    host = os.getenv('VD_FACADE_QDRANT_HOST', 'localhost')
    port = int(os.getenv('VD_FACADE_QDRANT_PORT', '6333'))
    collection = os.getenv('VD_FACADE_QDRANT_COLLECTION', 'vd_facade_test')

    engine = VectorSearchEngine(
        segmenter='langchain_recursive',
        embedder='langchain_hf',
        vector_store={
            'langchain_qdrant': {
                'host': host,
                'port': port,
                'collection_name': collection,
            }
        },
    )

    # clear any prior data
    engine.clear()

    docs = {
        'd1': 'Qdrant provides vector search over an HTTP API',
        'd2': 'This document talks about LangChain and embeddings.',
    }
    engine.add_documents(docs)

    # do a text search
    results = engine.search('vector search', k=3)
    print('Qdrant search results:', results)

    assert isinstance(results, list)
    assert results
