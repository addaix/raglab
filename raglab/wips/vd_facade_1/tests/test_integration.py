"""Integration tests for the vd_facade_1 facade.

These tests exercise high-level usage patterns a user would try first.

Default components used by the package (when not overridden):
- Segmenter: "simple_sentence" -- heuristic splitter (splits on periods/newlines)
- Embedder: "simple_count" -- tiny fixed embedding: [len(text), unique_chars]
- Vector DB: "memory" -- in-memory dict store that holds segments and vectors

Each test below documents the kind of data source it uses (list vs mapping),
what segmentation is in effect, what vectorization is used, and which vector
store backend is exercised. These are intentionally small and dependency-free
so they can run in CI without external services.
"""

import sys
import os

# Ensure package parent is importable when running tests from this folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raglab.wips.vd_facade_1.interface import VectorSearchEngine
from raglab.wips.vd_facade_1.registry import register_segmenter, register_embedder


def test_example_simple_usage():
    """Integration test: basic usage with defaults.

    Purpose / user story:
      - Quick local semantic search over a small collection of short documents.

    Data source:
      - A Python list of short text documents (strings). This simulates a small
        corpus loaded in-memory (e.g. user-provided list or small file set).

    Segmentation:
      - Uses the default registered segmenter ("simple_sentence"). It splits
        on periods/newlines; because the inputs are short, segments often equal
        the original documents.

    Embedding / vectorization:
      - Uses the default registered embedder ("simple_count"). This produces
        tiny numeric vectors (length and unique-chars) suitable for smoke tests.

    Vector DB / backend:
      - The default in-memory vector store ("memory") is used. This keeps
        everything local and ephemeral and is suitable for examples and tests.
    """
    import sys
    import os

    # Ensure package parent is importable when running tests from this folder
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from raglab.wips.vd_facade_1.interface import VectorSearchEngine
    from raglab.wips.vd_facade_1.registry import register_segmenter, register_embedder


    def test_example_simple_usage():
            """Integration test: basic usage with defaults.

            Purpose / user story:
                - Quick local semantic search over a small collection of short documents.

            Data source:
                - A Python list of short text documents (strings). This simulates a small
                    corpus loaded in-memory (e.g. user-provided list or small file set).

            Segmentation:
                - Uses the default registered segmenter ("simple_sentence"). It splits
                    on periods/newlines; because the inputs are short, segments often equal
                    the original documents.

            Embedding / vectorization:
                - Uses the default registered embedder ("simple_count"). This produces
                    tiny numeric vectors (length and unique-chars) suitable for smoke tests.

            Vector DB / backend:
                - The default in-memory vector store ("memory") is used. This keeps
                    everything local and ephemeral and is suitable for examples and tests.
            """

            import sys
            import os

            # Ensure the package parent is importable when running tests from this folder
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

            from raglab.wips.vd_facade_1.interface import VectorSearchEngine
            from raglab.wips.vd_facade_1.registry import register_segmenter, register_embedder


def test_example_simple_usage():
        """Integration test: basic usage with defaults.

        Purpose / user story:
            - Quick local semantic search over a small collection of short documents.

        Data source:
            - A Python list of short text documents (strings). This simulates a small
                corpus loaded in-memory (e.g. user-provided list or small file set).

        Segmentation:
            - Uses the default registered segmenter ("simple_sentence"). It splits
                on periods/newlines; because the inputs are short, segments often equal
                the original documents.

        Embedding / vectorization:
            - Uses the default registered embedder ("simple_count"). This produces
                tiny numeric vectors (length and unique-chars) suitable for smoke tests.

        Vector DB / backend:
            - The default in-memory vector store ("memory") is used. This keeps
                everything local and ephemeral and is suitable for examples and tests.
        """
        engine = VectorSearchEngine()
        engine.clear()

        docs = [
                "Python is a programming language",
                "Machine learning uses neural networks",
                "Vectors represent semantic meaning",
        ]

        engine.add_documents(docs)

        # size should be at least number of docs (segmenter may split)
        assert engine.size >= len(docs)

        results = engine.search("programming", k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # ensure returned segment text or key present
        assert results


def test_create_with_best_backend_and_search():
        """Integration test: factory method picks a backend and behaves the same.

        Purpose / user story:
            - Demonstrate the `create_with_best_backend` factory which selects the
                best available backend (LangChain-backed stores if available, otherwise
                falls back to the memory store).

        Data source:
            - Small list of strings representing simple documents.

        Segmentation / Embedding:
            - Same defaults as above ("simple_sentence" segmentation,
                "simple_count" embedder).

        Vector DB / backend:
            - This test exercises the selection/factory logic. In this environment
                there are no external vector DBs registered, so the fallback in-memory
                store is used. The test asserts behavior is consistent regardless of
                which backend is chosen.
        """
        engine = VectorSearchEngine.create_with_best_backend()
        engine.clear()
        engine.add_documents(["first doc about cats", "second doc about dogs"])
        assert engine.size >= 1
        res = engine.search("cats", k=1)
        assert isinstance(res, list)


def test_custom_component_registration():
        """Integration test: register and use custom components via the registry.

        Purpose / user story:
            - Show how a user can register application-specific segmenters and
                embedders and then use them with the high-level `VectorSearchEngine`.

        Data source:
            - A mapping (dict) of document id -> text. This is a common real-world
                format (e.g. filenames or record ids mapped to content).

        Custom components registered in the test:
            - `test_sentences` segmenter: simple sentence splitter that returns a
                list of sentence strings.
            - `test_dummy` embedder: deterministic 1-dimensional embedding that
                returns the length of each segment as a numeric vector.

        Vector DB / backend:
            - Still uses the in-memory "memory" store (registered by default). The
                test demonstrates that custom components can be injected while keeping
                the same storage backend.
        """

        @register_segmenter("test_sentences")
        def sentence_segmenter(text: str):
                # very small splitter
                return [s.strip() + "." for s in text.split('.') if s.strip()]

        @register_embedder("test_dummy")
        def dummy_embedder(segments):
                # create predictable 1-D embeddings
                if isinstance(segments, dict):
                        return {k: [len(v)] for k, v in segments.items()}
                return [[len(s)] for s in segments]

        engine = VectorSearchEngine(segmenter="test_sentences", embedder="test_dummy")
        engine.clear()
        engine.add_documents({"a": "hello world. second sentence"})
        assert engine.size >= 1
        out = engine.search("hello", k=2)
        assert out

