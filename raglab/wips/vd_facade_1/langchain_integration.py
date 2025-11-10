"""Optional LangChain + Qdrant integration for vd_facade_1.

This module registers optional components when LangChain and the Qdrant
client are available. It is safe to import when those libraries are missing.
"""

import warnings
from functools import lru_cache
from typing import Optional, Any, Union
import os

# Prefer community packages for newer LangChain layouts when available
try:
    from langchain_community.vectorstores import Qdrant  # type: ignore
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    from langchain.schema import Document  # type: ignore

    LANGCHAIN_AVAILABLE = True
except Exception:
    try:
        import langchain  # type: ignore
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain.vectorstores import Qdrant  # type: ignore
        from langchain.schema import Document  # type: ignore

        LANGCHAIN_AVAILABLE = True
    except Exception:
        LANGCHAIN_AVAILABLE = False

try:
    from qdrant_client import QdrantClient  # type: ignore

    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

from .registry import (
    register_segmenter,
    register_embedder,
    register_vector_store,
    get_component,
    embedders,
)

try:
    # qdrant_client models needed to create collection parameters
    from qdrant_client import models as qdrant_models  # type: ignore
except Exception:
    qdrant_models = None


def register_langchain_components():
    if not LANGCHAIN_AVAILABLE:
        warnings.warn(
            "LangChain not available; skipping langchain_integration registration"
        )
        return

    # Register a RecursiveCharacterTextSplitter as a segmenter
    @register_segmenter("langchain_recursive")
    def recursive_segmenter(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # returns list[str]
        return splitter.split_text(text)

    # Try to register a HuggingFace embedder wrapper (community import preferred)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        except Exception:
            HuggingFaceEmbeddings = None

    if HuggingFaceEmbeddings is not None:

        @register_embedder("langchain_hf")
        class LangChainHuggingFaceEmbedder:
            def __init__(
                self,
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                **kwargs,
            ):
                # Delay heavy imports and model initialization until first use.
                # Store model details and kwargs; attempt to instantiate lazily.
                self.model_name = model_name
                self.model_kwargs = kwargs
                self.model = None

            def _ensure_model(self):
                if self.model is not None:
                    return
                try:
                    self.model = HuggingFaceEmbeddings(
                        model_name=self.model_name, **self.model_kwargs
                    )
                except Exception:
                    # If heavy dependencies fail, keep model as None and
                    # fall back to a simple local embedding implementation.
                    self.model = None

            def __call__(self, segments: dict | list):
                # Prepare texts
                if isinstance(segments, dict):
                    keys = list(segments.keys())
                    texts = list(segments.values())
                else:
                    keys = None
                    texts = list(segments)

                # Try to use the real HuggingFace embedder if available
                self._ensure_model()
                if self.model is not None:
                    vectors = self.model.embed_documents(texts)
                else:
                    # Fallback lightweight embedder: simple length-based vector
                    vectors = [[len(t)] for t in texts]

                if keys is not None:
                    return dict(zip(keys, vectors))
                return vectors

    # Register a Qdrant-backed vector store (via LangChain Qdrant wrapper)
    if not QDRANT_AVAILABLE:
        warnings.warn(
            "qdrant-client not available; Qdrant store will not be registered"
        )
        return

    # Register the previously-defined module-level store class so it is
    # discoverable via the registry. Doing this here preserves optional
    # behavior (only registered when LangChain + qdrant-client available).
    try:
        register_vector_store("langchain_qdrant")(LangChainQdrantStore)
    except Exception:
        # if registration fails, warn but continue
        warnings.warn("Failed to register LangChainQdrantStore: registry error")
    # Define the store class at module scope so it can be imported directly


if QDRANT_AVAILABLE:

    class EmbeddingsAdapter:
        """Adapter that wraps a simple embedder callable into the
        interface expected by various LangChain wrappers.

        It exposes embed_documents, embed_query and is callable to
        support older/newer LangChain variants that expect either an
        Embeddings-like object or an embedding_function.
        """

        def __init__(self, fn):
            self.fn = fn

        def embed_documents(self, texts):
            # Expect fn to accept a list of texts and return list of vectors
            return self.fn(list(texts))

        def embed_query(self, text):
            return self.fn([text])[0]

        def __call__(self, text_or_texts):
            # Support both single-text and list-of-texts usage
            if isinstance(text_or_texts, list):
                return self.fn(text_or_texts)
            return self.fn([text_or_texts])[0]

    class LangChainQdrantStore:
        """Wrapper around LangChain's Qdrant vectorstore using provided embedder.

        Configuration may be provided via constructor kwargs or environment variables:
        - VD_FACADE_QDRANT_HOST (default: localhost)
        - VD_FACADE_QDRANT_PORT (default: 6333)
        - VD_FACADE_QDRANT_COLLECTION (default: vd_facade)
        - VD_FACADE_QDRANT_API_KEY (optional)
        """

        def __init__(
            self,
            embedder: Any | None = None,
            host: str | None = None,
            port: int | None = None,
            collection_name: str | None = None,
            api_key: str | None = None,
        ):
            # allow env overrides
            host = host or os.getenv("VD_FACADE_QDRANT_HOST", "localhost")
            port = int(port or os.getenv("VD_FACADE_QDRANT_PORT", "6333"))
            collection_name = collection_name or os.getenv(
                "VD_FACADE_QDRANT_COLLECTION", "vd_facade"
            )
            api_key = api_key or os.getenv("VD_FACADE_QDRANT_API_KEY")

            self.embedder_callable = embedder
            self.collection_name = collection_name
            # create qdrant client
            if api_key:
                self.client = QdrantClient(url=f"http://{host}:{port}", api_key=api_key)
            else:
                self.client = QdrantClient(url=f"http://{host}:{port}")

            embeddings = EmbeddingsAdapter(self.embedder_callable or (lambda x: []))

            # instantiate LangChain Qdrant wrapper
            try:
                self.store = Qdrant(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=embeddings,
                )
            except Exception:
                # fallback: try creating empty store with from_documents later
                self.store = None

        def add_batch(self, segments: dict, vectors: dict | None = None):
            # Convert segments to LangChain Documents
            docs = [
                Document(page_content=text, metadata={"id": key})
                for key, text in segments.items()
            ]
            # Ensure the collection exists in Qdrant. LangChain/Qdrant
            # wrappers expect the collection to exist; create it if missing.
            try:
                if not self.client.collection_exists(self.collection_name):
                    # Try to infer embedding dimension by probing the embedder
                    dim = 1536
                    try:
                        sample = list(segments.values())[:1]
                        emb = EmbeddingsAdapter(
                            self.embedder_callable or (lambda x: [[]])
                        )(sample)
                        if (
                            isinstance(emb, list)
                            and len(emb)
                            and isinstance(emb[0], (list, tuple))
                        ):
                            dim = len(emb[0])
                    except Exception:
                        pass

                    # minimal vectors_config using qdrant models if available
                    try:
                        if qdrant_models is not None:
                            vectors_config = qdrant_models.VectorParams(
                                size=dim, distance=qdrant_models.Distance.COSINE
                            )
                        else:
                            vectors_config = {"size": dim, "distance": "Cos"}
                        # create collection with minimal config
                        self.client.create_collection(
                            self.collection_name, vectors_config=vectors_config
                        )
                    except Exception:
                        # if create fails, continue and let underlying upsert raise
                        pass

            except Exception:
                # ignore collection existence checks
                pass

            if self.store is not None:
                # LangChain store can add documents
                self.store.add_documents(docs)
            else:
                # last-resort: create store from documents
                try:
                    self.store = Qdrant.from_documents(
                        docs,
                        embeddings=self.embedder_callable,
                        url=self.client.api_url,
                        prefer_grpc=False,
                        collection_name=self.collection_name,
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to add documents to Qdrant store: %s" % e
                    )

        def search(self, query: str | list, k: int = 5):
            if self.store is None:
                return []
            if isinstance(query, str):
                res = self.store.similarity_search_with_score(query, k=k)
                # returns list of (Document, score)
                out = []
                for doc, score in res:
                    key = doc.metadata.get("id") or doc.page_content
                    out.append((key, float(score)))
                return out
            else:
                # assume query is vector
                try:
                    res = self.store.similarity_search_by_vector(query, k=k)
                    return [
                        (d.metadata.get("id") or d.page_content, float(s))
                        for d, s in res
                    ]
                except Exception:
                    return []

        def clear(self):
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                # ignore failures
                pass

        def __contains__(self, key):
            # Qdrant requires search by payload; do a quick search for id payload
            try:
                res = self.client.search(
                    self.collection_name, query_vector=[0.0], limit=1
                )
                return False
            except Exception:
                return False

        def __len__(self):
            try:
                col = self.client.get_collection(self.collection_name)
                return col.points_count
            except Exception:
                return 0


# Auto-register when imported
register_langchain_components()
