"""Integration with existing raglab components."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ===== Segmentation Integration =====

class RaglabSemanticSegmenter:
    """Use raglab's semantic segmentation."""

    def __init__(self, max_tokens: int = 8000):
        """
        Initialize segmenter.

        Args:
            max_tokens: Maximum tokens per segment
        """
        self.max_tokens = max_tokens

    def __call__(self, text: str) -> List[str]:
        """Segment text using raglab's semantic segmentation."""
        try:
            from raglab.retrieval.segmentation_lib import (
                sentence_splits_ids,
                filtered_sentence_split_ids,
                sentence_splits,
                sentence_embeddings,
                consecutive_cosines,
                segment_keys,
                text_segments,
                sentence_num_tokens,
            )

            # Run segmentation pipeline
            split_ids = sentence_splits_ids(text)
            filtered_ids = filtered_sentence_split_ids(split_ids, text)

            if not filtered_ids:
                return [text]

            sentences = sentence_splits(text, filtered_ids)
            embeddings = sentence_embeddings(sentences)
            cosines = consecutive_cosines(embeddings)
            num_tokens = sentence_num_tokens(sentences)

            seg_keys = segment_keys(
                num_tokens,
                cosines,
                filtered_ids,
                max_tokens=self.max_tokens,
            )

            chunks = text_segments(text, seg_keys)
            return chunks

        except ImportError as e:
            logger.warning(f"Raglab segmentation not available: {e}")
            return [text]
        except Exception as e:
            logger.error(f"Error in semantic segmentation: {e}")
            return [text]


# ===== Vector DB Integration =====

class RaglabVectorDBAdapter:
    """Adapter for raglab's VectorDB/ChunkDB."""

    def __init__(self, chunk_db: Optional[Any] = None, **kwargs):
        """
        Initialize adapter.

        Args:
            chunk_db: ChunkDB instance or None to create new one
            **kwargs: Arguments for ChunkDB constructor
        """
        if chunk_db:
            self.db = chunk_db
        else:
            try:
                from raglab.retrieval.VectorDB import ChunkDB
                self.db = ChunkDB(**kwargs)
            except ImportError:
                raise ImportError("raglab VectorDB not available")

    def add(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a vector."""
        # ChunkDB uses docs, not individual vectors
        # Store metadata as doc
        if hasattr(self.db, 'docs'):
            text = metadata.get('text', '')
            self.db.docs[key] = text

    def add_batch(
        self,
        keys: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add batch of vectors."""
        for key, vector, metadata in zip(keys, vectors, metadatas):
            self.add(key, vector, metadata)

    def delete(self, key: str = None, prefix: str = None) -> None:
        """Delete vectors."""
        if hasattr(self.db, 'docs'):
            if key and key in self.db.docs:
                del self.db.docs[key]
            elif prefix:
                to_delete = [k for k in self.db.docs.keys() if k.startswith(prefix)]
                for k in to_delete:
                    del self.db.docs[k]

    def search(
        self,
        query: str | List[float],
        k: int = 5
    ) -> List[tuple[str, float, Dict[str, Any]]]:
        """Search."""
        if hasattr(self.db, 'search'):
            results = self.db.search(query, k=k)
            # Format results
            return [(r, 0.0, {}) for r in results]
        return []

    def clear(self) -> None:
        """Clear all vectors."""
        if hasattr(self.db, 'docs'):
            self.db.docs.clear()


# ===== Embedder Integration =====

def create_raglab_embedder(embedder_type: str = "openai", **kwargs):
    """
    Create embedder compatible with raglab.

    Args:
        embedder_type: Type of embedder
        **kwargs: Embedder-specific arguments

    Returns:
        Embedder function
    """
    if embedder_type == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = kwargs.get('api_key')
            model = kwargs.get('model', 'text-embedding-3-small')
            dimensions = kwargs.get('dimensions', 512)

            embeddings_model = OpenAIEmbeddings(
                api_key=api_key,
                model=model,
                dimensions=dimensions
            )

            def embedder(texts: List[str]) -> List[List[float]]:
                return embeddings_model.embed_documents(texts)

            return embedder

        except ImportError:
            raise ImportError("langchain-openai not installed")

    elif embedder_type == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
            model_name = kwargs.get('model', 'all-MiniLM-L6-v2')
            model = SentenceTransformer(model_name)

            def embedder(texts: List[str]) -> List[List[float]]:
                embeddings = model.encode(texts)
                return embeddings.tolist()

            return embedder

        except ImportError:
            raise ImportError("sentence-transformers not installed")

    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


# ===== Full Raglab Pipeline =====

def create_raglab_pipeline(
    folder_path: str,
    embedder_config: Optional[Dict[str, Any]] = None,
    use_semantic_segmentation: bool = True,
    **kwargs
):
    """
    Create a RAG pipeline fully integrated with raglab components.

    Args:
        folder_path: Path to folder to index
        embedder_config: Embedder configuration
        use_semantic_segmentation: Use raglab's semantic segmentation
        **kwargs: Additional pipeline arguments

    Returns:
        Configured RAGPipeline
    """
    from .sources import FolderSource
    from .pipeline import RAGPipeline

    # Create source
    source = FolderSource(folder_path=folder_path)

    # Create segmenter
    if use_semantic_segmentation:
        segmenter = RaglabSemanticSegmenter()
        kwargs['segmenter'] = segmenter

    # Create embedder
    if embedder_config:
        embedder = create_raglab_embedder(**embedder_config)
        kwargs['embedder'] = embedder

    # Create vector store adapter
    try:
        from raglab.retrieval.VectorDB import ChunkDB
        chunk_db = ChunkDB(docs={})
        vector_store = RaglabVectorDBAdapter(chunk_db=chunk_db)
        kwargs['vector_store'] = vector_store
    except ImportError:
        logger.warning("ChunkDB not available, skipping vector store")

    # Create pipeline
    pipeline = RAGPipeline(source=source, **kwargs)

    return pipeline
