"""Vector store integrations for RAG pipeline."""

from typing import List, Dict, Any, Optional, Protocol
from collections.abc import Sequence
from abc import ABC, abstractmethod


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores."""

    def add(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a single vector."""
        ...

    def add_batch(
        self,
        keys: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add multiple vectors in batch."""
        ...

    def delete(self, key: str = None, prefix: str = None) -> None:
        """Delete vector(s) by key or prefix."""
        ...

    def search(
        self,
        query: str | List[float],
        k: int = 5
    ) -> List[tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        ...

    def clear(self) -> None:
        """Clear all vectors."""
        ...


class BaseVectorStore(ABC):
    """Base class for vector store implementations."""

    @abstractmethod
    def add(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a single vector."""
        pass

    @abstractmethod
    def add_batch(
        self,
        keys: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add multiple vectors in batch."""
        pass

    @abstractmethod
    def delete(self, key: str = None, prefix: str = None) -> None:
        """Delete vector(s)."""
        pass

    @abstractmethod
    def search(
        self,
        query: str | List[float],
        k: int = 5
    ) -> List[tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors."""
        pass


class QdrantStore(BaseVectorStore):
    """Qdrant vector store integration."""

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        embedder: Optional[Any] = None,
        vector_size: int = 384,
    ):
        """
        Initialize Qdrant store.

        Args:
            collection_name: Name of the collection
            host: Qdrant host
            port: Qdrant port
            embedder: Embedder for query encoding
            vector_size: Size of vectors
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.qdrant_available = True
        except ImportError:
            self.qdrant_available = False
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

        self.collection_name = collection_name
        self.embedder = embedder
        self.vector_size = vector_size

        self.client = QdrantClient(host=host, port=port)
        self.Distance = Distance
        self.VectorParams = VectorParams
        self.PointStruct = PointStruct

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.VectorParams(
                    size=self.vector_size,
                    distance=self.Distance.COSINE
                ),
            )

    def add(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a single vector."""
        point_id = hash(key) % (2**63)  # Generate numeric ID from key
        payload = {"key": key, **metadata}

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                self.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def add_batch(
        self,
        keys: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add multiple vectors in batch."""
        points = []
        for key, vector, metadata in zip(keys, vectors, metadatas):
            point_id = hash(key) % (2**63)
            payload = {"key": key, **metadata}
            points.append(
                self.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def delete(self, key: str = None, prefix: str = None) -> None:
        """Delete vector(s)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if key:
            # Delete by exact key
            point_id = hash(key) % (2**63)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            )
        elif prefix:
            # Delete by prefix - needs to scroll and filter
            # This is a simplified implementation
            pass

    def search(
        self,
        query: str | List[float],
        k: int = 5
    ) -> List[tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        # If query is string, embed it
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("Embedder required for string queries")
            query_vector = self.embedder([query])[0]
        else:
            query_vector = query

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
        )

        return [
            (hit.payload.get("key"), hit.score, hit.payload)
            for hit in results
        ]

    def clear(self) -> None:
        """Clear all vectors."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()


class ChromaStore(BaseVectorStore):
    """ChromaDB vector store integration."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        embedder: Optional[Any] = None,
    ):
        """
        Initialize Chroma store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedder: Embedder for query encoding
        """
        try:
            import chromadb
            self.chroma_available = True
        except ImportError:
            self.chroma_available = False
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self.collection_name = collection_name
        self.embedder = embedder

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a single vector."""
        self.collection.add(
            ids=[key],
            embeddings=[vector],
            metadatas=[metadata],
        )

    def add_batch(
        self,
        keys: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add multiple vectors in batch."""
        self.collection.add(
            ids=keys,
            embeddings=vectors,
            metadatas=metadatas,
        )

    def delete(self, key: str = None, prefix: str = None) -> None:
        """Delete vector(s)."""
        if key:
            self.collection.delete(ids=[key])
        elif prefix:
            # Get all IDs with prefix
            results = self.collection.get()
            ids_to_delete = [id for id in results['ids'] if id.startswith(prefix)]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)

    def search(
        self,
        query: str | List[float],
        k: int = 5
    ) -> List[tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        # If query is string, embed it
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("Embedder required for string queries")
            query_vector = self.embedder([query])[0]
        else:
            query_vector = query

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
        )

        # Format results
        output = []
        for i in range(len(results['ids'][0])):
            key = results['ids'][0][i]
            distance = results['distances'][0][i]
            score = 1 - distance  # Convert distance to similarity
            metadata = results['metadatas'][0][i]
            output.append((key, score, metadata))

        return output

    def clear(self) -> None:
        """Clear all vectors."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


class FAISSStore(BaseVectorStore):
    """FAISS vector store integration."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "Flat",
        embedder: Optional[Any] = None,
    ):
        """
        Initialize FAISS store.

        Args:
            dimension: Vector dimension
            index_type: FAISS index type ("Flat", "IVFFlat", "HNSW")
            embedder: Embedder for query encoding
        """
        try:
            import faiss
            import numpy as np
            self.faiss_available = True
        except ImportError:
            self.faiss_available = False
            raise ImportError("faiss not installed. Run: pip install faiss-cpu or faiss-gpu")

        self.faiss = faiss
        self.np = np
        self.dimension = dimension
        self.embedder = embedder

        # Create index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Storage for metadata
        self.id_to_key: Dict[int, str] = {}
        self.key_to_id: Dict[str, int] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.next_id = 0

    def add(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a single vector."""
        if key in self.key_to_id:
            # Update existing
            idx = self.key_to_id[key]
            # FAISS doesn't support update easily, so we skip for now
        else:
            # Add new
            idx = self.next_id
            self.next_id += 1
            self.id_to_key[idx] = key
            self.key_to_id[key] = idx

        vector_np = self.np.array([vector], dtype='float32')
        self.index.add(vector_np)
        self.metadata[key] = metadata

    def add_batch(
        self,
        keys: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add multiple vectors in batch."""
        vectors_np = self.np.array(vectors, dtype='float32')
        self.index.add(vectors_np)

        for i, (key, metadata) in enumerate(zip(keys, metadatas)):
            if key not in self.key_to_id:
                idx = self.next_id + i
                self.id_to_key[idx] = key
                self.key_to_id[key] = idx
                self.metadata[key] = metadata

        self.next_id += len(keys)

    def delete(self, key: str = None, prefix: str = None) -> None:
        """Delete vector(s) - not efficiently supported by FAISS."""
        # FAISS doesn't support deletion well
        # Would need to rebuild index
        if key and key in self.key_to_id:
            # Just remove from metadata
            del self.metadata[key]

    def search(
        self,
        query: str | List[float],
        k: int = 5
    ) -> List[tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        # If query is string, embed it
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("Embedder required for string queries")
            query_vector = self.embedder([query])[0]
        else:
            query_vector = query

        query_np = self.np.array([query_vector], dtype='float32')
        distances, indices = self.index.search(query_np, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= self.next_id:
                continue
            key = self.id_to_key.get(idx)
            if key:
                score = 1 / (1 + dist)  # Convert distance to similarity
                metadata = self.metadata.get(key, {})
                results.append((key, score, metadata))

        return results

    def clear(self) -> None:
        """Clear all vectors."""
        self.index.reset()
        self.id_to_key.clear()
        self.key_to_id.clear()
        self.metadata.clear()
        self.next_id = 0


def create_vector_store(
    store_type: str,
    **kwargs
) -> BaseVectorStore:
    """
    Factory function to create vector stores.

    Args:
        store_type: Type of store ("qdrant", "chroma", "faiss")
        **kwargs: Store-specific arguments

    Returns:
        Vector store instance
    """
    if store_type.lower() == "qdrant":
        return QdrantStore(**kwargs)
    elif store_type.lower() == "chroma":
        return ChromaStore(**kwargs)
    elif store_type.lower() == "faiss":
        return FAISSStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
