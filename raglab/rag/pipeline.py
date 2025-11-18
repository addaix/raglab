"""RAG pipeline for content processing, vectorization, and vector DB updates."""

from typing import Any, Dict, List, Optional, Callable
from collections.abc import Mapping
from pathlib import Path
import logging

from .types import ContentMapping, UpdateTimeMapping, ContentKey
from .sources import FolderSource, MultiSource
from .refresh import RefreshMappingManager, SimpleRefreshStrategy

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the RAG pipeline: segmentation, vectorization, and vector DB updates.

    The pipeline:
    1. Sources → Content Mapping (with extension-based decoding)
    2. Content Mapping → Segments (via segmentation)
    3. Segments → Vectors (via embedding)
    4. Vectors → Vector DB (via indexing)
    5. Refresh logic determines what needs updating
    """

    def __init__(
        self,
        source: FolderSource | MultiSource,
        segmenter: Callable[[str], List[str]] | None = None,
        embedder: Callable[[List[str]], List[List[float]]] | None = None,
        vector_store: Any | None = None,
        refresh_manager: RefreshMappingManager | None = None,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            source: Content source (FolderSource or MultiSource)
            segmenter: Function to segment text into chunks
            embedder: Function to embed text chunks into vectors
            vector_store: Vector database store
            refresh_manager: Manager for refresh logic
            chunk_size: Size of text chunks for segmentation
            chunk_overlap: Overlap between chunks
        """
        self.source = source
        self.segmenter = segmenter or self._default_segmenter(chunk_size, chunk_overlap)
        self.embedder = embedder
        self.vector_store = vector_store
        self.refresh_manager = refresh_manager or RefreshMappingManager()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Track processed content
        self._segment_mapping: Dict[str, List[str]] = {}
        self._vector_mapping: Dict[str, List[float]] = {}

    def _default_segmenter(self, chunk_size: int, chunk_overlap: int) -> Callable[[str], List[str]]:
        """Create a default segmenter using langchain's RecursiveCharacterTextSplitter."""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )

            def segment(text: str) -> List[str]:
                return text_splitter.split_text(text)

            return segment
        except ImportError:
            logger.warning("langchain not installed, using simple segmentation")
            return self._simple_segmenter(chunk_size, chunk_overlap)

    def _simple_segmenter(self, chunk_size: int, chunk_overlap: int) -> Callable[[str], List[str]]:
        """Simple character-based segmenter."""
        def segment(text: str) -> List[str]:
            if len(text) <= chunk_size:
                return [text]

            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start = end - chunk_overlap
            return chunks

        return segment

    def segment_content(self, content_mapping: ContentMapping) -> Dict[ContentKey, List[str]]:
        """
        Segment content into chunks.

        Args:
            content_mapping: Mapping of content keys to content

        Returns:
            Mapping of content keys to lists of segments
        """
        segment_mapping = {}

        for key, content in content_mapping.items():
            try:
                segments = self.segmenter(content)
                segment_mapping[key] = segments
            except Exception as e:
                logger.error(f"Error segmenting {key}: {e}")
                segment_mapping[key] = [content]  # Fallback to full content

        return segment_mapping

    def vectorize_segments(
        self,
        segment_mapping: Dict[ContentKey, List[str]]
    ) -> Dict[str, List[float]]:
        """
        Vectorize segments using the embedder.

        Args:
            segment_mapping: Mapping of content keys to segments

        Returns:
            Mapping of segment keys to vectors
        """
        if self.embedder is None:
            logger.warning("No embedder configured, skipping vectorization")
            return {}

        vector_mapping = {}
        all_segments = []
        segment_keys = []

        # Flatten segments for batch processing
        for content_key, segments in segment_mapping.items():
            for idx, segment in enumerate(segments):
                segment_key = f"{content_key}::segment_{idx}"
                all_segments.append(segment)
                segment_keys.append(segment_key)

        try:
            # Batch embed all segments
            vectors = self.embedder(all_segments)

            # Map back to segment keys
            for segment_key, vector in zip(segment_keys, vectors):
                vector_mapping[segment_key] = vector

        except Exception as e:
            logger.error(f"Error during vectorization: {e}")

        return vector_mapping

    def update_vector_store(
        self,
        vector_mapping: Dict[str, List[float]],
        segment_mapping: Dict[ContentKey, List[str]],
        keys_to_refresh: set[ContentKey],
        keys_to_delete: set[ContentKey]
    ) -> None:
        """
        Update the vector store with new/modified vectors.

        Args:
            vector_mapping: Mapping of segment keys to vectors
            segment_mapping: Mapping of content keys to segments
            keys_to_refresh: Set of content keys that need refreshing
            keys_to_delete: Set of content keys that need deletion
        """
        if self.vector_store is None:
            logger.warning("No vector store configured, skipping update")
            return

        try:
            # Delete old entries
            for key in keys_to_delete:
                self._delete_from_store(key)

            # Update/add new entries
            for key in keys_to_refresh:
                if key in segment_mapping:
                    self._upsert_to_store(key, segment_mapping[key], vector_mapping)

        except Exception as e:
            logger.error(f"Error updating vector store: {e}")

    def _delete_from_store(self, content_key: ContentKey) -> None:
        """Delete all segments for a content key from the vector store."""
        # This depends on the vector store implementation
        # For now, we'll assume it has a delete method
        if hasattr(self.vector_store, 'delete'):
            try:
                # Delete by prefix
                self.vector_store.delete(prefix=content_key)
            except Exception as e:
                logger.error(f"Error deleting {content_key}: {e}")

    def _upsert_to_store(
        self,
        content_key: ContentKey,
        segments: List[str],
        vector_mapping: Dict[str, List[float]]
    ) -> None:
        """Upsert segments and vectors for a content key to the vector store."""
        # First, delete existing entries
        self._delete_from_store(content_key)

        # Then, add new entries
        for idx, segment in enumerate(segments):
            segment_key = f"{content_key}::segment_{idx}"
            if segment_key in vector_mapping:
                vector = vector_mapping[segment_key]

                if hasattr(self.vector_store, 'add'):
                    try:
                        self.vector_store.add(
                            key=segment_key,
                            vector=vector,
                            metadata={
                                'content_key': content_key,
                                'segment_index': idx,
                                'text': segment
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error adding {segment_key}: {e}")

    def run(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            force_refresh: If True, refresh all content regardless of changes

        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            'content_keys': 0,
            'segments': 0,
            'vectors': 0,
            'refreshed': 0,
            'deleted': 0,
            'new': 0,
            'modified': 0,
        }

        # Step 1: Get current content and update times
        logger.info("Fetching content from source...")
        content_mapping = self.source.get_content_mapping()
        update_times = self.source.get_update_times()

        stats['content_keys'] = len(content_mapping)

        # Step 2: Determine what needs refreshing
        if force_refresh:
            keys_to_refresh = set(content_mapping.keys())
            keys_to_delete = set()
            new_keys = set(content_mapping.keys())
            modified_keys = set()
        else:
            logger.info("Determining refresh requirements...")
            keys_to_refresh = self.refresh_manager.get_keys_to_refresh(update_times)
            keys_to_delete = self.refresh_manager.get_keys_to_delete(update_times)
            new_keys = self.refresh_manager.get_new_keys(update_times)
            modified_keys = self.refresh_manager.get_modified_keys(update_times)

        stats['refreshed'] = len(keys_to_refresh)
        stats['deleted'] = len(keys_to_delete)
        stats['new'] = len(new_keys)
        stats['modified'] = len(modified_keys)

        # Step 3: Segment content that needs refreshing
        if keys_to_refresh:
            logger.info(f"Segmenting {len(keys_to_refresh)} content items...")
            content_to_segment = {k: v for k, v in content_mapping.items() if k in keys_to_refresh}
            segment_mapping = self.segment_content(content_to_segment)
            self._segment_mapping.update(segment_mapping)

            total_segments = sum(len(segments) for segments in segment_mapping.values())
            stats['segments'] = total_segments
            logger.info(f"Created {total_segments} segments")

            # Step 4: Vectorize segments
            if self.embedder:
                logger.info("Vectorizing segments...")
                vector_mapping = self.vectorize_segments(segment_mapping)
                self._vector_mapping.update(vector_mapping)
                stats['vectors'] = len(vector_mapping)
                logger.info(f"Created {len(vector_mapping)} vectors")
            else:
                vector_mapping = {}

            # Step 5: Update vector store
            if self.vector_store:
                logger.info("Updating vector store...")
                self.update_vector_store(
                    vector_mapping,
                    segment_mapping,
                    keys_to_refresh,
                    keys_to_delete
                )
                logger.info("Vector store updated")

        # Step 6: Update refresh manager's stored times
        self.refresh_manager.update_times(update_times)

        logger.info(f"Pipeline complete: {stats}")
        return stats

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant content.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results with metadata
        """
        if self.vector_store is None:
            logger.warning("No vector store configured")
            return []

        try:
            if hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(query, k=k)
                return results
            else:
                logger.warning("Vector store does not support search")
                return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_stats(self) -> Dict[str, int]:
        """Get current pipeline statistics."""
        return {
            'total_segments': len(self._segment_mapping),
            'total_vectors': len(self._vector_mapping),
            'previous_content_keys': len(self.refresh_manager.previous_times),
        }
