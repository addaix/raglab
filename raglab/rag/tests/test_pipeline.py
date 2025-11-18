"""Integration tests for the RAG pipeline."""

import pytest
import tempfile
import shutil
from pathlib import Path
from raglab.rag import FolderSource, RAGPipeline, RefreshMappingManager


class MockEmbedder:
    """Mock embedder for testing."""

    def __call__(self, segments):
        """Return mock vectors."""
        return [[0.1, 0.2, 0.3] for _ in segments]


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self.data = {}
        self.deleted_keys = set()

    def add(self, key, vector, metadata):
        """Add a vector to the store."""
        self.data[key] = {
            'vector': vector,
            'metadata': metadata
        }

    def delete(self, prefix):
        """Delete vectors with the given prefix."""
        keys_to_delete = [k for k in self.data.keys() if k.startswith(prefix)]
        for key in keys_to_delete:
            del self.data[key]
            self.deleted_keys.add(key)

    def search(self, query, k=5):
        """Mock search."""
        return list(self.data.items())[:k]


class TestRAGPipeline:
    """Test RAG pipeline functionality."""

    @pytest.fixture
    def temp_folder(self):
        """Create a temporary folder with test files."""
        temp_dir = tempfile.mkdtemp()
        folder = Path(temp_dir)

        # Create test files
        (folder / "doc1.txt").write_text("This is the first document. It contains some text.")
        (folder / "doc2.txt").write_text("This is the second document. It has different content.")
        (folder / "doc3.md").write_text("# Markdown\n\nThis is a markdown document.")

        yield folder

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_pipeline_initialization(self, temp_folder):
        """Test pipeline initialization."""
        source = FolderSource(temp_folder)
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=MockVectorStore(),
        )

        assert pipeline.source == source
        assert pipeline.embedder is not None
        assert pipeline.vector_store is not None

    def test_pipeline_run(self, temp_folder):
        """Test running the pipeline."""
        source = FolderSource(temp_folder)
        vector_store = MockVectorStore()
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=vector_store,
            chunk_size=50,
            chunk_overlap=10,
        )

        # Run pipeline
        stats = pipeline.run()

        # Check stats
        assert stats['content_keys'] == 3  # 3 files
        assert stats['segments'] > 0  # Should have segments
        assert stats['vectors'] > 0  # Should have vectors
        assert stats['new'] == 3  # All files are new

        # Check vector store
        assert len(vector_store.data) > 0

    def test_pipeline_incremental_update(self, temp_folder):
        """Test incremental updates."""
        source = FolderSource(temp_folder)
        vector_store = MockVectorStore()
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=vector_store,
        )

        # First run
        stats1 = pipeline.run()
        initial_vector_count = len(vector_store.data)

        assert stats1['new'] == 3

        # Run again without changes
        stats2 = pipeline.run()

        assert stats2['refreshed'] == 0
        assert stats2['new'] == 0
        assert stats2['modified'] == 0

        # Modify a file
        import time
        time.sleep(0.1)  # Ensure timestamp changes
        (temp_folder / "doc1.txt").write_text("Modified content for document 1.")

        # Run again
        stats3 = pipeline.run()

        assert stats3['refreshed'] >= 1
        assert stats3['modified'] >= 1

    def test_pipeline_new_file(self, temp_folder):
        """Test adding a new file."""
        source = FolderSource(temp_folder)
        vector_store = MockVectorStore()
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=vector_store,
        )

        # First run
        stats1 = pipeline.run()
        assert stats1['content_keys'] == 3

        # Add a new file
        (temp_folder / "doc4.txt").write_text("This is a new document.")

        # Run again
        stats2 = pipeline.run()

        assert stats2['content_keys'] == 4
        assert stats2['new'] == 1

    def test_pipeline_delete_file(self, temp_folder):
        """Test deleting a file."""
        source = FolderSource(temp_folder)
        vector_store = MockVectorStore()
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=vector_store,
        )

        # First run
        stats1 = pipeline.run()
        assert stats1['content_keys'] == 3

        # Delete a file
        (temp_folder / "doc1.txt").unlink()

        # Run again
        stats2 = pipeline.run()

        assert stats2['content_keys'] == 2
        assert stats2['deleted'] >= 1

    def test_pipeline_segmentation(self, temp_folder):
        """Test content segmentation."""
        source = FolderSource(temp_folder)
        pipeline = RAGPipeline(
            source=source,
            chunk_size=20,  # Small chunks
            chunk_overlap=5,
        )

        content_mapping = source.get_content_mapping()
        segment_mapping = pipeline.segment_content(content_mapping)

        # Check that content is segmented
        assert len(segment_mapping) > 0
        for key, segments in segment_mapping.items():
            assert isinstance(segments, list)
            assert len(segments) > 0
            # With small chunk size, most docs should be split
            if len(content_mapping[key]) > 20:
                assert len(segments) > 1

    def test_pipeline_vectorization(self, temp_folder):
        """Test segment vectorization."""
        source = FolderSource(temp_folder)
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
        )

        content_mapping = source.get_content_mapping()
        segment_mapping = pipeline.segment_content(content_mapping)
        vector_mapping = pipeline.vectorize_segments(segment_mapping)

        # Check that vectors are created
        assert len(vector_mapping) > 0
        for key, vector in vector_mapping.items():
            assert isinstance(vector, list)
            assert len(vector) == 3  # MockEmbedder returns 3D vectors

    def test_pipeline_without_embedder(self, temp_folder):
        """Test pipeline without embedder."""
        source = FolderSource(temp_folder)
        pipeline = RAGPipeline(
            source=source,
            embedder=None,  # No embedder
        )

        # Should still run but skip vectorization
        stats = pipeline.run()

        assert stats['content_keys'] == 3
        assert stats['segments'] > 0
        assert stats['vectors'] == 0  # No vectors without embedder

    def test_pipeline_without_vector_store(self, temp_folder):
        """Test pipeline without vector store."""
        source = FolderSource(temp_folder)
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=None,  # No vector store
        )

        # Should still run but skip vector store updates
        stats = pipeline.run()

        assert stats['content_keys'] == 3
        assert stats['segments'] > 0
        assert stats['vectors'] > 0

    def test_pipeline_force_refresh(self, temp_folder):
        """Test force refresh."""
        source = FolderSource(temp_folder)
        vector_store = MockVectorStore()
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=vector_store,
        )

        # First run
        stats1 = pipeline.run()

        # Second run with force_refresh
        stats2 = pipeline.run(force_refresh=True)

        # Should refresh all files
        assert stats2['refreshed'] == stats2['content_keys']
        assert stats2['new'] == stats2['content_keys']

    def test_pipeline_get_stats(self, temp_folder):
        """Test getting pipeline stats."""
        source = FolderSource(temp_folder)
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
        )

        # Before running
        stats = pipeline.get_stats()
        assert stats['total_segments'] == 0

        # After running
        pipeline.run()
        stats = pipeline.get_stats()

        assert stats['total_segments'] > 0
        assert stats['previous_content_keys'] > 0

    def test_pipeline_search(self, temp_folder):
        """Test search functionality."""
        source = FolderSource(temp_folder)
        vector_store = MockVectorStore()
        pipeline = RAGPipeline(
            source=source,
            embedder=MockEmbedder(),
            vector_store=vector_store,
        )

        # Run pipeline
        pipeline.run()

        # Search
        results = pipeline.search("test query", k=5)

        # MockVectorStore returns results
        assert isinstance(results, list)
