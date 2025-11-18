"""
RAG (Retrieval-Augmented Generation) module for raglab.

This module provides a comprehensive RAG system with:
- Source management (folders, URLs, databases, APIs, Git repositories)
- Content mappings with extension-based decoding
- Smart chunking strategies (markdown-aware, code-aware, semantic)
- Update time tracking and intelligent refresh strategies
- Full pipeline for segmentation, vectorization, and vector DB updates
- Parallel processing and async operations
- Metadata management and tagging
- Event hooks and callbacks
- State persistence and checkpointing
- Content validation and deduplication
- Query enhancements and hybrid search
- Hierarchical indexing
- Multi-modal support (images, audio)
- Integration with raglab components
- CLI tool
- Configuration management

Example usage:
    >>> from raglab.rag import FolderSource, EnhancedRAGPipeline
    >>>
    >>> # Create a folder source with exclude patterns
    >>> source = FolderSource(
    ...     folder_path="/path/to/docs",
    ...     exclude_patterns=[r"\.git", r"__pycache__"],
    ...     extensions={'.md', '.txt', '.py'}
    ... )
    >>>
    >>> # Create an enhanced pipeline
    >>> pipeline = EnhancedRAGPipeline(
    ...     source=source,
    ...     embedder=your_embedder_function,
    ...     vector_store=your_vector_store,
    ...     enable_metadata=True,
    ...     enable_monitoring=True,
    ... )
    >>>
    >>> # Run the pipeline
    >>> stats = pipeline.run()
    >>>
    >>> # Search
    >>> results = pipeline.search("your query")
"""

# Core types
from .types import (
    ContentKey,
    ContentValue,
    ContentMapping,
    UpdateTimeMapping,
    RefreshMapping,
    SourceKey,
    Timestamp,
    ExcludeFunc,
    ContentDecoder,
    Source,
    RefreshStrategy,
)

# Decoders
from .decoders import (
    ExtensionBasedDecoder,
    text_decoder,
    markdown_decoder,
    python_decoder,
    json_decoder,
    yaml_decoder,
    pdf_decoder,
    docx_decoder,
    DEFAULT_DECODERS,
)

# Sources
from .sources import (
    FolderSource,
    MultiSource,
)

# Advanced sources
from .advanced_sources import (
    URLSource,
    DatabaseSource,
    APISource,
    GitRepoSource,
)

# Async sources
from .async_sources import (
    AsyncFolderSource,
    ParallelBatchProcessor,
    make_async_embedder,
)

# Refresh strategies
from .refresh import (
    SimpleRefreshStrategy,
    ThresholdRefreshStrategy,
    BatchRefreshStrategy,
    RefreshMappingManager,
)

# Chunking
from .chunking import (
    BaseChunker,
    MarkdownAwareChunker,
    CodeAwareChunker,
    SemanticChunker,
    create_chunker,
)

# Vector stores
from .vector_stores import (
    BaseVectorStore,
    QdrantStore,
    ChromaStore,
    FAISSStore,
    create_vector_store,
)

# Pipeline
from .pipeline import RAGPipeline
from .enhanced_pipeline import EnhancedRAGPipeline

# Metadata
from .metadata import (
    MetadataExtractor,
    MetadataManager,
    SegmentMetadata,
    TagManager,
)

# Event hooks
from .hooks import (
    EventType,
    Event,
    EventHandler,
    EventHookManager,
    logging_handler,
    progress_bar_handler,
    statistics_collector,
)

# Utilities
from .utils import (
    DeduplicationStrategy,
    StatePersistence,
    CheckpointManager,
    ContentValidator,
    EmbeddingCache,
)

# Advanced features
from .advanced_features import (
    QueryEnhancer,
    HybridSearch,
    Reranker,
    HierarchicalPipeline,
    PipelineMonitor,
    ImageSource,
    AudioSource,
)

# Configuration
from .config import (
    SourceConfig,
    PipelineConfig,
    EmbedderConfig,
    VectorStoreConfig,
    RefreshConfig,
    MetadataConfig,
    RAGConfig,
    load_config,
    save_config,
    validate_config,
    create_default_config,
    merge_configs,
)

# Raglab integration
from .raglab_integration import (
    RaglabSemanticSegmenter,
    RaglabVectorDBAdapter,
    create_raglab_embedder,
    create_raglab_pipeline,
)

# CLI (not exported by default, but available)
# from .cli import cli

__all__ = [
    # Types
    'ContentKey',
    'ContentValue',
    'ContentMapping',
    'UpdateTimeMapping',
    'RefreshMapping',
    'SourceKey',
    'Timestamp',
    'ExcludeFunc',
    'ContentDecoder',
    'Source',
    'RefreshStrategy',
    # Decoders
    'ExtensionBasedDecoder',
    'text_decoder',
    'markdown_decoder',
    'python_decoder',
    'json_decoder',
    'yaml_decoder',
    'pdf_decoder',
    'docx_decoder',
    'DEFAULT_DECODERS',
    # Sources
    'FolderSource',
    'MultiSource',
    'URLSource',
    'DatabaseSource',
    'APISource',
    'GitRepoSource',
    # Async
    'AsyncFolderSource',
    'ParallelBatchProcessor',
    'make_async_embedder',
    # Refresh
    'SimpleRefreshStrategy',
    'ThresholdRefreshStrategy',
    'BatchRefreshStrategy',
    'RefreshMappingManager',
    # Chunking
    'BaseChunker',
    'MarkdownAwareChunker',
    'CodeAwareChunker',
    'SemanticChunker',
    'create_chunker',
    # Vector stores
    'BaseVectorStore',
    'QdrantStore',
    'ChromaStore',
    'FAISSStore',
    'create_vector_store',
    # Pipeline
    'RAGPipeline',
    'EnhancedRAGPipeline',
    # Metadata
    'MetadataExtractor',
    'MetadataManager',
    'SegmentMetadata',
    'TagManager',
    # Hooks
    'EventType',
    'Event',
    'EventHandler',
    'EventHookManager',
    'logging_handler',
    'progress_bar_handler',
    'statistics_collector',
    # Utils
    'DeduplicationStrategy',
    'StatePersistence',
    'CheckpointManager',
    'ContentValidator',
    'EmbeddingCache',
    # Advanced features
    'QueryEnhancer',
    'HybridSearch',
    'Reranker',
    'HierarchicalPipeline',
    'PipelineMonitor',
    'ImageSource',
    'AudioSource',
    # Config
    'SourceConfig',
    'PipelineConfig',
    'EmbedderConfig',
    'VectorStoreConfig',
    'RefreshConfig',
    'MetadataConfig',
    'RAGConfig',
    'load_config',
    'save_config',
    'validate_config',
    'create_default_config',
    'merge_configs',
    # Raglab integration
    'RaglabSemanticSegmenter',
    'RaglabVectorDBAdapter',
    'create_raglab_embedder',
    'create_raglab_pipeline',
]
