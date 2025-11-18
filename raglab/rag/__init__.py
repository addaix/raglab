"""
RAG (Retrieval-Augmented Generation) module for raglab.

This module provides a comprehensive RAG system with:
- Source management (folders with exclude logic)
- Content mappings with extension-based decoding
- Update time tracking
- Refresh mappings to determine what needs updating
- Full pipeline for segmentation, vectorization, and vector DB updates

Example usage:
    >>> from raglab.rag import FolderSource, RAGPipeline
    >>>
    >>> # Create a folder source with exclude patterns
    >>> source = FolderSource(
    ...     folder_path="/path/to/docs",
    ...     exclude_patterns=[r"\.git", r"__pycache__"],
    ...     extensions={'.md', '.txt', '.py'}
    ... )
    >>>
    >>> # Create a pipeline
    >>> pipeline = RAGPipeline(
    ...     source=source,
    ...     embedder=your_embedder_function,
    ...     vector_store=your_vector_store
    ... )
    >>>
    >>> # Run the pipeline
    >>> stats = pipeline.run()
    >>>
    >>> # Search
    >>> results = pipeline.search("your query")
"""

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

from .sources import (
    FolderSource,
    MultiSource,
)

from .refresh import (
    SimpleRefreshStrategy,
    ThresholdRefreshStrategy,
    BatchRefreshStrategy,
    RefreshMappingManager,
)

from .pipeline import (
    RAGPipeline,
)

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
    # Refresh
    'SimpleRefreshStrategy',
    'ThresholdRefreshStrategy',
    'BatchRefreshStrategy',
    'RefreshMappingManager',
    # Pipeline
    'RAGPipeline',
]
