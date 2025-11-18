"""Configuration management for RAG module."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class SourceConfig:
    """Configuration for a source."""
    type: str
    path: Optional[str] = None
    exclude_patterns: List[str] = field(default_factory=list)
    exclude_paths: List[str] = field(default_factory=list)
    extensions: List[str] = field(default_factory=list)
    recursive: bool = True
    max_workers: int = 4
    use_async: bool = False

    # For URL source
    urls: List[str] = field(default_factory=list)
    crawler_depth: int = 0

    # For database source
    connection_string: Optional[str] = None
    query: Optional[str] = None

    # For git source
    repo_url: Optional[str] = None
    branch: str = "main"


@dataclass
class PipelineConfig:
    """Configuration for pipeline."""
    chunk_size: int = 400
    chunk_overlap: int = 100
    use_async: bool = False
    max_workers: int = 4
    chunking_strategy: str = "simple"  # simple, markdown, code, semantic


@dataclass
class EmbedderConfig:
    """Configuration for embedder."""
    type: str = "openai"  # openai, huggingface, sentence-transformers
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    api_key: Optional[str] = None
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    type: str = "chroma"  # chroma, qdrant, faiss
    collection_name: str = "rag_collection"
    persist_directory: Optional[str] = None
    host: str = "localhost"
    port: int = 6333
    dimension: int = 384


@dataclass
class RefreshConfig:
    """Configuration for refresh strategy."""
    strategy: str = "simple"  # simple, threshold, batch
    threshold_seconds: float = 60.0
    min_changed_count: Optional[int] = None
    min_changed_percentage: Optional[float] = None


@dataclass
class MetadataConfig:
    """Configuration for metadata extraction."""
    extract_file_stats: bool = True
    extract_mime_type: bool = True
    extract_hash: bool = True
    auto_tag_by_extension: bool = True
    auto_tag_by_directory: bool = True
    auto_tag_by_size: bool = True


@dataclass
class RAGConfig:
    """Complete RAG configuration."""
    sources: List[SourceConfig] = field(default_factory=list)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    embedder: Optional[EmbedderConfig] = None
    vector_store: Optional[VectorStoreConfig] = None
    refresh: RefreshConfig = field(default_factory=RefreshConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)


def load_config(config_path: str | Path) -> RAGConfig:
    """
    Load configuration from file.

    Args:
        config_path: Path to config file (YAML or JSON)

    Returns:
        RAGConfig instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load file
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Parse configuration
    config = RAGConfig()

    # Parse sources
    if 'sources' in data:
        config.sources = [
            SourceConfig(**source_data)
            for source_data in data['sources']
        ]

    # Parse pipeline
    if 'pipeline' in data:
        config.pipeline = PipelineConfig(**data['pipeline'])

    # Parse embedder
    if 'embedder' in data:
        config.embedder = EmbedderConfig(**data['embedder'])

    # Parse vector store
    if 'vector_store' in data:
        config.vector_store = VectorStoreConfig(**data['vector_store'])

    # Parse refresh
    if 'refresh' in data:
        config.refresh = RefreshConfig(**data['refresh'])

    # Parse metadata
    if 'metadata' in data:
        config.metadata = MetadataConfig(**data['metadata'])

    return config


def save_config(config: RAGConfig, output_path: str | Path) -> None:
    """
    Save configuration to file.

    Args:
        config: RAGConfig instance
        output_path: Path to save config
    """
    output_path = Path(output_path)

    # Convert to dict
    data = {
        'sources': [asdict(s) for s in config.sources],
        'pipeline': asdict(config.pipeline),
        'refresh': asdict(config.refresh),
        'metadata': asdict(config.metadata),
    }

    if config.embedder:
        data['embedder'] = asdict(config.embedder)

    if config.vector_store:
        data['vector_store'] = asdict(config.vector_store)

    # Save file
    with open(output_path, 'w') as f:
        if output_path.suffix in ['.yaml', '.yml']:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif output_path.suffix == '.json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {output_path.suffix}")


def validate_config(config: RAGConfig) -> List[str]:
    """
    Validate configuration.

    Args:
        config: RAGConfig instance

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate sources
    if not config.sources:
        errors.append("At least one source is required")

    for i, source in enumerate(config.sources):
        if source.type == "folder" and not source.path:
            errors.append(f"Source {i}: folder path is required")
        elif source.type == "url" and not source.urls:
            errors.append(f"Source {i}: URLs are required")
        elif source.type == "database" and not source.connection_string:
            errors.append(f"Source {i}: connection string is required")
        elif source.type == "git" and not source.repo_url:
            errors.append(f"Source {i}: repository URL is required")

    # Validate pipeline
    if config.pipeline.chunk_size <= 0:
        errors.append("chunk_size must be positive")
    if config.pipeline.chunk_overlap < 0:
        errors.append("chunk_overlap must be non-negative")
    if config.pipeline.chunk_overlap >= config.pipeline.chunk_size:
        errors.append("chunk_overlap must be less than chunk_size")

    # Validate embedder
    if config.embedder:
        if config.embedder.type not in ['openai', 'huggingface', 'sentence-transformers']:
            errors.append(f"Unknown embedder type: {config.embedder.type}")

    # Validate vector store
    if config.vector_store:
        if config.vector_store.type not in ['chroma', 'qdrant', 'faiss']:
            errors.append(f"Unknown vector store type: {config.vector_store.type}")

    # Validate refresh
    if config.refresh.strategy not in ['simple', 'threshold', 'batch']:
        errors.append(f"Unknown refresh strategy: {config.refresh.strategy}")

    return errors


def create_default_config(folder_path: str) -> RAGConfig:
    """
    Create a default configuration for a folder.

    Args:
        folder_path: Path to the folder to index

    Returns:
        RAGConfig instance
    """
    config = RAGConfig(
        sources=[
            SourceConfig(
                type="folder",
                path=folder_path,
                exclude_patterns=[r'\.git', r'__pycache__', r'\.pyc$'],
                extensions=['.md', '.txt', '.py'],
                recursive=True,
            )
        ],
        pipeline=PipelineConfig(
            chunk_size=400,
            chunk_overlap=100,
        ),
        embedder=EmbedderConfig(
            type="openai",
            model="text-embedding-3-small",
            dimensions=384,
        ),
        vector_store=VectorStoreConfig(
            type="chroma",
            collection_name="rag_collection",
        ),
        refresh=RefreshConfig(
            strategy="simple",
        ),
    )

    return config


def merge_configs(base: RAGConfig, override: RAGConfig) -> RAGConfig:
    """
    Merge two configurations, with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    merged = RAGConfig()

    # Merge sources
    merged.sources = override.sources if override.sources else base.sources

    # Merge pipeline (field by field)
    merged.pipeline = override.pipeline if override.pipeline else base.pipeline

    # Merge embedder
    merged.embedder = override.embedder if override.embedder else base.embedder

    # Merge vector store
    merged.vector_store = override.vector_store if override.vector_store else base.vector_store

    # Merge refresh
    merged.refresh = override.refresh if override.refresh else base.refresh

    # Merge metadata
    merged.metadata = override.metadata if override.metadata else base.metadata

    return merged
