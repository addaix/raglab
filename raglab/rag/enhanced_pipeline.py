"""Enhanced RAG pipeline with all advanced features."""

from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import logging

from .pipeline import RAGPipeline
from .hooks import EventHookManager, EventType, Event
from .metadata import MetadataManager, SegmentMetadata, TagManager
from .utils import StatePersistence, ContentValidator, EmbeddingCache, DeduplicationStrategy
from .advanced_features import PipelineMonitor
from .config import RAGConfig

logger = logging.getLogger(__name__)


class EnhancedRAGPipeline(RAGPipeline):
    """
    Enhanced RAG pipeline with all advanced features:
    - Event hooks
    - Metadata management
    - State persistence
    - Monitoring
    - Deduplication
    - Validation
    - Caching
    """

    def __init__(
        self,
        *args,
        enable_hooks: bool = True,
        enable_metadata: bool = True,
        enable_monitoring: bool = True,
        enable_validation: bool = False,
        enable_caching: bool = False,
        enable_deduplication: bool = False,
        validator: Optional[ContentValidator] = None,
        cache_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize enhanced pipeline.

        Args:
            *args: Arguments for base RAGPipeline
            enable_hooks: Enable event hooks
            enable_metadata: Enable metadata management
            enable_monitoring: Enable performance monitoring
            enable_validation: Enable content validation
            enable_caching: Enable embedding caching
            enable_deduplication: Enable content deduplication
            validator: Content validator
            cache_file: File for embedding cache
            **kwargs: Additional arguments for RAGPipeline
        """
        super().__init__(*args, **kwargs)

        # Advanced features
        self.hooks = EventHookManager() if enable_hooks else None
        self.metadata_manager = MetadataManager() if enable_metadata else None
        self.segment_metadata = SegmentMetadata() if enable_metadata else None
        self.tag_manager = TagManager() if enable_metadata else None
        self.monitor = PipelineMonitor() if enable_monitoring else None
        self.validator = validator if enable_validation else None
        self.cache = EmbeddingCache(cache_file) if enable_caching else None
        self.dedup = DeduplicationStrategy() if enable_deduplication else None

    @classmethod
    def from_config(cls, config: RAGConfig):
        """Create pipeline from configuration."""
        from .sources import FolderSource, MultiSource
        from .refresh import (
            SimpleRefreshStrategy,
            ThresholdRefreshStrategy,
            BatchRefreshStrategy,
            RefreshMappingManager
        )
        from .chunking import create_chunker
        from .raglab_integration import create_raglab_embedder
        from .vector_stores import create_vector_store

        # Create sources
        sources = []
        for source_cfg in config.sources:
            if source_cfg.type == "folder":
                source = FolderSource(
                    folder_path=source_cfg.path,
                    exclude_patterns=source_cfg.exclude_patterns,
                    exclude_paths=source_cfg.exclude_paths,
                    extensions=set(source_cfg.extensions) if source_cfg.extensions else None,
                    recursive=source_cfg.recursive,
                )
                sources.append(source)
            # Add other source types as needed

        # Combine sources
        if len(sources) == 1:
            source = sources[0]
        else:
            source = MultiSource(sources)

        # Create chunker
        chunker = create_chunker(
            config.pipeline.chunking_strategy,
            max_chunk_size=config.pipeline.chunk_size,
            chunk_overlap=config.pipeline.chunk_overlap,
        )

        # Create embedder
        embedder = None
        if config.embedder:
            embedder = create_raglab_embedder(
                embedder_type=config.embedder.type,
                model=config.embedder.model,
                api_key=config.embedder.api_key,
                dimensions=config.embedder.dimensions,
            )

        # Create vector store
        vector_store = None
        if config.vector_store:
            vector_store = create_vector_store(
                store_type=config.vector_store.type,
                collection_name=config.vector_store.collection_name,
                persist_directory=config.vector_store.persist_directory,
                host=config.vector_store.host,
                port=config.vector_store.port,
                dimension=config.vector_store.dimension,
                embedder=embedder,
            )

        # Create refresh strategy
        if config.refresh.strategy == "simple":
            refresh_strategy = SimpleRefreshStrategy()
        elif config.refresh.strategy == "threshold":
            refresh_strategy = ThresholdRefreshStrategy(
                threshold_seconds=config.refresh.threshold_seconds
            )
        elif config.refresh.strategy == "batch":
            refresh_strategy = BatchRefreshStrategy(
                min_changed_count=config.refresh.min_changed_count,
                min_changed_percentage=config.refresh.min_changed_percentage,
            )
        else:
            refresh_strategy = SimpleRefreshStrategy()

        refresh_manager = RefreshMappingManager(strategy=refresh_strategy)

        # Create pipeline
        return cls(
            source=source,
            segmenter=chunker.chunk if chunker else None,
            embedder=embedder,
            vector_store=vector_store,
            refresh_manager=refresh_manager,
            chunk_size=config.pipeline.chunk_size,
            chunk_overlap=config.pipeline.chunk_overlap,
            enable_metadata=config.metadata.extract_file_stats,
        )

    def run(
        self,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Run pipeline with all enhancements."""
        # Start monitoring
        if self.monitor:
            self.monitor.start()

        # Emit pipeline start event
        if self.hooks:
            self.hooks.emit(Event(type=EventType.PIPELINE_START))

        # Get content and update times
        logger.info("Fetching content from source...")
        content_mapping = self.source.get_content_mapping()
        update_times = self.source.get_update_times()

        # Extract metadata
        if self.metadata_manager and hasattr(self.source, 'folder_path'):
            logger.info("Extracting metadata...")
            metadata_mapping = self.metadata_manager.extract_metadata_mapping(
                content_mapping,
                self.source.folder_path
            )

            # Auto-tag
            if self.tag_manager:
                self.tag_manager.auto_tag_by_extension(metadata_mapping)
                self.tag_manager.auto_tag_by_size(metadata_mapping)

        # Deduplication
        if self.dedup:
            logger.info("Checking for duplicates...")
            content_mapping = self.dedup.remove_duplicates(content_mapping)

        # Validation
        if self.validator:
            logger.info("Validating content...")
            valid_content = {}
            for key, content in content_mapping.items():
                is_valid, issues = self.validator.validate(content)
                if is_valid:
                    valid_content[key] = content
                else:
                    logger.warning(f"Invalid content {key}: {issues}")

            content_mapping = valid_content

        # Run base pipeline
        stats = super().run(force_refresh, progress_callback)

        # End monitoring
        if self.monitor:
            self.monitor.end()
            self.monitor.record('total_files', stats['content_keys'])
            self.monitor.record('total_segments', stats['segments'])

        # Emit pipeline complete event
        if self.hooks:
            self.hooks.emit(Event(
                type=EventType.PIPELINE_COMPLETE,
                data=stats
            ))

        # Add monitoring stats
        if self.monitor:
            stats['monitoring'] = self.monitor.get_stats()

        return stats

    def save_state(self, file_path: str | Path) -> None:
        """Save pipeline state."""
        state = {
            'segment_mapping': self._segment_mapping,
            'vector_mapping': self._vector_mapping,
            'refresh_times': self.refresh_manager.previous_times,
        }

        if self.metadata_manager:
            state['metadata'] = self.metadata_manager.get_metadata_mapping()

        if self.tag_manager:
            state['tags'] = {k: list(v) for k, v in self.tag_manager._tags.items()}

        StatePersistence.save_state(state, file_path)
        logger.info(f"State saved to {file_path}")

    @classmethod
    def load_state(cls, file_path: str | Path) -> 'EnhancedRAGPipeline':
        """Load pipeline from saved state."""
        state = StatePersistence.load_state(file_path)

        # Create minimal pipeline
        from .sources import FolderSource
        pipeline = cls(source=FolderSource('.'))

        # Restore state
        pipeline._segment_mapping = state.get('segment_mapping', {})
        pipeline._vector_mapping = state.get('vector_mapping', {})
        pipeline.refresh_manager.previous_times = state.get('refresh_times', {})

        if 'metadata' in state and pipeline.metadata_manager:
            pipeline.metadata_manager._metadata_cache = state['metadata']

        if 'tags' in state and pipeline.tag_manager:
            for key, tags in state['tags'].items():
                pipeline.tag_manager.add_tags(key, tags)

        logger.info(f"State loaded from {file_path}")
        return pipeline
