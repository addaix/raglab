"""
Example usage of the RAG module.

This script demonstrates how to use the RAG module to index a folder of documents,
track updates, and maintain a vector database.
"""

import logging
from pathlib import Path
from raglab.rag import (
    FolderSource,
    MultiSource,
    RAGPipeline,
    RefreshMappingManager,
    SimpleRefreshStrategy,
    ExtensionBasedDecoder,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic example: Index a folder of markdown files."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage - Indexing Markdown Files")
    print("="*60)

    # Create a folder source
    source = FolderSource(
        folder_path="./docs",  # Replace with your folder
        exclude_patterns=[r"\.git", r"__pycache__", r"\.draft$"],
        extensions={'.md', '.txt'},
        recursive=True
    )

    # Create pipeline without embedder/vector store for demo
    pipeline = RAGPipeline(
        source=source,
        chunk_size=400,
        chunk_overlap=100
    )

    # Run pipeline
    stats = pipeline.run()
    print(f"\nIndexed {stats['content_keys']} files")
    print(f"Created {stats['segments']} segments")
    print(f"New files: {stats['new']}")


def example_with_exclude_logic():
    """Example with multiple exclude strategies."""
    print("\n" + "="*60)
    print("Example 2: Advanced Exclude Logic")
    print("="*60)

    # Custom exclude function
    def exclude_large_files(path: Path) -> bool:
        """Exclude files larger than 1MB."""
        if path.is_file():
            return path.stat().st_size > 1_000_000
        return False

    source = FolderSource(
        folder_path="./src",
        # Regex patterns
        exclude_patterns=[
            r"\.git",
            r"__pycache__",
            r"node_modules",
            r"\.pyc$",
            r"test_.*\.py$",  # Exclude test files
        ],
        # Explicit paths
        exclude_paths=[
            Path("./src/secrets.json"),
            Path("./src/temp"),
        ],
        # Custom function
        exclude_func=exclude_large_files,
        # Only Python and JavaScript
        extensions={'.py', '.js', '.ts'},
        recursive=True
    )

    content_mapping = source.get_content_mapping()
    print(f"\nFound {len(content_mapping)} files after exclusions")


def example_multi_source():
    """Example with multiple sources."""
    print("\n" + "="*60)
    print("Example 3: Multiple Sources")
    print("="*60)

    # Documentation source
    docs_source = FolderSource(
        folder_path="./docs",
        extensions={'.md', '.txt', '.rst'}
    )

    # Code source
    code_source = FolderSource(
        folder_path="./src",
        extensions={'.py', '.js'},
        exclude_patterns=[r"test_", r"__pycache__"]
    )

    # Examples source
    examples_source = FolderSource(
        folder_path="./examples",
        extensions={'.py', '.ipynb'}
    )

    # Combine sources
    multi_source = MultiSource([docs_source, code_source, examples_source])

    # Get combined content
    content_mapping = multi_source.get_content_mapping()
    print(f"\nTotal files from all sources: {len(content_mapping)}")

    # Show source distribution
    source_counts = {}
    for key in content_mapping.keys():
        source_id = key.split('/')[0]
        source_counts[source_id] = source_counts.get(source_id, 0) + 1

    for source_id, count in source_counts.items():
        print(f"  {source_id}: {count} files")


def example_incremental_updates():
    """Example showing incremental updates."""
    print("\n" + "="*60)
    print("Example 4: Incremental Updates")
    print("="*60)

    source = FolderSource(
        folder_path="./docs",
        extensions={'.md', '.txt'}
    )

    pipeline = RAGPipeline(
        source=source,
        chunk_size=400,
        chunk_overlap=100
    )

    # First run - indexes everything
    print("\nFirst run (initial indexing):")
    stats1 = pipeline.run()
    print(f"  Indexed {stats1['content_keys']} files")
    print(f"  New files: {stats1['new']}")

    # Second run - nothing changed
    print("\nSecond run (no changes):")
    stats2 = pipeline.run()
    print(f"  Refreshed: {stats2['refreshed']}")
    print(f"  Modified: {stats2['modified']}")
    print(f"  New: {stats2['new']}")

    # To simulate changes, you would:
    # 1. Modify a file
    # 2. Add a new file
    # 3. Delete a file
    # Then run again and see stats3['modified'], stats3['new'], stats3['deleted']


def example_custom_decoder():
    """Example with custom decoder."""
    print("\n" + "="*60)
    print("Example 5: Custom Decoder")
    print("="*60)

    # Create custom decoder for log files
    def log_decoder(file_path: Path) -> str:
        """Custom decoder that only extracts ERROR and WARNING lines."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Filter for important lines
            important_lines = [
                line for line in lines
                if 'ERROR' in line or 'WARNING' in line
            ]
            return ''.join(important_lines)

    # Create decoder and register custom handler
    decoder = ExtensionBasedDecoder()
    decoder.register_decoder('.log', log_decoder)

    # Use in source
    source = FolderSource(
        folder_path="./logs",
        decoder=decoder,
        extensions={'.log'}
    )

    content_mapping = source.get_content_mapping()
    print(f"\nProcessed {len(content_mapping)} log files")
    if content_mapping:
        first_key = list(content_mapping.keys())[0]
        print(f"Sample content from {first_key}:")
        print(content_mapping[first_key][:200])


def example_pipeline_with_embedder():
    """Example with embedder function (mock)."""
    print("\n" + "="*60)
    print("Example 6: Pipeline with Embedder")
    print("="*60)

    # Mock embedder function
    def mock_embedder(segments):
        """Mock embedder for demonstration."""
        import random
        # Return random vectors
        return [[random.random() for _ in range(384)] for _ in segments]

    source = FolderSource(
        folder_path="./docs",
        extensions={'.md', '.txt'}
    )

    pipeline = RAGPipeline(
        source=source,
        embedder=mock_embedder,
        chunk_size=400,
        chunk_overlap=100
    )

    stats = pipeline.run()
    print(f"\nIndexed {stats['content_keys']} files")
    print(f"Created {stats['segments']} segments")
    print(f"Generated {stats['vectors']} vectors")


def example_refresh_strategies():
    """Example with different refresh strategies."""
    print("\n" + "="*60)
    print("Example 7: Refresh Strategies")
    print("="*60)

    from raglab.rag import ThresholdRefreshStrategy, BatchRefreshStrategy

    # Strategy 1: Simple (default)
    print("\n1. Simple Strategy (refresh on any change)")
    strategy1 = SimpleRefreshStrategy()
    manager1 = RefreshMappingManager(strategy=strategy1)

    # Strategy 2: Threshold (only refresh if file changed > 5 minutes ago)
    print("\n2. Threshold Strategy (refresh if changed > 5 minutes ago)")
    strategy2 = ThresholdRefreshStrategy(threshold_seconds=300)
    manager2 = RefreshMappingManager(strategy=strategy2)

    # Strategy 3: Batch (only refresh if >= 5 files changed)
    print("\n3. Batch Strategy (refresh if >= 5 files changed)")
    strategy3 = BatchRefreshStrategy(min_changed_count=5)
    manager3 = RefreshMappingManager(strategy=strategy3)

    # Use in pipeline
    source = FolderSource("./docs", extensions={'.md'})

    pipeline = RAGPipeline(
        source=source,
        refresh_manager=manager2,  # Using threshold strategy
        chunk_size=400
    )

    stats = pipeline.run()
    print(f"\nProcessed with threshold strategy")


def example_getting_stats():
    """Example showing how to get statistics."""
    print("\n" + "="*60)
    print("Example 8: Getting Statistics")
    print("="*60)

    source = FolderSource(
        folder_path="./docs",
        extensions={'.md', '.txt'}
    )

    pipeline = RAGPipeline(source=source)

    # Before running
    print("\nBefore running pipeline:")
    stats_before = pipeline.get_stats()
    print(f"  Segments: {stats_before['total_segments']}")
    print(f"  Vectors: {stats_before['total_vectors']}")

    # Run pipeline
    run_stats = pipeline.run()

    # After running
    print("\nAfter running pipeline:")
    stats_after = pipeline.get_stats()
    print(f"  Segments: {stats_after['total_segments']}")
    print(f"  Previous content keys: {stats_after['previous_content_keys']}")

    print("\nRun statistics:")
    print(f"  Content keys: {run_stats['content_keys']}")
    print(f"  Segments: {run_stats['segments']}")
    print(f"  New: {run_stats['new']}")
    print(f"  Modified: {run_stats['modified']}")
    print(f"  Deleted: {run_stats['deleted']}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("RAG Module Examples")
    print("="*60)

    # Note: These examples assume certain directories exist
    # Uncomment the ones that are relevant to your setup

    # example_basic_usage()
    # example_with_exclude_logic()
    # example_multi_source()
    # example_incremental_updates()
    # example_custom_decoder()
    # example_pipeline_with_embedder()
    # example_refresh_strategies()
    # example_getting_stats()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nNote: Uncomment the examples you want to run in main()")
    print("and make sure the required directories exist.")


if __name__ == "__main__":
    main()
