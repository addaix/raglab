# RAG Module

A comprehensive Retrieval-Augmented Generation (RAG) module for raglab that provides intelligent content indexing, update tracking, and vector database management.

## Features

### 1. Source Management
- **FolderSource**: Manage folder-based content sources
- **Exclude Logic**: Three types of exclusion strategies:
  - Regex patterns for flexible path matching
  - Explicit file/folder lists
  - Custom exclude functions
- **Multi-source Support**: Combine multiple sources into a unified source

### 2. Content Mapping & Decoding
- **Extension-based Decoders**: Automatic content decoding based on file extension
- **Supported Formats**: Text, Markdown, Python, JSON, YAML, PDF, DOCX, and more
- **Custom Decoders**: Easily register custom decoders for specific file types

### 3. Update Time Tracking
- **Automatic Tracking**: Modification times automatically tracked for folder sources
- **Refresh Mapping**: Intelligent determination of what needs updating
- **Multiple Strategies**:
  - `SimpleRefreshStrategy`: Refresh on any change
  - `ThresholdRefreshStrategy`: Refresh only if change exceeds time threshold
  - `BatchRefreshStrategy`: Refresh when minimum number/percentage of files change

### 4. Content Processing Pipeline
- **Segmentation**: Break content into manageable chunks
- **Vectorization**: Convert text segments into embeddings
- **Vector DB Integration**: Automatic updates to vector database
- **Incremental Updates**: Only process changed content

## Architecture

```
Sources → Content Mapping → Segments → Vectors → Vector DB
                ↓
         Update Times → Refresh Mapping
```

## Installation

The RAG module is part of raglab. Make sure you have the required dependencies:

```bash
pip install langchain nltk
# Optional: for PDF/DOCX support
pip install pypdf python-docx
```

## Quick Start

### Basic Usage

```python
from raglab.rag import FolderSource, RAGPipeline

# Create a folder source with exclude patterns
source = FolderSource(
    folder_path="/path/to/docs",
    exclude_patterns=[r"\.git", r"__pycache__", r"\.pyc$"],
    extensions={'.md', '.txt', '.py'}  # Only include these extensions
)

# Create embedder function (example with OpenAI)
def my_embedder(segments):
    from langchain_openai import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()
    return embeddings_model.embed_documents(segments)

# Create vector store (example)
from your_vector_db import VectorStore
vector_store = VectorStore()

# Create and run pipeline
pipeline = RAGPipeline(
    source=source,
    embedder=my_embedder,
    vector_store=vector_store,
    chunk_size=400,
    chunk_overlap=100
)

# First run - indexes all content
stats = pipeline.run()
print(f"Indexed {stats['content_keys']} files, "
      f"created {stats['segments']} segments, "
      f"{stats['vectors']} vectors")

# Subsequent runs - only process changes
stats = pipeline.run()
print(f"Updated {stats['modified']} files, "
      f"added {stats['new']} new files, "
      f"deleted {stats['deleted']} files")

# Search
results = pipeline.search("your query", k=5)
```

### Advanced: Multiple Sources

```python
from raglab.rag import FolderSource, MultiSource, RAGPipeline

# Create multiple sources
docs_source = FolderSource(
    "/path/to/docs",
    extensions={'.md', '.txt'}
)

code_source = FolderSource(
    "/path/to/code",
    extensions={'.py', '.js'},
    exclude_patterns=[r"test_", r"__pycache__"]
)

# Combine sources
multi_source = MultiSource([docs_source, code_source])

# Create pipeline with combined sources
pipeline = RAGPipeline(
    source=multi_source,
    embedder=my_embedder,
    vector_store=vector_store
)

pipeline.run()
```

### Advanced: Custom Exclude Logic

```python
from pathlib import Path
from raglab.rag import FolderSource

# Exclude function based on file size
def exclude_large_files(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 1_000_000  # Exclude files > 1MB
    return False

source = FolderSource(
    folder_path="/path/to/docs",
    exclude_func=exclude_large_files,
    exclude_patterns=[r"\.git", r"node_modules"],
    exclude_paths=[Path("/path/to/docs/secret.txt")]
)
```

### Advanced: Custom Refresh Strategy

```python
from raglab.rag import (
    FolderSource,
    RAGPipeline,
    ThresholdRefreshStrategy,
    RefreshMappingManager
)

# Only refresh if files changed more than 5 minutes ago
strategy = ThresholdRefreshStrategy(threshold_seconds=300)
refresh_manager = RefreshMappingManager(strategy=strategy)

pipeline = RAGPipeline(
    source=source,
    embedder=my_embedder,
    vector_store=vector_store,
    refresh_manager=refresh_manager
)
```

### Advanced: Custom Decoder

```python
from pathlib import Path
from raglab.rag import FolderSource, ExtensionBasedDecoder

# Create custom decoder for .log files
def log_decoder(file_path: Path) -> str:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Only return error lines
        return '\n'.join(line for line in lines if 'ERROR' in line)

# Create decoder and register custom handler
decoder = ExtensionBasedDecoder()
decoder.register_decoder('.log', log_decoder)

# Use custom decoder in source
source = FolderSource(
    folder_path="/path/to/logs",
    decoder=decoder,
    extensions={'.log'}
)
```

## API Reference

### FolderSource

```python
FolderSource(
    folder_path: str | Path,
    decoder: ExtensionBasedDecoder | None = None,
    exclude_patterns: List[str] | None = None,
    exclude_paths: List[str | Path] | None = None,
    exclude_func: Callable[[Path], bool] | None = None,
    recursive: bool = True,
    extensions: Set[str] | None = None
)
```

**Methods:**
- `get_content_mapping()`: Returns mapping of file paths to content
- `get_update_times()`: Returns mapping of file paths to modification times
- `refresh()`: Clear caches and rescan folder
- `add_exclude_pattern(pattern)`: Add regex exclude pattern
- `add_exclude_path(path)`: Add explicit path to exclude
- `set_exclude_func(func)`: Set custom exclude function

### RAGPipeline

```python
RAGPipeline(
    source: FolderSource | MultiSource,
    segmenter: Callable[[str], List[str]] | None = None,
    embedder: Callable[[List[str]], List[List[float]]] | None = None,
    vector_store: Any | None = None,
    refresh_manager: RefreshMappingManager | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 100
)
```

**Methods:**
- `run(force_refresh=False)`: Run the full pipeline, returns statistics
- `search(query, k=5)`: Search for relevant content
- `get_stats()`: Get current pipeline statistics
- `segment_content(content_mapping)`: Segment content into chunks
- `vectorize_segments(segment_mapping)`: Vectorize segments
- `update_vector_store(...)`: Update vector database

### RefreshMappingManager

```python
RefreshMappingManager(
    strategy: RefreshStrategy | None = None
)
```

**Methods:**
- `get_refresh_mapping(current_times)`: Get mapping of what needs refreshing
- `get_keys_to_refresh(current_times)`: Get set of keys to refresh
- `get_new_keys(current_times)`: Get set of new keys
- `get_deleted_keys(current_times)`: Get set of deleted keys
- `get_modified_keys(current_times)`: Get set of modified keys
- `update_times(current_times)`: Update stored times

## Examples

### Example 1: Documentation Indexing

```python
from raglab.rag import FolderSource, RAGPipeline

# Index documentation
docs_source = FolderSource(
    folder_path="./docs",
    exclude_patterns=[r"\.draft$", r"_old"],
    extensions={'.md', '.rst', '.txt'}
)

pipeline = RAGPipeline(
    source=docs_source,
    embedder=embedder,
    vector_store=store
)

# Initial indexing
pipeline.run()

# ... make changes to docs ...

# Update index (only processes changes)
stats = pipeline.run()
print(f"Updated {stats['modified']} documents")
```

### Example 2: Code Search

```python
from raglab.rag import FolderSource, RAGPipeline

# Index code repository
code_source = FolderSource(
    folder_path="./src",
    exclude_patterns=[
        r"\.git",
        r"__pycache__",
        r"node_modules",
        r"\.pyc$",
        r"\.min\.js$"
    ],
    extensions={'.py', '.js', '.ts', '.java'}
)

pipeline = RAGPipeline(
    source=code_source,
    embedder=code_embedder,
    vector_store=code_store,
    chunk_size=800  # Larger chunks for code
)

pipeline.run()

# Search for code
results = pipeline.search("authentication implementation", k=10)
```

### Example 3: Multi-format Knowledge Base

```python
from raglab.rag import FolderSource, MultiSource, RAGPipeline

# Different sources for different content types
docs_source = FolderSource(
    "./knowledge_base/docs",
    extensions={'.md', '.txt'}
)

pdf_source = FolderSource(
    "./knowledge_base/pdfs",
    extensions={'.pdf'}
)

code_source = FolderSource(
    "./knowledge_base/examples",
    extensions={'.py', '.js'}
)

# Combine all sources
kb_source = MultiSource([docs_source, pdf_source, code_source])

pipeline = RAGPipeline(
    source=kb_source,
    embedder=embedder,
    vector_store=store
)

# Index everything
stats = pipeline.run()
print(f"Indexed {stats['content_keys']} items from knowledge base")
```

## Testing

Run tests with pytest:

```bash
pytest raglab/rag/tests/
```

Test coverage includes:
- Source management and exclude logic
- Content decoders
- Refresh strategies
- Full pipeline integration

## Performance Tips

1. **Batch Processing**: The pipeline automatically batches embedding operations for efficiency
2. **Incremental Updates**: Use the refresh mapping to only process changed content
3. **Chunk Size**: Adjust `chunk_size` based on your content type (larger for code, smaller for prose)
4. **Extensions Filter**: Limit to specific extensions to avoid processing unnecessary files
5. **Exclude Patterns**: Use exclude patterns to skip directories like `.git`, `node_modules`, etc.

## Contributing

When adding new features:
1. Update type definitions in `types.py`
2. Add comprehensive tests
3. Update this README with examples
4. Follow the existing code style

## License

Part of the raglab project.
