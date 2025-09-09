Note: When you see "facade" as a folder name or package name, please replace this to be `vd_facade_1`. 
That is the name we'll be working under. 

Make sure you run the tests (use temp folders/files if you need folders/files to work with), 
and/or write extra tests if you feel it's important. 

Use a `vd_facade_1/dev_notes.md` files as your scratch pad to write down some notes about what you tried, 
what didn't work, the reasons it didn't work, and what you finally did that worked. 
Be concise. You are just writing this for future AI developers. 



# Facade Architecture Specification for Vector Store Integration

This specification defines a facade system that provides a unified interface for vector store operations, seamlessly integrating the `imbed` package with optional LangChain backends. The architecture follows functional programming principles with thin object-oriented wrappers where appropriate, uses registry-based component management, and ensures zero hard dependencies on LangChain.

## Core Design Principles

1. **Protocol-First Design**: All interfaces defined as runtime-checkable protocols
2. **Registry-Based Components**: Functions registered in dictionaries, not class hierarchies
3. **Batch-First Operations**: All operations handle collections by default
4. **Lazy Evaluation**: Use generators and defer computation where possible
5. **Explicit State Management**: Clear separation between computation and storage
6. **Progressive Enhancement**: Gracefully upgrade from simple to advanced implementations

## Architecture Components

### 1. Type System and Protocols

```python
# File: facade/types.py
"""Core type definitions for the facade system"""

from typing import Protocol, runtime_checkable, Mapping, Sequence, Union, Iterable, Optional, Callable, Any
from typing import TypeVar, TypeAlias

# Import domain types from imbed
from imbed.imbed_types import (
    Segment, SegmentKey, SegmentMapping,
    Vector, VectorMapping,
    PlanarVector, PlanarVectorMapping
)

# Additional type aliases for clarity
ComponentKey: TypeAlias = str
ComponentSpec: TypeAlias = Union[str, dict[str, Any], Callable]
BatchResult: TypeAlias = Union[Mapping[SegmentKey, Any], Sequence[Any]]

# Runtime-checkable protocols
@runtime_checkable
class Segmenter(Protocol):
    """Protocol for text segmentation"""
    def __call__(self, text: str) -> Iterable[Segment]: ...

@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding generation - handles both single and batch"""
    def __call__(self, segments: Union[SegmentMapping, Iterable[Segment]]) -> BatchResult: ...

@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage and retrieval"""
    def add_batch(self, segments: SegmentMapping, vectors: Optional[VectorMapping] = None) -> None: ...
    def search(self, query: Union[str, Vector], k: int = 5) -> Sequence[tuple[SegmentKey, float]]: ...
    def clear(self) -> None: ...
    def __contains__(self, key: SegmentKey) -> bool: ...
    def __len__(self) -> int: ...

@runtime_checkable
class Indexer(Protocol):
    """Protocol for index operations"""
    def build(self, vectors: VectorMapping) -> Any: ...
    def search(self, query: Vector, k: int = 5) -> Sequence[tuple[SegmentKey, float]]: ...
    def update(self, key: SegmentKey, vector: Vector) -> None: ...
```

### 2. Registry System

```python
# File: facade/registry.py
"""Component registry management"""

from typing import Callable, Any, Optional, MutableMapping
from functools import partial
from collections import defaultdict
import os

# Component registries - global dictionaries for each component type
segmenters: MutableMapping[str, Segmenter] = {}
embedders: MutableMapping[str, Embedder] = {}
vector_stores: MutableMapping[str, Callable[..., VectorStore]] = {}  # Factories
indexers: MutableMapping[str, Callable[..., Indexer]] = {}  # Factories

def register_component(registry: MutableMapping, name: Optional[str] = None):
    """Generic component registration decorator"""
    def decorator(component: Callable):
        key = name or component.__name__
        registry[key] = component
        return component
    return decorator

# Specific registration decorators
register_segmenter = partial(register_component, segmenters)
register_embedder = partial(register_component, embedders)
register_vector_store = partial(register_component, vector_stores)
register_indexer = partial(register_component, indexers)

# Component retrieval with fallback
def get_component(registry: MutableMapping, spec: ComponentSpec, **kwargs) -> Any:
    """
    Retrieve a component from registry by specification.
    
    Args:
        registry: The component registry to search
        spec: Can be:
            - str: Key to lookup in registry
            - dict: {"component_name": {kwargs}} for partial application
            - Callable: Direct function/class to use
        **kwargs: Additional kwargs to pass to component
    
    Returns:
        The configured component
    """
    if callable(spec):
        return spec(**kwargs) if kwargs else spec
    
    if isinstance(spec, str):
        if spec not in registry:
            raise KeyError(f"Component '{spec}' not found in registry")
        component = registry[spec]
        return component(**kwargs) if kwargs else component
    
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError("Component spec dict must have exactly one key")
        name, params = next(iter(spec.items()))
        if name not in registry:
            raise KeyError(f"Component '{name}' not found in registry")
        component = registry[name]
        merged_kwargs = {**params, **kwargs}
        return component(**merged_kwargs) if merged_kwargs else component
    
    raise TypeError(f"Invalid component spec type: {type(spec)}")
```

### 3. Built-in Components

```python
# File: facade/components/builtin.py
"""Built-in component implementations using imbed"""

from imbed.segmentation_util import fixed_step_chunker
from imbed.components.segmentation import segmenters as imbed_segmenters
from imbed.components.vectorization import embedders as imbed_embedders
from imbed.util import cosine_similarity
import numpy as np
from typing import Union, Mapping, Sequence, Iterable
from collections.abc import Mapping as MappingABC

from ..registry import register_segmenter, register_embedder, register_vector_store
from ..types import Segment, SegmentKey, SegmentMapping, Vector, VectorMapping

# Register imbed segmenters
for name, segmenter in imbed_segmenters.items():
    register_segmenter(name)(segmenter)

# Register imbed embedders with batch handling
for name, embedder in imbed_embedders.items():
    @register_embedder(name)
    def batch_embedder(segments: Union[SegmentMapping, Iterable[Segment]], 
                      _embedder=embedder) -> Union[VectorMapping, list[Vector]]:
        """Wrapper to ensure batch handling"""
        if isinstance(segments, MappingABC):
            # Return mapping if input is mapping
            return _embedder(segments)
        else:
            # Return list for iterable input
            return _embedder(list(segments))

# Simple in-memory vector store
@register_vector_store("memory")
class MemoryVectorStore:
    """Simple in-memory vector store implementation"""
    
    def __init__(self, embedder=None):
        self.segments: dict[SegmentKey, Segment] = {}
        self.vectors: dict[SegmentKey, Vector] = {}
        self.embedder = embedder
    
    def add_batch(self, segments: SegmentMapping, vectors: Optional[VectorMapping] = None):
        """Add segments and optionally vectors"""
        self.segments.update(segments)
        
        if vectors is not None:
            self.vectors.update(vectors)
        elif self.embedder is not None:
            # Compute embeddings if not provided
            computed_vectors = self.embedder(segments)
            if isinstance(computed_vectors, MappingABC):
                self.vectors.update(computed_vectors)
            else:
                # If embedder returns list, zip with keys
                self.vectors.update(zip(segments.keys(), computed_vectors))
    
    def search(self, query: Union[str, Vector], k: int = 5) -> Sequence[tuple[SegmentKey, float]]:
        """Search for similar segments"""
        if not self.vectors:
            return []
        
        # Get query vector
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("No embedder configured for text queries")
            query_vector = self.embedder([query])[0]
        else:
            query_vector = query
        
        # Compute similarities
        similarities = []
        for key, vector in self.vectors.items():
            score = cosine_similarity(query_vector, vector)
            similarities.append((key, float(score)))
        
        # Return top k
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    
    def clear(self):
        """Clear all data"""
        self.segments.clear()
        self.vectors.clear()
    
    def __contains__(self, key: SegmentKey) -> bool:
        return key in self.segments
    
    def __len__(self) -> int:
        return len(self.segments)
```

### 4. LangChain Integration Module

```python
# File: facade/components/langchain_integration.py
"""Optional LangChain integration - only loaded if LangChain is available"""

import warnings
from typing import Optional, Any, Union
from functools import lru_cache

from ..registry import register_embedder, register_vector_store, register_segmenter
from ..types import SegmentMapping, VectorMapping, Segment, Vector

@lru_cache(maxsize=1)
def langchain_available() -> bool:
    """Check if LangChain is available"""
    try:
        import langchain
        return True
    except ImportError:
        return False

def register_langchain_components():
    """Register LangChain components if available"""
    if not langchain_available():
        return
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS, Chroma, Pinecone
        
        # Register text splitters as segmenters
        @register_segmenter("langchain_recursive")
        def recursive_segmenter(text: str, chunk_size: int = 1000, 
                               chunk_overlap: int = 200) -> list[Segment]:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.split_text(text)
        
        # Register embedders with batch handling
        @register_embedder("langchain_openai")
        class LangChainOpenAIEmbedder:
            def __init__(self, model: str = "text-embedding-3-small", **kwargs):
                self.embeddings = OpenAIEmbeddings(model=model, **kwargs)
            
            def __call__(self, segments: Union[SegmentMapping, list[Segment]]) -> Union[VectorMapping, list[Vector]]:
                if isinstance(segments, dict):
                    texts = list(segments.values())
                    vectors = self.embeddings.embed_documents(texts)
                    return dict(zip(segments.keys(), vectors))
                else:
                    texts = list(segments)
                    return self.embeddings.embed_documents(texts)
        
        # Register vector store factories
        @register_vector_store("langchain_faiss")
        class LangChainFAISSStore:
            """FAISS vector store wrapper"""
            
            def __init__(self, embedder=None, **kwargs):
                import faiss
                import numpy as np
                
                if embedder is None:
                    # Use a default embedder
                    from ..registry import get_component, embedders
                    embedder = get_component(embedders, "default")
                
                self.embedder = embedder
                self.store = None
                self.kwargs = kwargs
                self.key_to_id = {}
                self.id_to_key = {}
                self.next_id = 0
            
            def add_batch(self, segments: SegmentMapping, vectors: Optional[VectorMapping] = None):
                # Implementation details for FAISS integration
                # This would create or update the FAISS index
                pass
            
            def search(self, query: Union[str, Vector], k: int = 5) -> list[tuple[SegmentKey, float]]:
                # Implementation details for FAISS search
                pass
            
            def clear(self):
                self.store = None
                self.key_to_id.clear()
                self.id_to_key.clear()
                self.next_id = 0
            
            def __contains__(self, key: SegmentKey) -> bool:
                return key in self.key_to_id
            
            def __len__(self) -> int:
                return len(self.key_to_id)
        
    except ImportError as e:
        warnings.warn(f"Could not import LangChain components: {e}")

# Auto-register on import if LangChain is available
register_langchain_components()
```

### 5. Main Facade Interface

```python
# File: facade/interface.py
"""Main facade interface for vector store operations"""

from typing import Optional, Union, Any, Callable
from dataclasses import dataclass, field
from functools import partial

from imbed.imbed_types import SegmentsSpec
from imbed.util import ensure_segments_mapping

from .registry import (
    get_component, 
    segmenters, 
    embedders, 
    vector_stores,
    indexers
)
from .types import (
    ComponentSpec, 
    SegmentMapping, 
    VectorMapping,
    Segment,
    SegmentKey
)

@dataclass
class VectorSearchEngine:
    """
    Main facade for vector search operations.
    
    This class provides a high-level interface that:
    1. Accepts various input formats
    2. Handles segmentation, embedding, and indexing
    3. Provides search functionality
    4. Automatically selects best available backend
    """
    
    # Component specifications (can be string, dict, or callable)
    segmenter: ComponentSpec = "default"
    embedder: ComponentSpec = "default"
    vector_store: ComponentSpec = "memory"
    
    # Configuration
    auto_segment: bool = True
    auto_embed: bool = True
    batch_size: int = 100
    
    # Internal state (initialized in __post_init__)
    _segmenter: Any = field(init=False, default=None)
    _embedder: Any = field(init=False, default=None)
    _vector_store: Any = field(init=False, default=None)
    
    def __post_init__(self):
        """Initialize components from specifications"""
        self._segmenter = get_component(segmenters, self.segmenter)
        self._embedder = get_component(embedders, self.embedder)
        
        # Pass embedder to vector store if it needs it
        store_kwargs = {}
        if isinstance(self.vector_store, str) or isinstance(self.vector_store, dict):
            store_kwargs['embedder'] = self._embedder
        self._vector_store = get_component(vector_stores, self.vector_store, **store_kwargs)
    
    def add_documents(self, 
                     documents: Union[SegmentsSpec, Mapping[str, str]],
                     *,
                     segment: Optional[bool] = None,
                     embed: Optional[bool] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: Can be:
                - A single string
                - A list of strings  
                - A mapping of IDs to strings
                - Already segmented content (SegmentMapping)
            segment: Override auto_segment for this call
            embed: Override auto_embed for this call
        """
        # Determine whether to segment
        should_segment = segment if segment is not None else self.auto_segment
        should_embed = embed if embed is not None else self.auto_embed
        
        # Ensure we have a mapping
        if isinstance(documents, str):
            documents = {"0": documents}
        elif isinstance(documents, list):
            documents = {str(i): doc for i, doc in enumerate(documents)}
        
        # Segment if needed
        if should_segment and self._segmenter:
            segments = {}
            for doc_id, text in documents.items():
                doc_segments = list(self._segmenter(text))
                for i, seg in enumerate(doc_segments):
                    segments[f"{doc_id}_{i}"] = seg
        else:
            segments = ensure_segments_mapping(documents)
        
        # Add to store (embedding happens inside if configured)
        self._vector_store.add_batch(segments)
    
    def search(self, 
              query: Union[str, Vector], 
              k: int = 5,
              *,
              return_segments: bool = True) -> list[tuple[Union[SegmentKey, Segment], float]]:
        """
        Search for similar segments.
        
        Args:
            query: Query text or vector
            k: Number of results
            return_segments: If True, return segment text; if False, return keys
            
        Returns:
            List of (segment_or_key, score) tuples
        """
        results = self._vector_store.search(query, k)
        
        if return_segments and hasattr(self._vector_store, 'segments'):
            # Return actual segment text if available
            return [(self._vector_store.segments.get(key, key), score) 
                   for key, score in results]
        
        return results
    
    def clear(self) -> None:
        """Clear all data from the vector store"""
        self._vector_store.clear()
    
    @property
    def size(self) -> int:
        """Number of segments in the store"""
        return len(self._vector_store)
    
    @classmethod
    def create_with_best_backend(cls, **kwargs) -> 'VectorSearchEngine':
        """
        Factory method that automatically selects the best available backend.
        
        Priority order:
        1. LangChain FAISS (if available)
        2. LangChain Chroma (if available)  
        3. Memory store (always available)
        """
        # Check what's available
        from .registry import vector_stores
        
        if "langchain_faiss" in vector_stores:
            return cls(vector_store="langchain_faiss", **kwargs)
        elif "langchain_chroma" in vector_stores:
            return cls(vector_store="langchain_chroma", **kwargs)
        else:
            return cls(vector_store="memory", **kwargs)
```

### 6. Configuration Management

```python
# File: facade/config.py
"""Configuration management for the facade"""

import os
from typing import Optional, Any
from dataclasses import dataclass
from config2py import get_config as get_config_value

# Environment variable prefix
ENV_PREFIX = "VECTOR_FACADE_"

@dataclass
class FacadeConfig:
    """Configuration for the vector facade system"""
    
    # Default components
    default_segmenter: str = "default"
    default_embedder: str = "default"
    default_vector_store: str = "memory"
    
    # Behavior flags
    auto_segment: bool = True
    auto_embed: bool = True
    prefer_langchain: bool = True
    
    # Performance
    batch_size: int = 100
    cache_embeddings: bool = True
    
    @classmethod
    def from_env(cls) -> 'FacadeConfig':
        """Load configuration from environment variables"""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            env_var = f"{ENV_PREFIX}{field_name.upper()}"
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert to appropriate type
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == bool:
                    value = value.lower() in ('true', '1', 'yes')
                elif field_type == int:
                    value = int(value)
                kwargs[field_name] = value
        return cls(**kwargs)

# Global config instance
config = FacadeConfig.from_env()
```

### 7. Usage Examples

```python
# File: facade/examples.py
"""Usage examples for the facade system"""

def example_simple_usage():
    """Basic usage with default components"""
    from facade.interface import VectorSearchEngine
    
    # Create engine with defaults
    engine = VectorSearchEngine()
    
    # Add documents
    engine.add_documents([
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Vectors represent semantic meaning"
    ])
    
    # Search
    results = engine.search("programming", k=2)
    for segment, score in results:
        print(f"{score:.3f}: {segment}")

def example_with_langchain():
    """Using LangChain components if available"""
    from facade.interface import VectorSearchEngine
    
    # Explicitly use LangChain components
    engine = VectorSearchEngine(
        segmenter="langchain_recursive",
        embedder="langchain_openai",
        vector_store="langchain_faiss"
    )
    
    # Add documents with custom segmentation params
    engine.segmenter = {"langchain_recursive": {"chunk_size": 500}}
    engine.add_documents(large_documents)
    
    results = engine.search("specific query")

def example_custom_components():
    """Register and use custom components"""
    from facade.registry import register_segmenter, register_embedder
    from facade.interface import VectorSearchEngine
    
    # Register custom segmenter
    @register_segmenter("custom_sentences")
    def sentence_segmenter(text: str) -> list[str]:
        import re
        return re.split(r'[.!?]+', text)
    
    # Register custom embedder
    @register_embedder("custom_tfidf")
    class TFIDFEmbedder:
        def __init__(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer()
            self.fitted = False
        
        def __call__(self, segments):
            texts = list(segments.values()) if isinstance(segments, dict) else list(segments)
            if not self.fitted:
                vectors = self.vectorizer.fit_transform(texts)
                self.fitted = True
            else:
                vectors = self.vectorizer.transform(texts)
            
            if isinstance(segments, dict):
                return dict(zip(segments.keys(), vectors.toarray()))
            return vectors.toarray().tolist()
    
    # Use custom components
    engine = VectorSearchEngine(
        segmenter="custom_sentences",
        embedder="custom_tfidf"
    )

def example_progressive_enhancement():
    """Automatically use best available backend"""
    from facade.interface import VectorSearchEngine
    
    # This will use LangChain if available, otherwise fallback to memory
    engine = VectorSearchEngine.create_with_best_backend()
    
    # Check what backend is being used
    print(f"Using vector store: {engine._vector_store.__class__.__name__}")
    
    # Works the same regardless of backend
    engine.add_documents(documents)
    results = engine.search("query")
```

## Implementation Checklist

When implementing this facade system:

1. **Start with core protocols** (`types.py`) - these define the contracts
2. **Implement the registry system** (`registry.py`) - this manages components
3. **Add built-in components** (`components/builtin.py`) - basic functionality
4. **Create the main interface** (`interface.py`) - user-facing API
5. **Add LangChain integration** (`components/langchain_integration.py`) - optional enhancement
6. **Implement configuration** (`config.py`) - environment-based settings
7. **Write comprehensive tests** - ensure all components work together
8. **Add examples and documentation** - show usage patterns

## Key Design Decisions Explained

### Why Registries Over Classes?
- Functions are simpler to test and compose
- No inheritance complexity
- Easy to add new implementations without modifying existing code
- Matches the `imbed` package patterns

### Why Runtime-Checkable Protocols?
- Allows `isinstance()` checks for dynamic dispatch
- Better than abstract base classes for Python
- Works with any object that has the right methods

### Why Separate LangChain Integration?
- No hard dependency on LangChain
- Can be completely removed without affecting core
- Clear boundary between core and extensions

### Why Batch-First Design?
- More efficient for real-world usage
- Single operations are just batch size 1
- Aligns with both `imbed` and LangChain patterns

## Testing Strategy

1. **Unit tests** for each component in isolation
2. **Integration tests** for component interactions
3. **Fallback tests** to ensure graceful degradation
4. **Performance tests** for batch operations
5. **LangChain tests** that skip if not available

This architecture provides a clean, extensible, and maintainable facade system that integrates well with both `imbed` and optional LangChain components while following functional programming principles where appropriate.



# Extras

Take the instructions above with the grain of salt. If they are incomplete, complete them. If they are incorrect, correct them. Also take into account certain design choices that may not be enforced in them.

First of all, anytime there is a hardcoded value that could possibly need to be changed, especially as far as configurations and settings and parameters go, make sure you specify those towards the top of the module as variables that use the values that you see in the instructions above as default, but that will first look at a environmental variable or configuration file (e.g. json) to see if the user chooses a different value. 

Favor using generator functions. For example, if you make lists, don't do it with append, do it 
with a generator function that list then cast to a list (e.g. `(list(gen_func(...)))`). 

Fever, dependency injection, but implemented in a functional paradigm way. 
Functions are full citizens: Use arguments that are callable. Use `functools.partial` where ever it makes sense. 


Also, importantly, use `dol`. Look it up if you can (it's also know as `py2store`). 
Below are a few highlighted tools from `dol`. 

## a few often useful `dol` tools

Often the first data source or target you work with is local files. `dol.Files` gives you
a `MutableMapping` view over a folder so you can read, write, list and delete files using
simple dict-like operations instead of juggling `os.path` and `open`read/write calls. That
makes higher-level code more independent of the underlying storage, so later you can
swap in S3, a database, or another adapter with minimal changes.

Below are three compact, practical tools you will use frequently when working with local
files: `Files` (and `TextFiles`), `filt_iter`, and `wrap_kvs`.

### Files / TextFiles

- What: `Files(rootdir, ...)` (class or instance) produces a MutableMapping where keys
  are relative paths under `rootdir` (strings like `'path/to/file.txt'`) and values are
  the raw bytes of the file contents.
- Why: Treat folders like dicts. You can iterate, check containment, read and write using
  familiar mapping operations.
- Basic contract:
  - Input: a filesystem directory path (rootdir).
  - Output: mapping where
    - keys -> relative file paths (strings)
    - values -> bytes read/written from/to files
  - Error modes: key validation enforces keys are under the rootdir; invalid keys raise KeyError.

Example:

```python
from dol import Files

files = Files('/path/to/root')  # instance wrapping that folder
list(files)               # list of relative file paths (keys)
files['doc.txt'] = b'hello'  # write bytes
print(files['doc.txt'])      # b'hello'
assert 'doc.txt' in files
del files['doc.txt']         # delete file
```

If you prefer automatic text handling (str instead of bytes), use `TextFiles`
which opens files in text mode by default:

```python
from dol import TextFiles
texts = TextFiles('/path/to/root')
texts['notes.txt'] = 'a string'   # writes text
print(texts['notes.txt'])         # reads str
```

### filt_iter — filter the mapping view

- What: `filt_iter` produces a wrapper (class or instance) that restricts the mapping
  to keys satisfying a filter. The filter can be a callable (k -> bool) or an iterable
  of allowed keys. There are convenient helpers like `filt_iter.suffixes`,
  `filt_iter.prefixes`, and `filt_iter.regex` for common cases.
- Why: Make a focused view (e.g. "only .json files") without copying data. Useful for
  pipelines and for composing with other transformations.

Example — only list and access `.json` files:

```python
from dol import Files
from dol.trans import filt_iter

files = Files('/path/to/root')
json_view = filt_iter.suffixes('.json')(files)
list(json_view)           # only .json keys are shown
obj = json_view['data.json']  # behaves like files['data.json'] (same underlying data)
# writing to a non-matching key raises KeyError
try:
    json_view['other.txt'] = b'no'
except KeyError:
    pass
```

You can also use `filt_iter(filt=callable)` to build arbitrary predicates.

### wrap_kvs — add key/value transformations (codecs, codecs-per-key)

- What: `wrap_kvs` (or its helpers like `kv_wrap`, `KeyCodec`, `ValueCodec`) wraps a
  store so you can transparently transform incoming/outgoing keys and values. Common
  uses are decoding bytes into Python objects on read, and encoding them on write.
- Why: Keep serialization and key-layout concerns orthogonal to business logic.

Contract:
  - Inputs: underlying store (class/instance) and transformation functions.
  - Outputs: mapping that applies the transformations:
    - `value_decoder` / `obj_of_data` converts stored format -> Python value on read
    - `value_encoder` / `data_of_obj` converts Python value -> stored format on write
    - `key_of_id` / `id_of_key` transform keys in/out (e.g. add/remove extensions or prefixes)

Example — treat files as JSON objects instead of bytes:

```python
import json
from dol import Files
from dol.trans import wrap_kvs

files = Files('/path/to/root')
json_store = wrap_kvs(
    files,
    value_decoder=lambda b: json.loads(b.decode()),
    value_encoder=lambda obj: json.dumps(obj, indent=2).encode(),
)

json_store['data.json'] = {'a': 1}        # writes pretty JSON bytes
print(json_store['data.json'])           # reads Python dict
```

You can chain `wrap_kvs` and `filt_iter` to get both decoding and focused views. For
example: restrict to `.json` files and expose them as Python dicts:

```python
json_only = filt_iter.suffixes('.json')(json_store)
for k in json_only:
    print(k, type(json_only[k]))  # keys end with .json and values are dicts
```

Notes, tips and edge-cases
- `Files` keys are relative paths; validation ensures keys are under the rootdir.
- `filt_iter` filters the mapping API surface: non-matching writes/reads raise KeyError.
- `wrap_kvs` accepts several parameter names (value_decoder/value_encoder or
  obj_of_data/data_of_obj) — they are aliases; prefer the clearer names in your code.
- Chain small wrappers: Files -> wrap_kvs (codec) -> filt_iter (view) keeps code
  modular and makes swapping storage implementations easy.

That should give an AI the essentials it needs to read and write local files using
`dol` idioms and to compose filters and codecs when implementing higher-level logic.
