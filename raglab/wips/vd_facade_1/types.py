"""Core type definitions for the vd_facade_1 facade system.

Fallback-friendly simple type aliases so the package can be imported
without optional heavy dependencies.
"""

from typing import (
    Protocol,
    runtime_checkable,
    Mapping,
    Sequence,
    Union,
    Iterable,
    Optional,
    Callable,
    Any,
    Dict,
    List,
    Tuple,
)

try:
    from imbed.imbed_types import (
        Segment,
        SegmentKey,
        SegmentMapping,
        Vector,
        VectorMapping,
        PlanarVector,
        PlanarVectorMapping,
    )
except Exception:
    # lightweight aliases
    Segment = str
    SegmentKey = str
    SegmentMapping = Dict[SegmentKey, str]
    Vector = List[float]
    VectorMapping = Dict[SegmentKey, Vector]
    PlanarVector = Vector
    PlanarVectorMapping = VectorMapping

# Component spec types
ComponentSpec = Union[str, dict, Callable]
BatchResult = Union[Mapping[SegmentKey, Any], Sequence[Any]]


@runtime_checkable
class Segmenter(Protocol):
    def __call__(self, text: str) -> Iterable[str]: ...


@runtime_checkable
class Embedder(Protocol):
    def __call__(
        self, segments: Union[SegmentMapping, Iterable[str]]
    ) -> BatchResult: ...


@runtime_checkable
class VectorStore(Protocol):
    def add_batch(
        self, segments: SegmentMapping, vectors: Optional[VectorMapping] = None
    ) -> None: ...
    def search(
        self, query: Union[str, Vector], k: int = 5
    ) -> Sequence[Tuple[SegmentKey, float]]: ...
    def clear(self) -> None: ...
    def __contains__(self, key: SegmentKey) -> bool: ...
    def __len__(self) -> int: ...


@runtime_checkable
class Indexer(Protocol):
    def build(self, vectors: VectorMapping) -> Any: ...
    def search(
        self, query: Vector, k: int = 5
    ) -> Sequence[Tuple[SegmentKey, float]]: ...
    def update(self, key: SegmentKey, vector: Vector) -> None: ...
