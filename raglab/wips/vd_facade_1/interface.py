"""Main facade interface for vd_facade_1"""

from dataclasses import dataclass, field
from typing import Optional, Union, Mapping, Any

from .registry import get_component, segmenters, embedders, vector_stores
from .types import SegmentMapping, SegmentKey, Vector, VectorMapping
from .components import simple_sentence_segmenter


@dataclass
class VectorSearchEngine:
    segmenter: Any = "simple_sentence"
    embedder: Any = "simple_count"
    vector_store: Any = "memory"

    auto_segment: bool = True
    auto_embed: bool = True

    _segmenter: Any = field(init=False, default=None)
    _embedder: Any = field(init=False, default=None)
    _vector_store: Any = field(init=False, default=None)

    def __post_init__(self):
        self._segmenter = get_component(segmenters, self.segmenter)
        self._embedder = get_component(embedders, self.embedder)
        store_kwargs = {"embedder": self._embedder}
        self._vector_store = get_component(
            vector_stores, self.vector_store, **store_kwargs
        )

    def add_documents(
        self,
        documents: Union[str, list[str], Mapping[str, str], SegmentMapping],
        *,
        segment: Optional[bool] = None,
        embed: Optional[bool] = None,
    ):
        should_segment = segment if segment is not None else self.auto_segment
        should_embed = embed if embed is not None else self.auto_embed

        if isinstance(documents, str):
            documents = {"0": documents}
        elif isinstance(documents, list):
            documents = {str(i): d for i, d in enumerate(documents)}

        if should_segment and self._segmenter:
            segments = {}
            for doc_id, text in documents.items():
                for i, seg in enumerate(self._segmenter(text)):
                    segments[f"{doc_id}_{i}"] = seg
        else:
            segments = documents  # assume mapping

        self._vector_store.add_batch(segments)

    def search(
        self, query: Union[str, Vector], k: int = 5, return_segments: bool = True
    ):
        results = self._vector_store.search(query, k)
        if return_segments and hasattr(self._vector_store, "segments"):
            return [
                (self._vector_store.segments.get(key, key), score)
                for key, score in results
            ]
        return results

    def clear(self):
        self._vector_store.clear()

    @property
    def size(self) -> int:
        return len(self._vector_store)

    @classmethod
    def create_with_best_backend(cls, **kwargs):
        if "memory" in vector_stores:
            return cls(vector_store="memory", **kwargs)
        return cls(**kwargs)
