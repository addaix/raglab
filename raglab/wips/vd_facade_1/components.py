"""Built-in components for vd_facade_1 (simplified)."""

from typing import Union, Mapping, Sequence, Iterable, Optional
from collections.abc import Mapping as MappingABC
from math import sqrt

from .registry import register_segmenter, register_embedder, register_vector_store
from .types import SegmentMapping, SegmentKey, Vector, VectorMapping


# Simple segmenter: split on newline and sentences
@register_segmenter("simple_sentence")
def simple_sentence_segmenter(text: str) -> list[str]:
    # Very small heuristic splitter
    parts = [p.strip() for p in text.replace("\n", " ").split('.') if p.strip()]
    return [p + '.' for p in parts]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # naive cosine similarity
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@register_embedder("simple_count")
def simple_count_embedder(segments: Union[SegmentMapping, Iterable[str]]):
    """Return very small embeddings: [len(text), unique_chars]"""
    if isinstance(segments, MappingABC):
        return {k: [len(v), len(set(v))] for k, v in segments.items()}

    return [[len(s), len(set(s))] for s in segments]


@register_vector_store("memory")
class MemoryVectorStore:
    """In-memory store that keeps segments and vectors."""

    def __init__(self, embedder=None):
        self.segments: dict[SegmentKey, str] = {}
        self.vectors: dict[SegmentKey, Vector] = {}
        self.embedder = embedder

    def add_batch(
        self, segments: SegmentMapping, vectors: Optional[VectorMapping] = None
    ):
        self.segments.update(segments)
        if vectors is not None:
            self.vectors.update(vectors)
        elif self.embedder is not None:
            computed = self.embedder(segments)
            if isinstance(computed, MappingABC):
                self.vectors.update(computed)
            else:
                # computed is list in same order as segments
                for k, v in zip(segments.keys(), computed):
                    self.vectors[k] = v

    def search(
        self, query: Union[str, Vector], k: int = 5
    ) -> list[tuple[SegmentKey, float]]:
        if not self.vectors:
            return []
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("No embedder configured for text queries")
            qv = self.embedder({"_q": query})
            if isinstance(qv, MappingABC):
                qv = list(qv.values())[0]
            else:
                qv = qv[0]
        else:
            qv = query

        scores = [(k, float(_cosine(qv, v))) for k, v in self.vectors.items()]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def clear(self):
        self.segments.clear()
        self.vectors.clear()

    def __contains__(self, key: SegmentKey) -> bool:
        return key in self.segments

    def __len__(self) -> int:
        return len(self.segments)
