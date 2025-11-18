"""Advanced features: query enhancements, hierarchical indexing, monitoring, multi-modal."""

from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import defaultdict
from datetime import datetime
import time


# ===== Query Enhancements =====

class QueryEnhancer:
    """Enhance queries for better retrieval."""

    def __init__(self, embedder: Optional[Callable] = None):
        """Initialize query enhancer."""
        self.embedder = embedder

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms/related terms."""
        # Simple expansion - add common variations
        expansions = [query]

        # Add question variations
        if '?' not in query:
            expansions.append(query + '?')
            expansions.append(f"what is {query}")
            expansions.append(f"how to {query}")

        return expansions

    def rewrite_query(self, query: str) -> str:
        """Rewrite query for better matching."""
        # Remove filler words
        fillers = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        words = query.lower().split()
        filtered = [w for w in words if w not in fillers]
        return ' '.join(filtered)


class HybridSearch:
    """Hybrid search combining dense and sparse retrieval."""

    def __init__(
        self,
        vector_search_func: Callable,
        bm25_search_func: Optional[Callable] = None,
        alpha: float = 0.5,
    ):
        """
        Initialize hybrid search.

        Args:
            vector_search_func: Dense vector search function
            bm25_search_func: Sparse BM25 search function
            alpha: Weight for vector search (1-alpha for BM25)
        """
        self.vector_search = vector_search_func
        self.bm25_search = bm25_search_func
        self.alpha = alpha

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search."""
        # Vector search
        vector_results = self.vector_search(query, k=k*2)

        # If no BM25, return vector results
        if not self.bm25_search:
            return vector_results[:k]

        # BM25 search
        bm25_results = self.bm25_search(query, k=k*2)

        # Combine scores
        combined_scores = defaultdict(float)

        for result in vector_results:
            key = result['key']
            combined_scores[key] += self.alpha * result['score']

        for result in bm25_results:
            key = result['key']
            combined_scores[key] += (1 - self.alpha) * result['score']

        # Sort by combined score
        sorted_keys = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Build results
        all_results = {r['key']: r for r in vector_results + bm25_results}
        final_results = []
        for key, score in sorted_keys[:k]:
            result = all_results.get(key, {'key': key})
            result['score'] = score
            final_results.append(result)

        return final_results


class Reranker:
    """Rerank search results."""

    def __init__(self, rerank_model: Optional[Callable] = None):
        """Initialize reranker."""
        self.rerank_model = rerank_model

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank results."""
        if not self.rerank_model:
            return results[:k]

        # Use model to compute relevance scores
        texts = [r.get('text', '') for r in results]
        scores = self.rerank_model(query, texts)

        # Update scores
        for result, score in zip(results, scores):
            result['rerank_score'] = score

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)

        return reranked[:k]


# ===== Hierarchical Indexing =====

class HierarchicalPipeline:
    """Multi-level indexing at different granularities."""

    def __init__(
        self,
        embedder: Optional[Callable] = None,
        vector_store_factory: Optional[Callable] = None,
    ):
        """Initialize hierarchical pipeline."""
        from .pipeline import RAGPipeline

        # Document level (large chunks)
        self.document_level = RAGPipeline(
            source=None,
            embedder=embedder,
            vector_store=vector_store_factory("document") if vector_store_factory else None,
            chunk_size=2000,
            chunk_overlap=200,
        )

        # Paragraph level (medium chunks)
        self.paragraph_level = RAGPipeline(
            source=None,
            embedder=embedder,
            vector_store=vector_store_factory("paragraph") if vector_store_factory else None,
            chunk_size=400,
            chunk_overlap=50,
        )

        # Sentence level (small chunks)
        self.sentence_level = RAGPipeline(
            source=None,
            embedder=embedder,
            vector_store=vector_store_factory("sentence") if vector_store_factory else None,
            chunk_size=100,
            chunk_overlap=10,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        strategy: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Search across levels.

        Args:
            query: Search query
            k: Number of results
            strategy: "auto", "document", "paragraph", "sentence", "fusion"

        Returns:
            Search results
        """
        if strategy == "document":
            return self.document_level.search(query, k=k)
        elif strategy == "paragraph":
            return self.paragraph_level.search(query, k=k)
        elif strategy == "sentence":
            return self.sentence_level.search(query, k=k)
        elif strategy == "fusion":
            return self._fusion_search(query, k)
        else:  # auto
            return self._auto_search(query, k)

    def _fusion_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fusion search across all levels."""
        # Get results from all levels
        doc_results = self.document_level.search(query, k=k)
        para_results = self.paragraph_level.search(query, k=k)
        sent_results = self.sentence_level.search(query, k=k)

        # Combine with weights
        all_results = {}
        for r in doc_results:
            all_results[r['key']] = r
            all_results[r['key']]['combined_score'] = r['score'] * 0.5

        for r in para_results:
            if r['key'] in all_results:
                all_results[r['key']]['combined_score'] += r['score'] * 0.3
            else:
                all_results[r['key']] = r
                all_results[r['key']]['combined_score'] = r['score'] * 0.3

        for r in sent_results:
            if r['key'] in all_results:
                all_results[r['key']]['combined_score'] += r['score'] * 0.2
            else:
                all_results[r['key']] = r
                all_results[r['key']]['combined_score'] = r['score'] * 0.2

        # Sort by combined score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )

        return sorted_results[:k]

    def _auto_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Auto-select level based on query."""
        # Simple heuristic: short queries use sentence, long use document
        if len(query.split()) < 5:
            return self.sentence_level.search(query, k=k)
        elif len(query.split()) < 15:
            return self.paragraph_level.search(query, k=k)
        else:
            return self.document_level.search(query, k=k)


# ===== Monitoring & Metrics =====

class PipelineMonitor:
    """Monitor pipeline performance and collect metrics."""

    def __init__(self):
        """Initialize monitor."""
        self.metrics = defaultdict(list)
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.time()

    def end(self) -> None:
        """End monitoring."""
        self.end_time = time.time()

    def record(self, metric_name: str, value: Any) -> None:
        """Record a metric."""
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now().isoformat(),
        })

    def increment(self, counter_name: str) -> None:
        """Increment a counter."""
        current = self.get_latest(counter_name, 0)
        self.record(counter_name, current + 1)

    def get_latest(self, metric_name: str, default: Any = None) -> Any:
        """Get latest value of a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]['value']
        return default

    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        stats = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time if self.end_time and self.start_time else None,
        }

        # Compute aggregates
        for metric_name, values in self.metrics.items():
            if values:
                numeric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]
                if numeric_values:
                    stats[f'{metric_name}_total'] = sum(numeric_values)
                    stats[f'{metric_name}_avg'] = sum(numeric_values) / len(numeric_values)
                    stats[f'{metric_name}_min'] = min(numeric_values)
                    stats[f'{metric_name}_max'] = max(numeric_values)
                stats[f'{metric_name}_count'] = len(values)
                stats[f'{metric_name}_latest'] = values[-1]['value']

        return stats

    def get_metrics(self) -> Dict[str, List[Dict]]:
        """Get raw metrics."""
        return dict(self.metrics)


# ===== Multi-modal Support =====

class ImageSource:
    """Source for images with caption generation."""

    def __init__(
        self,
        folder_path: str,
        image_captioner: Optional[Callable] = None,
        ocr_extractor: Optional[Callable] = None,
    ):
        """
        Initialize image source.

        Args:
            folder_path: Path to folder with images
            image_captioner: Function to generate captions from images
            ocr_extractor: Function to extract text from images (OCR)
        """
        from pathlib import Path
        self.folder_path = Path(folder_path)
        self.image_captioner = image_captioner
        self.ocr_extractor = ocr_extractor

    def get_content_mapping(self) -> Dict[str, str]:
        """Get content from images."""
        content_map = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        for img_path in self.folder_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                key = str(img_path.relative_to(self.folder_path))

                # Generate caption
                caption = ""
                if self.image_captioner:
                    try:
                        caption = self.image_captioner(str(img_path))
                    except Exception as e:
                        caption = f"[Caption error: {e}]"

                # Extract text via OCR
                ocr_text = ""
                if self.ocr_extractor:
                    try:
                        ocr_text = self.ocr_extractor(str(img_path))
                    except Exception as e:
                        ocr_text = f"[OCR error: {e}]"

                # Combine caption and OCR text
                content = f"Caption: {caption}\nText: {ocr_text}" if caption or ocr_text else f"Image: {key}"
                content_map[key] = content

        return content_map

    def get_update_times(self) -> Dict[str, float]:
        """Get update times for images."""
        update_times = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        for img_path in self.folder_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                key = str(img_path.relative_to(self.folder_path))
                update_times[key] = img_path.stat().st_mtime

        return update_times

    def refresh(self) -> None:
        """Refresh (no caching for now)."""
        pass


class AudioSource:
    """Source for audio files with transcription."""

    def __init__(
        self,
        folder_path: str,
        transcriber: Optional[Callable] = None,
    ):
        """
        Initialize audio source.

        Args:
            folder_path: Path to folder with audio files
            transcriber: Function to transcribe audio files
        """
        from pathlib import Path
        self.folder_path = Path(folder_path)
        self.transcriber = transcriber

    def get_content_mapping(self) -> Dict[str, str]:
        """Get content from audio via transcription."""
        content_map = {}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}

        for audio_path in self.folder_path.rglob('*'):
            if audio_path.suffix.lower() in audio_extensions:
                key = str(audio_path.relative_to(self.folder_path))

                if self.transcriber:
                    try:
                        transcript = self.transcriber(str(audio_path))
                        content_map[key] = transcript
                    except Exception as e:
                        content_map[key] = f"[Transcription error: {e}]"
                else:
                    content_map[key] = f"Audio file: {key}"

        return content_map

    def get_update_times(self) -> Dict[str, float]:
        """Get update times for audio files."""
        update_times = {}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}

        for audio_path in self.folder_path.rglob('*'):
            if audio_path.suffix.lower() in audio_extensions:
                key = str(audio_path.relative_to(self.folder_path))
                update_times[key] = audio_path.stat().st_mtime

        return update_times

    def refresh(self) -> None:
        """Refresh."""
        pass
