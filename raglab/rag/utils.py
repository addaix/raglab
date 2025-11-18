"""Utility functions for RAG module: deduplication, state persistence, etc."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections.abc import Mapping
from datetime import datetime


# ===== Content Deduplication =====

class DeduplicationStrategy:
    """Detect and handle duplicate content."""

    def __init__(self, method: str = "hash"):
        """
        Initialize deduplication strategy.

        Args:
            method: Deduplication method ("hash", "similarity")
        """
        self.method = method

    def find_duplicates(
        self,
        content_mapping: Mapping[str, str]
    ) -> Dict[str, List[str]]:
        """
        Find duplicate content.

        Args:
            content_mapping: Mapping of keys to content

        Returns:
            Dictionary mapping content hash to list of duplicate keys
        """
        if self.method == "hash":
            return self._find_by_hash(content_mapping)
        elif self.method == "similarity":
            return self._find_by_similarity(content_mapping)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _find_by_hash(
        self,
        content_mapping: Mapping[str, str]
    ) -> Dict[str, List[str]]:
        """Find duplicates by content hash."""
        hash_to_keys: Dict[str, List[str]] = {}

        for key, content in content_mapping.items():
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if content_hash not in hash_to_keys:
                hash_to_keys[content_hash] = []
            hash_to_keys[content_hash].append(key)

        # Only return duplicates (hash with multiple keys)
        return {h: keys for h, keys in hash_to_keys.items() if len(keys) > 1}

    def _find_by_similarity(
        self,
        content_mapping: Mapping[str, str],
        threshold: float = 0.95
    ) -> Dict[str, List[str]]:
        """Find duplicates by similarity (requires embedder)."""
        # This would require embeddings and cosine similarity
        # Simplified version using hash for now
        return self._find_by_hash(content_mapping)

    def remove_duplicates(
        self,
        content_mapping: Mapping[str, str],
        strategy: str = "keep_first"
    ) -> Mapping[str, str]:
        """
        Remove duplicates from content mapping.

        Args:
            content_mapping: Original mapping
            strategy: "keep_first", "keep_shortest_key", "keep_longest_content"

        Returns:
            Deduplicated mapping
        """
        duplicates = self.find_duplicates(content_mapping)
        to_remove = set()

        for hash_key, dup_keys in duplicates.items():
            if strategy == "keep_first":
                # Keep first, remove rest
                to_remove.update(dup_keys[1:])
            elif strategy == "keep_shortest_key":
                # Keep key with shortest name
                shortest = min(dup_keys, key=len)
                to_remove.update([k for k in dup_keys if k != shortest])
            elif strategy == "keep_longest_content":
                # Keep the one with longest content
                longest = max(dup_keys, key=lambda k: len(content_mapping[k]))
                to_remove.update([k for k in dup_keys if k != longest])

        return {k: v for k, v in content_mapping.items() if k not in to_remove}


# ===== State Persistence =====

class StatePersistence:
    """Handle state persistence for RAG pipeline."""

    @staticmethod
    def save_state(
        state: Dict[str, Any],
        file_path: str | Path,
        format: str = "json"
    ) -> None:
        """
        Save pipeline state to file.

        Args:
            state: State dictionary
            file_path: Path to save state
            format: Format ("json" or "pickle")
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        state['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'format': format,
        }

        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        elif format == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def load_state(
        file_path: str | Path,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Load pipeline state from file.

        Args:
            file_path: Path to state file
            format: Format ("json" or "pickle")

        Returns:
            State dictionary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"State file not found: {file_path}")

        if format == "json":
            with open(file_path, 'r') as f:
                state = json.load(f)
        elif format == "pickle":
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")

        return state


class CheckpointManager:
    """Manage checkpoints for long-running operations."""

    def __init__(self, checkpoint_dir: str | Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """Save a checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        StatePersistence.save_state(state, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Load a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        return StatePersistence.load_state(checkpoint_path)

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return [p.stem for p in self.checkpoint_dir.glob("checkpoint_*.json")]

    def delete_checkpoint(self, checkpoint_name: str) -> None:
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()


# ===== Content Validation =====

class ContentValidator:
    """Validate content quality."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: Optional[int] = None,
        check_meaningful: bool = True,
    ):
        """
        Initialize validator.

        Args:
            min_length: Minimum content length
            max_length: Maximum content length
            check_meaningful: Check for meaningful text
        """
        self.min_length = min_length
        self.max_length = max_length
        self.check_meaningful = check_meaningful

    def validate(self, content: str) -> tuple[bool, List[str]]:
        """
        Validate content.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Length checks
        if len(content) < self.min_length:
            issues.append(f"Content too short (min: {self.min_length})")

        if self.max_length and len(content) > self.max_length:
            issues.append(f"Content too long (max: {self.max_length})")

        # Meaningful text check
        if self.check_meaningful and not self._has_meaningful_text(content):
            issues.append("No meaningful text found")

        return len(issues) == 0, issues

    def _has_meaningful_text(self, content: str) -> bool:
        """Check if content has meaningful text."""
        # Remove whitespace
        stripped = content.strip()
        if not stripped:
            return False

        # Check for alphanumeric characters
        alpha_count = sum(c.isalnum() for c in stripped)
        return alpha_count > 5


# ===== Incremental Embedding Cache =====

class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, cache_file: Optional[str | Path] = None):
        """
        Initialize embedding cache.

        Args:
            cache_file: Path to cache file (optional)
        """
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: Dict[str, List[float]] = {}

        if self.cache_file and self.cache_file.exists():
            self.load()

    def get(self, content_hash: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        return self._cache.get(content_hash)

    def put(self, content_hash: str, embedding: List[float]) -> None:
        """Put embedding in cache."""
        self._cache[content_hash] = embedding

    def has(self, content_hash: str) -> bool:
        """Check if embedding is cached."""
        return content_hash in self._cache

    def compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get_or_compute(
        self,
        content: str,
        compute_func: callable
    ) -> List[float]:
        """Get embedding from cache or compute it."""
        content_hash = self.compute_hash(content)

        if self.has(content_hash):
            return self.get(content_hash)
        else:
            embedding = compute_func(content)
            self.put(content_hash, embedding)
            return embedding

    def save(self) -> None:
        """Save cache to file."""
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f)

    def load(self) -> None:
        """Load cache from file."""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self._cache = json.load(f)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)
