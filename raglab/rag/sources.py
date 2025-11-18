"""Source management for RAG content with exclude logic."""

import re
from pathlib import Path
from typing import Set, List, Callable, Optional
from collections.abc import Mapping
from .types import ContentMapping, UpdateTimeMapping, ExcludeFunc, ContentKey
from .decoders import ExtensionBasedDecoder


class FolderSource:
    """
    Manages a folder as a content source with exclude logic.

    Supports three types of exclusions:
    1. Regex patterns for path matching
    2. Explicit file/folder list
    3. Custom exclude function
    """

    def __init__(
        self,
        folder_path: str | Path,
        decoder: Optional[ExtensionBasedDecoder] = None,
        exclude_patterns: List[str] | None = None,
        exclude_paths: List[str | Path] | None = None,
        exclude_func: ExcludeFunc | None = None,
        recursive: bool = True,
        extensions: Set[str] | None = None,
    ):
        """
        Initialize a folder source.

        Args:
            folder_path: Path to the folder
            decoder: Content decoder (defaults to ExtensionBasedDecoder)
            exclude_patterns: List of regex patterns to exclude
            exclude_paths: Explicit list of paths to exclude
            exclude_func: Custom function to determine exclusion
            recursive: Whether to recursively scan subfolders
            extensions: Set of allowed file extensions (e.g., {'.py', '.md'}). None means all.
        """
        self.folder_path = Path(folder_path)
        self.decoder = decoder or ExtensionBasedDecoder()
        self.exclude_patterns = exclude_patterns or []
        self.exclude_paths = set(Path(p) for p in (exclude_paths or []))
        self.exclude_func = exclude_func
        self.recursive = recursive
        self.extensions = set(ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                             for ext in (extensions or set()))

        # Compile regex patterns
        self._compiled_patterns = [re.compile(pattern) for pattern in self.exclude_patterns]

        # Cache for content and update times
        self._content_cache: dict[ContentKey, str] = {}
        self._update_time_cache: dict[ContentKey, float] = {}
        self._last_scan_time: float = 0

    def _should_exclude(self, path: Path) -> bool:
        """
        Determine if a path should be excluded.

        Args:
            path: Path to check

        Returns:
            True if the path should be excluded
        """
        # Check explicit paths
        if path in self.exclude_paths or path.resolve() in self.exclude_paths:
            return True

        # Check regex patterns
        path_str = str(path)
        for pattern in self._compiled_patterns:
            if pattern.search(path_str):
                return True

        # Check custom function
        if self.exclude_func and self.exclude_func(path):
            return True

        # Check extensions
        if self.extensions and path.suffix.lower() not in self.extensions:
            return True

        return False

    def _scan_folder(self) -> List[Path]:
        """
        Scan the folder and return all non-excluded files.

        Returns:
            List of file paths
        """
        files = []

        if not self.folder_path.exists():
            return files

        if not self.folder_path.is_dir():
            raise ValueError(f"{self.folder_path} is not a directory")

        # Use rglob for recursive, glob for non-recursive
        pattern = '**/*' if self.recursive else '*'

        for path in self.folder_path.glob(pattern):
            if path.is_file() and not self._should_exclude(path):
                files.append(path)

        return files

    def get_content_mapping(self) -> ContentMapping:
        """
        Return a mapping of file paths to their content.

        Returns:
            Mapping from file paths to content
        """
        files = self._scan_folder()
        content_map = {}

        for file_path in files:
            # Use relative path as key
            key = str(file_path.relative_to(self.folder_path))
            content = self.decoder(file_path)
            content_map[key] = content
            self._content_cache[key] = content

        return content_map

    def get_update_times(self) -> UpdateTimeMapping:
        """
        Return a mapping of file paths to their last modification times.

        Returns:
            Mapping from file paths to timestamps
        """
        files = self._scan_folder()
        update_times = {}

        for file_path in files:
            # Use relative path as key
            key = str(file_path.relative_to(self.folder_path))
            # Get modification time
            mtime = file_path.stat().st_mtime
            update_times[key] = mtime
            self._update_time_cache[key] = mtime

        return update_times

    def refresh(self) -> None:
        """Refresh the cached content and update times."""
        import time
        self._content_cache.clear()
        self._update_time_cache.clear()
        self._last_scan_time = time.time()

    def add_exclude_pattern(self, pattern: str) -> None:
        """Add a regex pattern to the exclude list."""
        self.exclude_patterns.append(pattern)
        self._compiled_patterns.append(re.compile(pattern))

    def add_exclude_path(self, path: str | Path) -> None:
        """Add a path to the exclude list."""
        self.exclude_paths.add(Path(path))

    def set_exclude_func(self, func: ExcludeFunc) -> None:
        """Set the custom exclude function."""
        self.exclude_func = func


class MultiSource:
    """Combines multiple sources into a single source."""

    def __init__(self, sources: List[FolderSource]):
        """
        Initialize with multiple sources.

        Args:
            sources: List of FolderSource instances
        """
        self.sources = sources

    def get_content_mapping(self) -> ContentMapping:
        """Combine content from all sources."""
        combined = {}
        for i, source in enumerate(self.sources):
            source_content = source.get_content_mapping()
            # Prefix keys to avoid collisions
            for key, value in source_content.items():
                prefixed_key = f"source_{i}/{key}"
                combined[prefixed_key] = value
        return combined

    def get_update_times(self) -> UpdateTimeMapping:
        """Combine update times from all sources."""
        combined = {}
        for i, source in enumerate(self.sources):
            source_times = source.get_update_times()
            # Prefix keys to match content mapping
            for key, value in source_times.items():
                prefixed_key = f"source_{i}/{key}"
                combined[prefixed_key] = value
        return combined

    def refresh(self) -> None:
        """Refresh all sources."""
        for source in self.sources:
            source.refresh()

    def add_source(self, source: FolderSource) -> None:
        """Add a new source."""
        self.sources.append(source)
