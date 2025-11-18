"""Core type definitions for the RAG module."""

from typing import Protocol, runtime_checkable, Union, Optional, Any, Dict, List, Callable
from collections.abc import Mapping, Iterable
from pathlib import Path

# Basic types
SourceKey = str  # Unique identifier for a source
ContentKey = str  # Key for content (e.g., file path)
ContentValue = str  # The actual content
Timestamp = float  # Unix timestamp for modification times

# Exclude function type
ExcludeFunc = Callable[[Path], bool]

# Content mapping types
ContentMapping = Mapping[ContentKey, ContentValue]
UpdateTimeMapping = Mapping[ContentKey, Timestamp]
RefreshMapping = Mapping[ContentKey, bool]


@runtime_checkable
class ContentDecoder(Protocol):
    """Protocol for decoding content from files based on extension."""

    def __call__(self, file_path: Path) -> str:
        """Decode a file and return its text content."""
        ...


@runtime_checkable
class Source(Protocol):
    """Protocol for content sources."""

    def get_content_mapping(self) -> ContentMapping:
        """Return a mapping of content keys to content values."""
        ...

    def get_update_times(self) -> UpdateTimeMapping:
        """Return a mapping of content keys to their last update times."""
        ...

    def refresh(self) -> None:
        """Refresh the source to pick up any changes."""
        ...


@runtime_checkable
class RefreshStrategy(Protocol):
    """Protocol for determining which content needs refreshing."""

    def __call__(
        self,
        current_times: UpdateTimeMapping,
        previous_times: UpdateTimeMapping
    ) -> RefreshMapping:
        """Return a mapping indicating which keys need refreshing."""
        ...
