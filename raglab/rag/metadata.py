"""Metadata management for RAG content."""

import os
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from collections.abc import Mapping
from datetime import datetime
import hashlib

from .types import ContentKey, ContentMapping


class MetadataExtractor:
    """Extract metadata from files."""

    def __init__(
        self,
        extract_file_stats: bool = True,
        extract_mime_type: bool = True,
        extract_hash: bool = True,
        extract_custom: Optional[Dict[str, Callable[[Path], Any]]] = None,
    ):
        """
        Initialize metadata extractor.

        Args:
            extract_file_stats: Extract file size, dates, etc.
            extract_mime_type: Extract MIME type
            extract_hash: Compute content hash
            extract_custom: Custom extractors {name: extractor_function}
        """
        self.extract_file_stats = extract_file_stats
        self.extract_mime_type = extract_mime_type
        self.extract_hash = extract_hash
        self.extract_custom = extract_custom or {}

    def extract(self, file_path: Path, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from a file.

        Args:
            file_path: Path to the file
            content: Optional file content (to avoid re-reading)

        Returns:
            Dictionary of metadata
        """
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'extension': file_path.suffix.lower(),
        }

        # File statistics
        if self.extract_file_stats:
            try:
                stats = file_path.stat()
                metadata.update({
                    'file_size': stats.st_size,
                    'created_time': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    'modified_time': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    'accessed_time': datetime.fromtimestamp(stats.st_atime).isoformat(),
                })
            except Exception:
                pass

        # MIME type
        if self.extract_mime_type:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            metadata['mime_type'] = mime_type or 'application/octet-stream'

        # Content hash
        if self.extract_hash and content is not None:
            metadata['content_hash'] = self._compute_hash(content)
        elif self.extract_hash:
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    metadata['file_hash'] = file_hash
            except Exception:
                pass

        # Custom extractors
        for name, extractor in self.extract_custom.items():
            try:
                metadata[name] = extractor(file_path)
            except Exception as e:
                metadata[f'{name}_error'] = str(e)

        return metadata

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class MetadataManager:
    """Manage metadata for content mappings."""

    def __init__(self, extractor: Optional[MetadataExtractor] = None):
        """
        Initialize metadata manager.

        Args:
            extractor: Metadata extractor (defaults to MetadataExtractor())
        """
        self.extractor = extractor or MetadataExtractor()
        self._metadata_cache: Dict[ContentKey, Dict[str, Any]] = {}

    def extract_metadata_mapping(
        self,
        content_mapping: ContentMapping,
        folder_path: Path,
    ) -> Dict[ContentKey, Dict[str, Any]]:
        """
        Extract metadata for all content in a mapping.

        Args:
            content_mapping: Mapping of content keys to content
            folder_path: Base folder path

        Returns:
            Mapping of content keys to metadata
        """
        metadata_mapping = {}

        for key, content in content_mapping.items():
            file_path = folder_path / key
            metadata = self.extractor.extract(file_path, content)
            metadata_mapping[key] = metadata
            self._metadata_cache[key] = metadata

        return metadata_mapping

    def add_metadata(self, key: ContentKey, metadata: Dict[str, Any]) -> None:
        """Add metadata for a content key."""
        if key in self._metadata_cache:
            self._metadata_cache[key].update(metadata)
        else:
            self._metadata_cache[key] = metadata

    def get_metadata(self, key: ContentKey) -> Optional[Dict[str, Any]]:
        """Get metadata for a content key."""
        return self._metadata_cache.get(key)

    def filter_by_metadata(
        self,
        content_mapping: ContentMapping,
        filter_func: Callable[[Dict[str, Any]], bool],
    ) -> ContentMapping:
        """
        Filter content by metadata.

        Args:
            content_mapping: Content mapping to filter
            filter_func: Function that returns True for content to keep

        Returns:
            Filtered content mapping
        """
        filtered = {}

        for key, content in content_mapping.items():
            metadata = self.get_metadata(key)
            if metadata and filter_func(metadata):
                filtered[key] = content

        return filtered

    def get_metadata_mapping(self) -> Dict[ContentKey, Dict[str, Any]]:
        """Get all cached metadata."""
        return dict(self._metadata_cache)

    def clear_cache(self) -> None:
        """Clear metadata cache."""
        self._metadata_cache.clear()


class SegmentMetadata:
    """Manage metadata for segments."""

    def __init__(self):
        """Initialize segment metadata manager."""
        self._segment_metadata: Dict[str, Dict[str, Any]] = {}

    def add_segment_metadata(
        self,
        segment_key: str,
        content_key: ContentKey,
        segment_index: int,
        segment_text: str,
        content_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add metadata for a segment.

        Args:
            segment_key: Unique key for the segment
            content_key: Key of the source content
            segment_index: Index of segment within the content
            segment_text: The segment text
            content_metadata: Optional metadata from the source content
        """
        metadata = {
            'content_key': content_key,
            'segment_index': segment_index,
            'segment_length': len(segment_text),
            'text': segment_text,
        }

        # Inherit from content metadata
        if content_metadata:
            metadata['source_file'] = content_metadata.get('file_path')
            metadata['source_file_size'] = content_metadata.get('file_size')
            metadata['source_modified_time'] = content_metadata.get('modified_time')
            metadata['mime_type'] = content_metadata.get('mime_type')

        self._segment_metadata[segment_key] = metadata

    def get_segment_metadata(self, segment_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a segment."""
        return self._segment_metadata.get(segment_key)

    def get_segments_for_content(self, content_key: ContentKey) -> List[str]:
        """Get all segment keys for a content key."""
        return [
            seg_key for seg_key, meta in self._segment_metadata.items()
            if meta.get('content_key') == content_key
        ]

    def clear_content_segments(self, content_key: ContentKey) -> None:
        """Clear all segments for a content key."""
        to_remove = self.get_segments_for_content(content_key)
        for seg_key in to_remove:
            self._segment_metadata.pop(seg_key, None)

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all segment metadata."""
        return dict(self._segment_metadata)


class TagManager:
    """Manage tags for content."""

    def __init__(self):
        """Initialize tag manager."""
        self._tags: Dict[ContentKey, set[str]] = {}
        self._reverse_index: Dict[str, set[ContentKey]] = {}

    def add_tag(self, key: ContentKey, tag: str) -> None:
        """Add a tag to a content key."""
        if key not in self._tags:
            self._tags[key] = set()
        self._tags[key].add(tag)

        if tag not in self._reverse_index:
            self._reverse_index[tag] = set()
        self._reverse_index[tag].add(key)

    def add_tags(self, key: ContentKey, tags: List[str]) -> None:
        """Add multiple tags to a content key."""
        for tag in tags:
            self.add_tag(key, tag)

    def remove_tag(self, key: ContentKey, tag: str) -> None:
        """Remove a tag from a content key."""
        if key in self._tags:
            self._tags[key].discard(tag)
        if tag in self._reverse_index:
            self._reverse_index[tag].discard(key)

    def get_tags(self, key: ContentKey) -> set[str]:
        """Get all tags for a content key."""
        return self._tags.get(key, set())

    def get_content_by_tag(self, tag: str) -> set[ContentKey]:
        """Get all content keys with a specific tag."""
        return self._reverse_index.get(tag, set())

    def get_content_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
    ) -> set[ContentKey]:
        """
        Get content keys matching tags.

        Args:
            tags: List of tags to match
            match_all: If True, require all tags. If False, require any tag.

        Returns:
            Set of content keys
        """
        if not tags:
            return set()

        tag_sets = [self.get_content_by_tag(tag) for tag in tags]

        if match_all:
            return set.intersection(*tag_sets) if tag_sets else set()
        else:
            return set.union(*tag_sets) if tag_sets else set()

    def auto_tag_by_extension(self, metadata_mapping: Dict[ContentKey, Dict[str, Any]]) -> None:
        """Automatically tag content by file extension."""
        for key, metadata in metadata_mapping.items():
            ext = metadata.get('extension', '').lower()
            if ext:
                self.add_tag(key, f'ext:{ext}')

    def auto_tag_by_directory(
        self,
        metadata_mapping: Dict[ContentKey, Dict[str, Any]],
        folder_path: Path,
    ) -> None:
        """Automatically tag content by directory."""
        for key, metadata in metadata_mapping.items():
            file_path = Path(metadata.get('file_path', ''))
            try:
                rel_path = file_path.relative_to(folder_path)
                for parent in rel_path.parents:
                    if parent != Path('.'):
                        self.add_tag(key, f'dir:{parent}')
            except ValueError:
                pass

    def auto_tag_by_size(self, metadata_mapping: Dict[ContentKey, Dict[str, Any]]) -> None:
        """Automatically tag content by file size."""
        for key, metadata in metadata_mapping.items():
            size = metadata.get('file_size', 0)
            if size < 1024:
                self.add_tag(key, 'size:tiny')
            elif size < 10 * 1024:
                self.add_tag(key, 'size:small')
            elif size < 100 * 1024:
                self.add_tag(key, 'size:medium')
            elif size < 1024 * 1024:
                self.add_tag(key, 'size:large')
            else:
                self.add_tag(key, 'size:huge')
