"""Async versions of sources for parallel processing."""

import asyncio
import aiofiles
from pathlib import Path
from typing import List, Set, Optional, Callable
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools

from .types import ContentMapping, UpdateTimeMapping, ContentKey, ExcludeFunc
from .decoders import ExtensionBasedDecoder
from .sources import FolderSource


class AsyncFolderSource(FolderSource):
    """
    Async version of FolderSource for parallel file processing.

    Uses asyncio for I/O-bound operations and process pools for CPU-bound decoding.
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
        max_workers: int = 4,
        use_process_pool: bool = False,
    ):
        """
        Initialize async folder source.

        Args:
            max_workers: Maximum number of parallel workers
            use_process_pool: Use ProcessPoolExecutor instead of ThreadPoolExecutor
            **kwargs: Same as FolderSource
        """
        super().__init__(
            folder_path=folder_path,
            decoder=decoder,
            exclude_patterns=exclude_patterns,
            exclude_paths=exclude_paths,
            exclude_func=exclude_func,
            recursive=recursive,
            extensions=extensions,
        )
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool

    async def _read_file_async(self, file_path: Path) -> tuple[str, str]:
        """
        Read and decode a file asynchronously.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (key, content)
        """
        key = str(file_path.relative_to(self.folder_path))

        try:
            # For simple text files, use async I/O
            if file_path.suffix.lower() in {'.txt', '.md', '.py', '.js', '.json'}:
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
            else:
                # For complex decoders (PDF, DOCX), use thread/process pool
                loop = asyncio.get_event_loop()
                executor_class = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
                with executor_class(max_workers=1) as executor:
                    content = await loop.run_in_executor(
                        executor,
                        self.decoder,
                        file_path
                    )
        except Exception as e:
            content = f"[Error reading {file_path}]: {e}"

        return key, content

    async def get_content_mapping_async(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ContentMapping:
        """
        Get content mapping asynchronously with parallel processing.

        Args:
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            Mapping from file paths to content
        """
        files = self._scan_folder()
        total = len(files)

        content_map = {}

        # Process files in batches to avoid overwhelming the system
        batch_size = self.max_workers * 2

        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]

            # Create tasks for this batch
            tasks = [self._read_file_async(file_path) for file_path in batch]

            # Execute batch in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    continue
                key, content = result
                content_map[key] = content
                self._content_cache[key] = content

            # Call progress callback
            if progress_callback:
                current = min(i + batch_size, total)
                progress_callback(current, total)

        return content_map

    def get_content_mapping(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ContentMapping:
        """
        Synchronous wrapper for get_content_mapping_async.

        Args:
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            Mapping from file paths to content
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.get_content_mapping_async(progress_callback)
        )

    async def _get_file_mtime_async(self, file_path: Path) -> tuple[str, float]:
        """Get file modification time asynchronously."""
        key = str(file_path.relative_to(self.folder_path))
        mtime = file_path.stat().st_mtime
        return key, mtime

    async def get_update_times_async(self) -> UpdateTimeMapping:
        """Get update times asynchronously."""
        files = self._scan_folder()

        tasks = [self._get_file_mtime_async(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        update_times = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            key, mtime = result
            update_times[key] = mtime
            self._update_time_cache[key] = mtime

        return update_times

    def get_update_times(self) -> UpdateTimeMapping:
        """Synchronous wrapper for get_update_times_async."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.get_update_times_async())


class ParallelBatchProcessor:
    """
    Utility for parallel batch processing of content.

    Useful for embedding and other CPU-intensive operations.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = 4,
        use_process_pool: bool = False,
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Size of each batch
            max_workers: Maximum parallel workers
            use_process_pool: Use processes instead of threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool

    async def process_batches_async(
        self,
        items: List,
        process_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List:
        """
        Process items in parallel batches.

        Args:
            items: Items to process
            process_func: Function to apply to each batch
            progress_callback: Optional progress callback

        Returns:
            List of processed results
        """
        total = len(items)
        results = []

        executor_class = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()

            # Create batches
            batches = [
                items[i:i + self.batch_size]
                for i in range(0, len(items), self.batch_size)
            ]

            # Process batches in parallel
            tasks = [
                loop.run_in_executor(executor, process_func, batch)
                for batch in batches
            ]

            # Gather results with progress tracking
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                batch_result = await coro
                results.extend(batch_result)

                if progress_callback:
                    current = min((i + 1) * self.batch_size, total)
                    progress_callback(current, total)

        return results

    def process_batches(
        self,
        items: List,
        process_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List:
        """Synchronous wrapper for process_batches_async."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.process_batches_async(items, process_func, progress_callback)
        )


def make_async_embedder(sync_embedder: Callable, max_workers: int = 2):
    """
    Convert a synchronous embedder to async with batching.

    Args:
        sync_embedder: Synchronous embedding function
        max_workers: Number of parallel workers

    Returns:
        Async embedder function
    """
    processor = ParallelBatchProcessor(
        batch_size=32,
        max_workers=max_workers,
        use_process_pool=False  # Most embedders use GPU/network
    )

    def async_embedder(segments: List[str]) -> List[List[float]]:
        """Async embedder wrapper."""
        # Split into batches and process in parallel
        def embed_batch(batch):
            return sync_embedder(batch)

        return processor.process_batches(segments, embed_batch)

    return async_embedder
