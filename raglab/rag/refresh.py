"""Refresh mapping and strategies for determining what content needs updating."""

from typing import Dict, Set
from collections.abc import Mapping
from .types import UpdateTimeMapping, RefreshMapping, ContentKey


class SimpleRefreshStrategy:
    """
    Simple refresh strategy that compares modification times.

    Returns True for a key if:
    - The key is new (not in previous times)
    - The modification time has changed
    """

    def __call__(
        self,
        current_times: UpdateTimeMapping,
        previous_times: UpdateTimeMapping
    ) -> RefreshMapping:
        """
        Determine which keys need refreshing.

        Args:
            current_times: Current modification times
            previous_times: Previous modification times

        Returns:
            Mapping indicating which keys need refreshing
        """
        refresh_map = {}

        # Check all current keys
        for key, current_time in current_times.items():
            if key not in previous_times:
                # New key
                refresh_map[key] = True
            elif current_time > previous_times[key]:
                # Modified key
                refresh_map[key] = True
            else:
                # Unchanged
                refresh_map[key] = False

        # Handle deleted keys (in previous but not in current)
        for key in previous_times:
            if key not in current_times:
                # Mark as needing refresh (for deletion)
                refresh_map[key] = True

        return refresh_map


class ThresholdRefreshStrategy:
    """
    Refresh strategy with a time threshold.

    Only refreshes if the time difference exceeds a threshold.
    """

    def __init__(self, threshold_seconds: float = 60.0):
        """
        Initialize with a time threshold.

        Args:
            threshold_seconds: Minimum time difference to trigger refresh
        """
        self.threshold = threshold_seconds

    def __call__(
        self,
        current_times: UpdateTimeMapping,
        previous_times: UpdateTimeMapping
    ) -> RefreshMapping:
        """Determine which keys need refreshing based on threshold."""
        refresh_map = {}

        for key, current_time in current_times.items():
            if key not in previous_times:
                # New key
                refresh_map[key] = True
            elif current_time - previous_times[key] > self.threshold:
                # Modified beyond threshold
                refresh_map[key] = True
            else:
                # Within threshold
                refresh_map[key] = False

        # Handle deleted keys
        for key in previous_times:
            if key not in current_times:
                refresh_map[key] = True

        return refresh_map


class BatchRefreshStrategy:
    """
    Refresh strategy that batches updates.

    Only triggers refresh if a minimum number/percentage of files have changed.
    """

    def __init__(
        self,
        min_changed_count: int | None = None,
        min_changed_percentage: float | None = None
    ):
        """
        Initialize with batch criteria.

        Args:
            min_changed_count: Minimum number of changed files to trigger batch refresh
            min_changed_percentage: Minimum percentage of changed files (0.0 to 1.0)
        """
        self.min_changed_count = min_changed_count
        self.min_changed_percentage = min_changed_percentage

    def __call__(
        self,
        current_times: UpdateTimeMapping,
        previous_times: UpdateTimeMapping
    ) -> RefreshMapping:
        """Determine if batch refresh criteria are met."""
        # First, identify changed keys
        changed_keys = set()

        for key, current_time in current_times.items():
            if key not in previous_times or current_time > previous_times[key]:
                changed_keys.add(key)

        # Add deleted keys
        for key in previous_times:
            if key not in current_times:
                changed_keys.add(key)

        # Check if batch criteria are met
        total_keys = len(current_times) + len(previous_times)
        changed_count = len(changed_keys)
        changed_percentage = changed_count / total_keys if total_keys > 0 else 0

        should_refresh_batch = False

        if self.min_changed_count is not None and changed_count >= self.min_changed_count:
            should_refresh_batch = True

        if self.min_changed_percentage is not None and changed_percentage >= self.min_changed_percentage:
            should_refresh_batch = True

        # If batch refresh is triggered, refresh all changed keys
        if should_refresh_batch:
            return {key: (key in changed_keys) for key in set(current_times.keys()) | set(previous_times.keys())}
        else:
            # No refresh
            return {key: False for key in set(current_times.keys()) | set(previous_times.keys())}


class RefreshMappingManager:
    """
    Manages refresh mappings and tracks what needs updating.

    This class maintains the state of previous update times and computes
    refresh mappings on demand.
    """

    def __init__(self, strategy: SimpleRefreshStrategy | None = None):
        """
        Initialize the refresh mapping manager.

        Args:
            strategy: Refresh strategy to use (defaults to SimpleRefreshStrategy)
        """
        self.strategy = strategy or SimpleRefreshStrategy()
        self.previous_times: Dict[ContentKey, float] = {}

    def get_refresh_mapping(self, current_times: UpdateTimeMapping) -> RefreshMapping:
        """
        Get the refresh mapping based on current update times.

        Args:
            current_times: Current modification times

        Returns:
            Mapping indicating which keys need refreshing
        """
        refresh_map = self.strategy(current_times, self.previous_times)
        return refresh_map

    def update_times(self, current_times: UpdateTimeMapping) -> None:
        """
        Update the stored previous times.

        Args:
            current_times: Current modification times to store
        """
        self.previous_times = dict(current_times)

    def get_keys_to_refresh(self, current_times: UpdateTimeMapping) -> Set[ContentKey]:
        """
        Get the set of keys that need refreshing.

        Args:
            current_times: Current modification times

        Returns:
            Set of keys that need refreshing
        """
        refresh_map = self.get_refresh_mapping(current_times)
        return {key for key, should_refresh in refresh_map.items() if should_refresh}

    def get_keys_to_delete(self, current_times: UpdateTimeMapping) -> Set[ContentKey]:
        """
        Get the set of keys that have been deleted.

        Args:
            current_times: Current modification times

        Returns:
            Set of keys that have been deleted
        """
        deleted_keys = set(self.previous_times.keys()) - set(current_times.keys())
        return deleted_keys

    def get_new_keys(self, current_times: UpdateTimeMapping) -> Set[ContentKey]:
        """
        Get the set of new keys.

        Args:
            current_times: Current modification times

        Returns:
            Set of new keys
        """
        new_keys = set(current_times.keys()) - set(self.previous_times.keys())
        return new_keys

    def get_modified_keys(self, current_times: UpdateTimeMapping) -> Set[ContentKey]:
        """
        Get the set of modified keys (existing keys with changed times).

        Args:
            current_times: Current modification times

        Returns:
            Set of modified keys
        """
        modified = set()
        for key, current_time in current_times.items():
            if key in self.previous_times and current_time > self.previous_times[key]:
                modified.add(key)
        return modified
