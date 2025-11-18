"""Tests for refresh mapping functionality."""

import pytest
from raglab.rag.refresh import (
    SimpleRefreshStrategy,
    ThresholdRefreshStrategy,
    BatchRefreshStrategy,
    RefreshMappingManager,
)


class TestSimpleRefreshStrategy:
    """Test SimpleRefreshStrategy."""

    def test_new_keys(self):
        """Test that new keys are marked for refresh."""
        strategy = SimpleRefreshStrategy()
        previous_times = {"file1.txt": 100.0}
        current_times = {"file1.txt": 100.0, "file2.txt": 200.0}

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is False  # Unchanged
        assert refresh_map["file2.txt"] is True   # New

    def test_modified_keys(self):
        """Test that modified keys are marked for refresh."""
        strategy = SimpleRefreshStrategy()
        previous_times = {"file1.txt": 100.0}
        current_times = {"file1.txt": 150.0}  # Modified

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is True

    def test_deleted_keys(self):
        """Test that deleted keys are marked for refresh."""
        strategy = SimpleRefreshStrategy()
        previous_times = {"file1.txt": 100.0, "file2.txt": 200.0}
        current_times = {"file1.txt": 100.0}  # file2.txt deleted

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is False
        assert refresh_map["file2.txt"] is True  # Deleted

    def test_unchanged_keys(self):
        """Test that unchanged keys are not marked for refresh."""
        strategy = SimpleRefreshStrategy()
        previous_times = {"file1.txt": 100.0}
        current_times = {"file1.txt": 100.0}

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is False


class TestThresholdRefreshStrategy:
    """Test ThresholdRefreshStrategy."""

    def test_within_threshold(self):
        """Test that changes within threshold are not refreshed."""
        strategy = ThresholdRefreshStrategy(threshold_seconds=60.0)
        previous_times = {"file1.txt": 100.0}
        current_times = {"file1.txt": 130.0}  # 30 seconds change

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is False

    def test_exceeds_threshold(self):
        """Test that changes exceeding threshold are refreshed."""
        strategy = ThresholdRefreshStrategy(threshold_seconds=60.0)
        previous_times = {"file1.txt": 100.0}
        current_times = {"file1.txt": 170.0}  # 70 seconds change

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is True

    def test_new_keys_refreshed(self):
        """Test that new keys are always refreshed regardless of threshold."""
        strategy = ThresholdRefreshStrategy(threshold_seconds=60.0)
        previous_times = {}
        current_times = {"file1.txt": 100.0}

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is True


class TestBatchRefreshStrategy:
    """Test BatchRefreshStrategy."""

    def test_min_changed_count(self):
        """Test batch refresh based on minimum changed count."""
        strategy = BatchRefreshStrategy(min_changed_count=2)

        # Only 1 file changed - should not trigger batch refresh
        previous_times = {"file1.txt": 100.0, "file2.txt": 200.0}
        current_times = {"file1.txt": 150.0, "file2.txt": 200.0}

        refresh_map = strategy(current_times, previous_times)

        assert all(not should_refresh for should_refresh in refresh_map.values())

        # 2 files changed - should trigger batch refresh
        current_times2 = {"file1.txt": 150.0, "file2.txt": 250.0}

        refresh_map2 = strategy(current_times2, previous_times)

        assert refresh_map2["file1.txt"] is True
        assert refresh_map2["file2.txt"] is True

    def test_min_changed_percentage(self):
        """Test batch refresh based on minimum changed percentage."""
        strategy = BatchRefreshStrategy(min_changed_percentage=0.5)

        # 1 out of 2 changed (50%) - should trigger
        previous_times = {"file1.txt": 100.0, "file2.txt": 200.0}
        current_times = {"file1.txt": 150.0, "file2.txt": 200.0}

        refresh_map = strategy(current_times, previous_times)

        assert refresh_map["file1.txt"] is True

        # 1 out of 3 changed (33%) - should not trigger
        previous_times2 = {"file1.txt": 100.0, "file2.txt": 200.0, "file3.txt": 300.0}
        current_times2 = {"file1.txt": 150.0, "file2.txt": 200.0, "file3.txt": 300.0}

        refresh_map2 = strategy(current_times2, previous_times2)

        assert all(not should_refresh for should_refresh in refresh_map2.values())


class TestRefreshMappingManager:
    """Test RefreshMappingManager."""

    def test_get_refresh_mapping(self):
        """Test getting refresh mapping."""
        manager = RefreshMappingManager()

        # First call - all new
        current_times = {"file1.txt": 100.0, "file2.txt": 200.0}
        refresh_map = manager.get_refresh_mapping(current_times)

        assert refresh_map["file1.txt"] is True
        assert refresh_map["file2.txt"] is True

        # Update times
        manager.update_times(current_times)

        # Second call - nothing changed
        refresh_map2 = manager.get_refresh_mapping(current_times)

        assert refresh_map2["file1.txt"] is False
        assert refresh_map2["file2.txt"] is False

    def test_get_keys_to_refresh(self):
        """Test getting keys that need refreshing."""
        manager = RefreshMappingManager()

        current_times = {"file1.txt": 100.0, "file2.txt": 200.0}
        manager.update_times(current_times)

        # Modify file1
        current_times2 = {"file1.txt": 150.0, "file2.txt": 200.0}
        keys_to_refresh = manager.get_keys_to_refresh(current_times2)

        assert "file1.txt" in keys_to_refresh
        assert "file2.txt" not in keys_to_refresh

    def test_get_new_keys(self):
        """Test getting new keys."""
        manager = RefreshMappingManager()

        current_times = {"file1.txt": 100.0}
        manager.update_times(current_times)

        # Add file2
        current_times2 = {"file1.txt": 100.0, "file2.txt": 200.0}
        new_keys = manager.get_new_keys(current_times2)

        assert "file2.txt" in new_keys
        assert "file1.txt" not in new_keys

    def test_get_deleted_keys(self):
        """Test getting deleted keys."""
        manager = RefreshMappingManager()

        current_times = {"file1.txt": 100.0, "file2.txt": 200.0}
        manager.update_times(current_times)

        # Delete file2
        current_times2 = {"file1.txt": 100.0}
        deleted_keys = manager.get_keys_to_delete(current_times2)

        assert "file2.txt" in deleted_keys
        assert "file1.txt" not in deleted_keys

    def test_get_modified_keys(self):
        """Test getting modified keys."""
        manager = RefreshMappingManager()

        current_times = {"file1.txt": 100.0, "file2.txt": 200.0}
        manager.update_times(current_times)

        # Modify file1
        current_times2 = {"file1.txt": 150.0, "file2.txt": 200.0}
        modified_keys = manager.get_modified_keys(current_times2)

        assert "file1.txt" in modified_keys
        assert "file2.txt" not in modified_keys

    def test_custom_strategy(self):
        """Test using a custom refresh strategy."""
        strategy = ThresholdRefreshStrategy(threshold_seconds=60.0)
        manager = RefreshMappingManager(strategy=strategy)

        current_times = {"file1.txt": 100.0}
        manager.update_times(current_times)

        # Small change (30 seconds)
        current_times2 = {"file1.txt": 130.0}
        keys_to_refresh = manager.get_keys_to_refresh(current_times2)

        # Should not refresh due to threshold
        assert "file1.txt" not in keys_to_refresh

        # Large change (70 seconds)
        current_times3 = {"file1.txt": 170.0}
        keys_to_refresh2 = manager.get_keys_to_refresh(current_times3)

        # Should refresh
        assert "file1.txt" in keys_to_refresh2
