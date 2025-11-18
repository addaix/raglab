"""Tests for source management."""

import pytest
import tempfile
import shutil
from pathlib import Path
from raglab.rag.sources import FolderSource, MultiSource
from raglab.rag.decoders import ExtensionBasedDecoder


class TestFolderSource:
    """Test FolderSource functionality."""

    @pytest.fixture
    def temp_folder(self):
        """Create a temporary folder with test files."""
        temp_dir = tempfile.mkdtemp()
        folder = Path(temp_dir)

        # Create test files
        (folder / "file1.txt").write_text("Content of file 1")
        (folder / "file2.md").write_text("# Markdown content")
        (folder / "file3.py").write_text("print('hello')")

        # Create subdirectory
        sub_dir = folder / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file4.txt").write_text("Content in subdirectory")

        # Create files to exclude
        (folder / ".git").mkdir()
        (folder / ".git" / "config").write_text("git config")
        (folder / "__pycache__").mkdir()
        (folder / "__pycache__" / "cache.pyc").write_text("bytecode")

        yield folder

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_basic_folder_source(self, temp_folder):
        """Test basic folder source functionality."""
        source = FolderSource(temp_folder)
        content_mapping = source.get_content_mapping()

        # Should have all files except .git and __pycache__
        assert len(content_mapping) >= 4
        assert "file1.txt" in content_mapping
        assert content_mapping["file1.txt"] == "Content of file 1"

    def test_exclude_patterns(self, temp_folder):
        """Test regex exclude patterns."""
        source = FolderSource(
            temp_folder,
            exclude_patterns=[r"\.git", r"__pycache__"]
        )
        content_mapping = source.get_content_mapping()

        # Check that .git and __pycache__ are excluded
        keys = list(content_mapping.keys())
        assert not any(".git" in key for key in keys)
        assert not any("__pycache__" in key for key in keys)

    def test_exclude_paths(self, temp_folder):
        """Test explicit path exclusion."""
        exclude_file = temp_folder / "file1.txt"
        source = FolderSource(
            temp_folder,
            exclude_paths=[exclude_file]
        )
        content_mapping = source.get_content_mapping()

        assert "file1.txt" not in content_mapping
        assert "file2.md" in content_mapping

    def test_exclude_function(self, temp_folder):
        """Test custom exclude function."""
        def exclude_py_files(path: Path) -> bool:
            return path.suffix == '.py'

        source = FolderSource(
            temp_folder,
            exclude_func=exclude_py_files
        )
        content_mapping = source.get_content_mapping()

        assert "file3.py" not in content_mapping
        assert "file1.txt" in content_mapping

    def test_extension_filter(self, temp_folder):
        """Test extension filtering."""
        source = FolderSource(
            temp_folder,
            extensions={'.txt'}
        )
        content_mapping = source.get_content_mapping()

        # Only .txt files should be included
        assert "file1.txt" in content_mapping
        assert "subdir/file4.txt" in content_mapping
        assert "file2.md" not in content_mapping
        assert "file3.py" not in content_mapping

    def test_non_recursive(self, temp_folder):
        """Test non-recursive scanning."""
        source = FolderSource(
            temp_folder,
            recursive=False
        )
        content_mapping = source.get_content_mapping()

        # Should not include files in subdirectory
        assert "file1.txt" in content_mapping
        assert "subdir/file4.txt" not in content_mapping

    def test_update_times(self, temp_folder):
        """Test update time tracking."""
        source = FolderSource(temp_folder)
        update_times = source.get_update_times()

        assert "file1.txt" in update_times
        assert isinstance(update_times["file1.txt"], float)
        assert update_times["file1.txt"] > 0

    def test_refresh(self, temp_folder):
        """Test refresh functionality."""
        source = FolderSource(temp_folder)

        # Get initial content
        content1 = source.get_content_mapping()
        assert len(content1) > 0

        # Add a new file
        (temp_folder / "new_file.txt").write_text("New content")

        # Refresh should pick up the new file
        source.refresh()
        content2 = source.get_content_mapping()
        assert "new_file.txt" in content2


class TestMultiSource:
    """Test MultiSource functionality."""

    @pytest.fixture
    def temp_folders(self):
        """Create multiple temporary folders."""
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()

        folder1 = Path(temp_dir1)
        folder2 = Path(temp_dir2)

        # Create files in folder1
        (folder1 / "file1.txt").write_text("Content 1")
        (folder1 / "file2.txt").write_text("Content 2")

        # Create files in folder2
        (folder2 / "file3.txt").write_text("Content 3")
        (folder2 / "file4.txt").write_text("Content 4")

        yield folder1, folder2

        # Cleanup
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)

    def test_multi_source(self, temp_folders):
        """Test combining multiple sources."""
        folder1, folder2 = temp_folders

        source1 = FolderSource(folder1)
        source2 = FolderSource(folder2)
        multi_source = MultiSource([source1, source2])

        content_mapping = multi_source.get_content_mapping()

        # Should have files from both sources with prefixed keys
        assert len(content_mapping) == 4
        assert any("source_0/file1.txt" in key for key in content_mapping.keys())
        assert any("source_1/file3.txt" in key for key in content_mapping.keys())

    def test_multi_source_update_times(self, temp_folders):
        """Test update times from multiple sources."""
        folder1, folder2 = temp_folders

        source1 = FolderSource(folder1)
        source2 = FolderSource(folder2)
        multi_source = MultiSource([source1, source2])

        update_times = multi_source.get_update_times()
        assert len(update_times) == 4

    def test_add_source(self, temp_folders):
        """Test adding a source to MultiSource."""
        folder1, folder2 = temp_folders

        source1 = FolderSource(folder1)
        multi_source = MultiSource([source1])

        # Initially should have 2 files
        content1 = multi_source.get_content_mapping()
        assert len(content1) == 2

        # Add another source
        source2 = FolderSource(folder2)
        multi_source.add_source(source2)

        # Should now have 4 files
        content2 = multi_source.get_content_mapping()
        assert len(content2) == 4
