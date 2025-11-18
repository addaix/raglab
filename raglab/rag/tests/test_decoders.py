"""Tests for content decoders."""

import pytest
import tempfile
import shutil
from pathlib import Path
from raglab.rag.decoders import (
    ExtensionBasedDecoder,
    text_decoder,
    markdown_decoder,
    python_decoder,
    json_decoder,
)


class TestDecoders:
    """Test content decoders."""

    @pytest.fixture
    def temp_folder(self):
        """Create temporary test files."""
        temp_dir = tempfile.mkdtemp()
        folder = Path(temp_dir)

        # Create test files
        (folder / "test.txt").write_text("Plain text content")
        (folder / "test.md").write_text("# Markdown\n\nSome markdown content")
        (folder / "test.py").write_text("def hello():\n    print('hello')")
        (folder / "test.json").write_text('{"key": "value"}')

        yield folder

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_text_decoder(self, temp_folder):
        """Test plain text decoder."""
        file_path = temp_folder / "test.txt"
        content = text_decoder(file_path)

        assert content == "Plain text content"

    def test_markdown_decoder(self, temp_folder):
        """Test markdown decoder."""
        file_path = temp_folder / "test.md"
        content = markdown_decoder(file_path)

        assert "# Markdown" in content
        assert "markdown content" in content

    def test_python_decoder(self, temp_folder):
        """Test Python decoder."""
        file_path = temp_folder / "test.py"
        content = python_decoder(file_path)

        assert "def hello()" in content

    def test_json_decoder(self, temp_folder):
        """Test JSON decoder."""
        file_path = temp_folder / "test.json"
        content = json_decoder(file_path)

        assert '"key"' in content
        assert '"value"' in content

    def test_extension_based_decoder(self, temp_folder):
        """Test ExtensionBasedDecoder."""
        decoder = ExtensionBasedDecoder()

        # Test .txt
        content_txt = decoder(temp_folder / "test.txt")
        assert content_txt == "Plain text content"

        # Test .md
        content_md = decoder(temp_folder / "test.md")
        assert "# Markdown" in content_md

        # Test .py
        content_py = decoder(temp_folder / "test.py")
        assert "def hello()" in content_py

        # Test .json
        content_json = decoder(temp_folder / "test.json")
        assert '"key"' in content_json

    def test_register_custom_decoder(self, temp_folder):
        """Test registering a custom decoder."""
        decoder = ExtensionBasedDecoder()

        # Create custom decoder
        def custom_decoder(file_path: Path) -> str:
            return "CUSTOM: " + file_path.read_text()

        # Register for .txt
        decoder.register_decoder('.txt', custom_decoder)

        # Test custom decoder
        content = decoder(temp_folder / "test.txt")
        assert content.startswith("CUSTOM:")
        assert "Plain text content" in content

    def test_unregister_decoder(self, temp_folder):
        """Test unregistering a decoder."""
        decoder = ExtensionBasedDecoder()

        # Unregister .txt decoder
        decoder.unregister_decoder('.txt')

        # Should use default decoder
        content = decoder(temp_folder / "test.txt")
        assert content == "Plain text content"  # Default still works

    def test_unknown_extension(self, temp_folder):
        """Test decoder with unknown extension."""
        decoder = ExtensionBasedDecoder()

        # Create file with unknown extension
        unknown_file = temp_folder / "test.unknown"
        unknown_file.write_text("Unknown content")

        # Should use default decoder
        content = decoder(unknown_file)
        assert content == "Unknown content"

    def test_custom_default_decoder(self, temp_folder):
        """Test custom default decoder."""
        def custom_default(file_path: Path) -> str:
            return f"[File: {file_path.name}]"

        decoder = ExtensionBasedDecoder(default_decoder=custom_default)

        # Create file with unknown extension
        unknown_file = temp_folder / "test.unknown"
        unknown_file.write_text("Unknown content")

        # Should use custom default
        content = decoder(unknown_file)
        assert content == "[File: test.unknown]"

    def test_decoder_error_handling(self, temp_folder):
        """Test decoder error handling."""
        decoder = ExtensionBasedDecoder()

        # Try to decode non-existent file
        non_existent = temp_folder / "nonexistent.txt"
        content = decoder(non_existent)

        # Should return error message
        assert "Decode error" in content
        assert "nonexistent.txt" in content
