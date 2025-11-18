"""Content decoders for different file types."""

from pathlib import Path
from typing import Dict, Callable
import mimetypes

# Default decoders for common file types


def text_decoder(file_path: Path) -> str:
    """Decode plain text files."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def binary_decoder(file_path: Path) -> str:
    """Decode binary files (returns empty string)."""
    return ""


def markdown_decoder(file_path: Path) -> str:
    """Decode markdown files."""
    return text_decoder(file_path)


def python_decoder(file_path: Path) -> str:
    """Decode Python source files."""
    return text_decoder(file_path)


def json_decoder(file_path: Path) -> str:
    """Decode JSON files."""
    return text_decoder(file_path)


def yaml_decoder(file_path: Path) -> str:
    """Decode YAML files."""
    return text_decoder(file_path)


def pdf_decoder(file_path: Path) -> str:
    """Decode PDF files (requires pypdf)."""
    try:
        import pypdf
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except ImportError:
        return f"[PDF file - pypdf not installed]: {file_path}"
    except Exception as e:
        return f"[PDF decode error]: {e}"


def docx_decoder(file_path: Path) -> str:
    """Decode DOCX files (requires python-docx)."""
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        return f"[DOCX file - python-docx not installed]: {file_path}"
    except Exception as e:
        return f"[DOCX decode error]: {e}"


# Extension-based decoder registry
DEFAULT_DECODERS: Dict[str, Callable[[Path], str]] = {
    '.txt': text_decoder,
    '.md': markdown_decoder,
    '.markdown': markdown_decoder,
    '.py': python_decoder,
    '.js': text_decoder,
    '.ts': text_decoder,
    '.jsx': text_decoder,
    '.tsx': text_decoder,
    '.java': text_decoder,
    '.c': text_decoder,
    '.cpp': text_decoder,
    '.h': text_decoder,
    '.hpp': text_decoder,
    '.cs': text_decoder,
    '.go': text_decoder,
    '.rs': text_decoder,
    '.rb': text_decoder,
    '.php': text_decoder,
    '.html': text_decoder,
    '.htm': text_decoder,
    '.xml': text_decoder,
    '.css': text_decoder,
    '.scss': text_decoder,
    '.json': json_decoder,
    '.yaml': yaml_decoder,
    '.yml': yaml_decoder,
    '.toml': text_decoder,
    '.ini': text_decoder,
    '.cfg': text_decoder,
    '.conf': text_decoder,
    '.sh': text_decoder,
    '.bash': text_decoder,
    '.zsh': text_decoder,
    '.fish': text_decoder,
    '.sql': text_decoder,
    '.r': text_decoder,
    '.R': text_decoder,
    '.tex': text_decoder,
    '.pdf': pdf_decoder,
    '.docx': docx_decoder,
    # Add more as needed
}


class ExtensionBasedDecoder:
    """Decoder that selects the appropriate decoder based on file extension."""

    def __init__(
        self,
        decoders: Dict[str, Callable[[Path], str]] | None = None,
        default_decoder: Callable[[Path], str] | None = None
    ):
        """
        Initialize the extension-based decoder.

        Args:
            decoders: Dictionary mapping file extensions to decoder functions
            default_decoder: Fallback decoder for unknown extensions
        """
        self.decoders = decoders if decoders is not None else DEFAULT_DECODERS.copy()
        self.default_decoder = default_decoder or text_decoder

    def __call__(self, file_path: Path) -> str:
        """Decode a file based on its extension."""
        extension = file_path.suffix.lower()
        decoder = self.decoders.get(extension, self.default_decoder)
        try:
            return decoder(file_path)
        except Exception as e:
            return f"[Decode error for {file_path}]: {e}"

    def register_decoder(self, extension: str, decoder: Callable[[Path], str]) -> None:
        """Register a new decoder for a specific extension."""
        if not extension.startswith('.'):
            extension = '.' + extension
        self.decoders[extension.lower()] = decoder

    def unregister_decoder(self, extension: str) -> None:
        """Remove a decoder for a specific extension."""
        if not extension.startswith('.'):
            extension = '.' + extension
        self.decoders.pop(extension.lower(), None)
