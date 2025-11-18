"""Smart chunking strategies for different content types."""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import ast


class BaseChunker:
    """Base class for chunking strategies."""

    def chunk(self, text: str) -> List[str]:
        """
        Chunk text into segments.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        raise NotImplementedError

    def chunk_with_metadata(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Chunk text and return with metadata.

        Args:
            text: Text to chunk

        Returns:
            List of (chunk, metadata) tuples
        """
        chunks = self.chunk(text)
        return [(chunk, {}) for chunk in chunks]


class MarkdownAwareChunker(BaseChunker):
    """
    Markdown-aware chunking that respects document structure.

    Splits on headers while maintaining context.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        respect_headers: bool = True,
        include_header_context: bool = True,
    ):
        """
        Initialize markdown chunker.

        Args:
            max_chunk_size: Maximum size of chunks
            respect_headers: Split on headers
            include_header_context: Include parent headers in chunks
        """
        self.max_chunk_size = max_chunk_size
        self.respect_headers = respect_headers
        self.include_header_context = include_header_context

        # Regex for markdown headers
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def _parse_headers(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown headers from text."""
        headers = []
        for match in self.header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2)
            position = match.start()
            headers.append({
                'level': level,
                'title': title,
                'position': position,
            })
        return headers

    def chunk_with_metadata(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk markdown with header metadata."""
        headers = self._parse_headers(text)

        if not headers or not self.respect_headers:
            # Fallback to simple chunking
            return self._simple_chunk(text)

        chunks_with_meta = []
        header_stack = []  # Stack of parent headers

        # Add text before first header
        if headers[0]['position'] > 0:
            pre_text = text[:headers[0]['position']].strip()
            if pre_text:
                chunks_with_meta.append((pre_text, {'section': 'Introduction'}))

        # Process each section
        for i, header in enumerate(headers):
            # Update header stack
            while header_stack and header_stack[-1]['level'] >= header['level']:
                header_stack.pop()
            header_stack.append(header)

            # Get section content
            start = header['position']
            end = headers[i + 1]['position'] if i + 1 < len(headers) else len(text)
            section_text = text[start:end].strip()

            # Build context from header stack
            header_context = ' > '.join([h['title'] for h in header_stack])

            # Split large sections
            if len(section_text) > self.max_chunk_size:
                # Split into smaller chunks
                sub_chunks = self._split_text(section_text, self.max_chunk_size)
                for j, chunk in enumerate(sub_chunks):
                    metadata = {
                        'header': header['title'],
                        'header_level': header['level'],
                        'header_context': header_context,
                        'section_part': j + 1,
                        'section_total': len(sub_chunks),
                    }
                    chunks_with_meta.append((chunk, metadata))
            else:
                metadata = {
                    'header': header['title'],
                    'header_level': header['level'],
                    'header_context': header_context,
                }
                chunks_with_meta.append((section_text, metadata))

        return chunks_with_meta

    def _split_text(self, text: str, max_size: int) -> List[str]:
        """Split text into chunks of max_size."""
        # Split on paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > max_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _simple_chunk(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Fallback simple chunking."""
        chunks = self._split_text(text, self.max_chunk_size)
        return [(chunk, {}) for chunk in chunks]

    def chunk(self, text: str) -> List[str]:
        """Chunk markdown text."""
        chunks_with_meta = self.chunk_with_metadata(text)
        return [chunk for chunk, _ in chunks_with_meta]


class CodeAwareChunker(BaseChunker):
    """
    Code-aware chunking that respects code structure.

    Uses AST parsing for Python, regex for other languages.
    """

    def __init__(
        self,
        language: str = "python",
        max_chunk_size: int = 1000,
        include_imports: bool = True,
    ):
        """
        Initialize code chunker.

        Args:
            language: Programming language
            max_chunk_size: Maximum chunk size
            include_imports: Include imports in each chunk
        """
        self.language = language.lower()
        self.max_chunk_size = max_chunk_size
        self.include_imports = include_imports

    def chunk_with_metadata(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk code with metadata."""
        if self.language == "python":
            return self._chunk_python(text)
        elif self.language in ["javascript", "typescript", "java", "c", "cpp"]:
            return self._chunk_generic(text)
        else:
            return self._simple_chunk(text)

    def _chunk_python(self, code: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk Python code using AST."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._simple_chunk(code)

        chunks_with_meta = []
        imports = []

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                import_text = '\n'.join(code.split('\n')[start:end])
                imports.append(import_text)

        imports_text = '\n'.join(imports) if imports else ""

        # Extract top-level constructs
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                chunk_text, metadata = self._extract_function(node, code)
                if self.include_imports and imports_text:
                    chunk_text = imports_text + '\n\n' + chunk_text
                chunks_with_meta.append((chunk_text, metadata))

            elif isinstance(node, ast.ClassDef):
                chunk_text, metadata = self._extract_class(node, code)
                if self.include_imports and imports_text:
                    chunk_text = imports_text + '\n\n' + chunk_text
                chunks_with_meta.append((chunk_text, metadata))

        return chunks_with_meta

    def _extract_function(self, node: ast.FunctionDef, code: str) -> Tuple[str, Dict[str, Any]]:
        """Extract function code and metadata."""
        lines = code.split('\n')
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)

        func_code = '\n'.join(lines[start:end])

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        metadata = {
            'type': 'function',
            'name': node.name,
            'docstring': docstring,
            'line_start': start + 1,
            'line_end': end,
        }

        return func_code, metadata

    def _extract_class(self, node: ast.ClassDef, code: str) -> Tuple[str, Dict[str, Any]]:
        """Extract class code and metadata."""
        lines = code.split('\n')
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)

        class_code = '\n'.join(lines[start:end])

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract methods
        methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]

        metadata = {
            'type': 'class',
            'name': node.name,
            'docstring': docstring,
            'methods': methods,
            'line_start': start + 1,
            'line_end': end,
        }

        return class_code, metadata

    def _chunk_generic(self, code: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk code using regex patterns."""
        # Simple pattern matching for functions
        patterns = {
            'javascript': r'(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)',
            'java': r'(?:public|private|protected)\s+(?:static\s+)?[\w<>,\s]+\s+\w+\s*\([^)]*\)',
            'c': r'[\w\s\*]+\s+\w+\s*\([^)]*\)\s*\{',
        }

        pattern = patterns.get(self.language, r'\w+\s*\([^)]*\)\s*\{')
        matches = list(re.finditer(pattern, code))

        if not matches:
            return self._simple_chunk(code)

        chunks_with_meta = []
        for i, match in enumerate(matches):
            start = match.start()
            # Find end by matching braces
            end = self._find_block_end(code, match.end())

            chunk_text = code[start:end]
            metadata = {
                'type': 'function',
                'line_start': code[:start].count('\n') + 1,
            }
            chunks_with_meta.append((chunk_text, metadata))

        return chunks_with_meta

    def _find_block_end(self, code: str, start: int) -> int:
        """Find the end of a code block by matching braces."""
        brace_count = 0
        in_string = False
        string_char = None

        for i in range(start, len(code)):
            char = code[i]

            # Handle strings
            if char in ['"', "'"] and (i == 0 or code[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return i + 1

        return len(code)

    def _simple_chunk(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Fallback simple chunking."""
        # Split on double newlines
        chunks = []
        current = []
        current_size = 0

        for line in text.split('\n'):
            line_size = len(line)
            if current_size + line_size > self.max_chunk_size and current:
                chunks.append('\n'.join(current))
                current = [line]
                current_size = line_size
            else:
                current.append(line)
                current_size += line_size

        if current:
            chunks.append('\n'.join(current))

        return [(chunk, {}) for chunk in chunks]

    def chunk(self, text: str) -> List[str]:
        """Chunk code."""
        chunks_with_meta = self.chunk_with_metadata(text)
        return [chunk for chunk, _ in chunks_with_meta]


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using embeddings and similarity.

    Integrates with raglab's semantic segmentation.
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        max_chunk_size: int = 1000,
    ):
        """
        Initialize semantic chunker.

        Args:
            embedder: Embedder for semantic similarity
            max_chunk_size: Maximum chunk size
        """
        self.embedder = embedder
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        """Chunk using semantic similarity."""
        try:
            # Try to use raglab's semantic segmentation
            from raglab.retrieval.segmentation_lib import (
                sentence_splits_ids,
                filtered_sentence_split_ids,
                sentence_splits,
                sentence_embeddings,
                consecutive_cosines,
                segment_keys,
                text_segments,
            )

            # Run segmentation pipeline
            split_ids = sentence_splits_ids(text)
            filtered_ids = filtered_sentence_split_ids(split_ids, text)
            sentences = sentence_splits(text, filtered_ids)

            if not sentences:
                return [text]

            embeddings = sentence_embeddings(sentences)
            cosines = consecutive_cosines(embeddings)

            from raglab.retrieval.segmentation_lib import sentence_num_tokens
            num_tokens = sentence_num_tokens(sentences)

            seg_keys = segment_keys(
                num_tokens,
                cosines,
                filtered_ids,
                max_tokens=self.max_chunk_size,
            )

            chunks = text_segments(text, seg_keys)
            return chunks

        except ImportError:
            # Fallback to simple chunking
            return self._simple_chunk(text)

    def _simple_chunk(self, text: str) -> List[str]:
        """Simple fallback chunking."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0

        for sent in sentences:
            sent_size = len(sent)
            if current_size + sent_size > self.max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sent]
                current_size = sent_size
            else:
                current_chunk.append(sent)
                current_size += sent_size

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks


def create_chunker(
    content_type: str,
    **kwargs
) -> BaseChunker:
    """
    Factory function to create appropriate chunker.

    Args:
        content_type: Type of content ("markdown", "code", "semantic", "simple")
        **kwargs: Chunker-specific arguments

    Returns:
        Chunker instance
    """
    if content_type.lower() == "markdown":
        return MarkdownAwareChunker(**kwargs)
    elif content_type.lower() == "code":
        return CodeAwareChunker(**kwargs)
    elif content_type.lower() == "semantic":
        return SemanticChunker(**kwargs)
    else:
        # Simple chunker
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        max_size = kwargs.get('max_chunk_size', 1000)
        overlap = kwargs.get('chunk_overlap', 100)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size,
            chunk_overlap=overlap
        )

        class SimpleChunker(BaseChunker):
            def chunk(self, text: str) -> List[str]:
                return splitter.split_text(text)

        return SimpleChunker()
