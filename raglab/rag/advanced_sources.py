"""Advanced source types: URL, Database, API, Git, etc."""

from pathlib import Path
from typing import List, Dict, Optional, Any
from collections.abc import Mapping
import tempfile
import subprocess

from .types import ContentMapping, UpdateTimeMapping, ContentKey


class URLSource:
    """Web scraping source for URLs."""

    def __init__(
        self,
        urls: List[str],
        crawler_depth: int = 0,
        follow_links: bool = False,
    ):
        """
        Initialize URL source.

        Args:
            urls: List of URLs to scrape
            crawler_depth: Depth to crawl (0 = only given URLs)
            follow_links: Whether to follow links
        """
        self.urls = urls
        self.crawler_depth = crawler_depth
        self.follow_links = follow_links
        self._content_cache = {}
        self._update_times = {}

    def get_content_mapping(self) -> ContentMapping:
        """Fetch content from URLs."""
        content_map = {}

        for url in self.urls:
            try:
                content = self._fetch_url(url)
                content_map[url] = content
                self._content_cache[url] = content
            except Exception as e:
                content_map[url] = f"[Error fetching {url}]: {e}"

        return content_map

    def _fetch_url(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text

        except ImportError:
            return "[Error: requests or beautifulsoup4 not installed]"
        except Exception as e:
            return f"[Error fetching URL]: {e}"

    def get_update_times(self) -> UpdateTimeMapping:
        """Get update times for URLs (use Last-Modified header)."""
        import time
        update_times = {}

        for url in self.urls:
            try:
                import requests
                response = requests.head(url, timeout=10)
                last_modified = response.headers.get('Last-Modified')

                if last_modified:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(last_modified)
                    update_times[url] = dt.timestamp()
                else:
                    update_times[url] = time.time()
            except Exception:
                update_times[url] = time.time()

        return update_times

    def refresh(self) -> None:
        """Refresh cached content."""
        self._content_cache.clear()
        self._update_times.clear()


class DatabaseSource:
    """Database source for SQL queries."""

    def __init__(
        self,
        connection_string: str,
        query: str,
        text_column: str = "text",
        id_column: str = "id",
    ):
        """
        Initialize database source.

        Args:
            connection_string: Database connection string
            query: SQL query to execute
            text_column: Column containing text content
            id_column: Column to use as content key
        """
        self.connection_string = connection_string
        self.query = query
        self.text_column = text_column
        self.id_column = id_column

    def get_content_mapping(self) -> ContentMapping:
        """Fetch content from database."""
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(self.connection_string)

            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(self.query))
                content_map = {}

                for row in result:
                    row_dict = dict(row._mapping)
                    key = str(row_dict.get(self.id_column, ''))
                    text = str(row_dict.get(self.text_column, ''))
                    content_map[key] = text

            return content_map

        except ImportError:
            return {"error": "[Error: sqlalchemy not installed]"}
        except Exception as e:
            return {"error": f"[Error querying database]: {e}"}

    def get_update_times(self) -> UpdateTimeMapping:
        """Get update times (uses current time)."""
        import time
        content_map = self.get_content_mapping()
        return {key: time.time() for key in content_map}

    def refresh(self) -> None:
        """Refresh (no-op for database)."""
        pass


class APISource:
    """API source for REST APIs."""

    def __init__(
        self,
        endpoint: str,
        auth: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        response_path: str = "data",
        id_field: str = "id",
        text_field: str = "text",
    ):
        """
        Initialize API source.

        Args:
            endpoint: API endpoint URL
            auth: Authentication dict (e.g., {"Authorization": "Bearer token"})
            headers: Additional headers
            params: Query parameters
            response_path: Path to data in response (dot notation)
            id_field: Field to use as content key
            text_field: Field containing text content
        """
        self.endpoint = endpoint
        self.auth = auth or {}
        self.headers = headers or {}
        self.params = params or {}
        self.response_path = response_path
        self.id_field = id_field
        self.text_field = text_field

    def get_content_mapping(self) -> ContentMapping:
        """Fetch content from API."""
        try:
            import requests

            headers = {**self.headers, **self.auth}
            response = requests.get(
                self.endpoint,
                headers=headers,
                params=self.params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Navigate to data path
            for key in self.response_path.split('.'):
                if key:
                    data = data[key]

            # Extract content
            content_map = {}
            if isinstance(data, list):
                for item in data:
                    key = str(item.get(self.id_field, ''))
                    text = str(item.get(self.text_field, ''))
                    content_map[key] = text
            elif isinstance(data, dict):
                key = str(data.get(self.id_field, ''))
                text = str(data.get(self.text_field, ''))
                content_map[key] = text

            return content_map

        except ImportError:
            return {"error": "[Error: requests not installed]"}
        except Exception as e:
            return {"error": f"[Error fetching from API]: {e}"}

    def get_update_times(self) -> UpdateTimeMapping:
        """Get update times."""
        import time
        content_map = self.get_content_mapping()
        return {key: time.time() for key in content_map}

    def refresh(self) -> None:
        """Refresh (no-op for API)."""
        pass


class GitRepoSource:
    """Git repository source."""

    def __init__(
        self,
        repo_url: str,
        branch: str = "main",
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize Git repository source.

        Args:
            repo_url: Git repository URL
            branch: Branch to checkout
            file_patterns: File patterns to include (e.g., ["*.md", "*.py"])
            exclude_patterns: Patterns to exclude
        """
        self.repo_url = repo_url
        self.branch = branch
        self.file_patterns = file_patterns or ["*"]
        self.exclude_patterns = exclude_patterns or []
        self.temp_dir = None

    def _clone_repo(self) -> Path:
        """Clone repository to temp directory."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp())

        try:
            # Clone repo
            subprocess.run(
                ['git', 'clone', '--branch', self.branch, '--depth', '1', self.repo_url, str(self.temp_dir)],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repo: {e}")

        return self.temp_dir

    def get_content_mapping(self) -> ContentMapping:
        """Get content from git repository."""
        try:
            repo_path = self._clone_repo()

            # Use FolderSource to read files
            from .sources import FolderSource

            source = FolderSource(
                folder_path=repo_path,
                exclude_patterns=self.exclude_patterns,
                extensions=set(self.file_patterns) if self.file_patterns != ["*"] else None,
            )

            return source.get_content_mapping()

        except Exception as e:
            return {"error": f"[Error reading git repo]: {e}"}

    def get_update_times(self) -> UpdateTimeMapping:
        """Get update times from git."""
        try:
            repo_path = self._clone_repo()

            from .sources import FolderSource
            source = FolderSource(folder_path=repo_path)

            return source.get_update_times()

        except Exception:
            import time
            content_map = self.get_content_mapping()
            return {key: time.time() for key in content_map}

    def refresh(self) -> None:
        """Refresh by pulling latest changes."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        self._clone_repo()

    def __del__(self):
        """Cleanup temp directory."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
