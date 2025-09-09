"""
LangChain Vector Database Manager

Unified tool for discovering, checking status, and installing vector databases
that can be used with LangChain. Combines discovery and installation capabilities.
"""

import importlib
import platform
import socket
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any, Union, Tuple
from enum import Enum
from pathlib import Path

try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False


class OSType(Enum):
    """Supported operating system types."""
    MACOS = "macos"
    LINUX = "linux"
    WINDOWS = "windows"


@dataclass
class InstallStep:
    """Single installation step with description and commands."""
    description: str
    commands: List[str]
    optional: bool = False
    note: Optional[str] = None


@dataclass
class VectorDbConfig:
    """
    Complete configuration for a vector database including discovery and installation.
    
    This class combines functionality for checking status and generating installation
    instructions for vector databases.
    """
    # Basic identification
    name: str  # Must be a valid Python identifier (e.g., 'qdrant', 'mongodb_atlas')
    aliases: Set[str] = field(default_factory=set)  # Alternative names
    
    # LangChain integration
    langchain_module: str = ""
    pip_packages: List[str] = field(default_factory=list)
    system_dependencies: List[str] = field(default_factory=list)
    
    # URLs and documentation
    installation_url: str = ""
    documentation_url: str = ""
    
    # Installation steps by OS
    setup_steps: Dict[OSType, List[InstallStep]] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    post_install_notes: List[str] = field(default_factory=list)
    
    # Service configuration
    default_port: Optional[int] = None
    default_host: str = "localhost"
    cloud_service: bool = False
    embedded_mode: bool = False
    
    # Custom check functions
    service_check: Optional[Callable[[], Union[str, bool]]] = None
    metadata_collector: Optional[Callable[[], Dict[str, Any]]] = None
    
    def check_langchain_available(self) -> bool:
        """Check if the LangChain integration module is available."""
        if not self.langchain_module:
            return False
        try:
            importlib.import_module(self.langchain_module)
            return True
        except ImportError:
            return False
    
    def check_dependencies_installed(self) -> Tuple[bool, List[str], List[str]]:
        """
        Check if Python dependencies are installed.
        
        Returns:
            Tuple of (all_met, installed_packages, missing_packages)
        """
        installed = []
        missing = []
        
        for package in self.pip_packages:
            # Handle package names with extras like package[extra]
            base_package = package.split('[')[0]
            # Convert package name to module name (e.g., faiss-cpu -> faiss)
            module_name = base_package.replace('-', '_')
            
            try:
                importlib.import_module(module_name)
                installed.append(package)
            except ImportError:
                # Try without underscore conversion
                try:
                    importlib.import_module(base_package)
                    installed.append(package)
                except ImportError:
                    missing.append(package)
        
        return (len(missing) == 0, installed, missing)
    
    def check_service_running(self) -> Union[str, bool]:
        """
        Check if the vector database service is running.
        
        Returns:
            URI string if service is accessible (e.g., "http://localhost:6333")
            False if service is not accessible
        """
        # Use custom service check if provided
        if self.service_check:
            return self.service_check()
        
        # For cloud services, check environment variables
        if self.cloud_service:
            if self.name == "mongodb_atlas":
                uri = os.getenv("MONGODB_ATLAS_URI")
                return uri if uri else False
            elif self.name == "pinecone":
                api_key = os.getenv("PINECONE_API_KEY")
                return "pinecone.io" if api_key else False
        
        # For embedded databases, always return True if dependencies are met
        if self.embedded_mode:
            deps_met, _, _ = self.check_dependencies_installed()
            return f"embedded://{self.name}" if deps_met else False
        
        # Default: check if service is running on default port
        if self.default_port:
            if self._check_port(self.default_host, self.default_port):
                return f"http://{self.default_host}:{self.default_port}"
        
        return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Collect metadata about the current installation."""
        metadata = {
            "type": "cloud" if self.cloud_service else "local",
            "embedded_supported": self.embedded_mode,
        }
        
        if self.default_port:
            metadata["default_port"] = self.default_port
        
        # Use custom metadata collector if provided
        if self.metadata_collector:
            try:
                metadata.update(self.metadata_collector())
            except Exception as e:
                metadata["metadata_error"] = str(e)
        
        return metadata
    
    def print_pip_install(self) -> str:
        """Generate pip install instructions."""
        if not self.pip_packages:
            return "# No Python packages required"
        
        deps_met, installed, missing = self.check_dependencies_installed()
        
        lines = []
        if deps_met:
            lines.append(f"# ✅ All Python packages already installed")
            lines.append(f"# Installed: {', '.join(installed)}")
            lines.append(f"# To update, run:")
            lines.append(f"pip install -U {' '.join(self.pip_packages)}")
        else:
            if installed:
                lines.append(f"# ✅ Already installed: {', '.join(installed)}")
            lines.append(f"# ❌ Missing packages: {', '.join(missing)}")
            lines.append(f"# Install with:")
            lines.append(f"pip install {' '.join(missing)}")
        
        return "\n".join(lines)
    
    def print_system_dependencies(self, os_type: OSType) -> str:
        """Generate system dependencies installation instructions."""
        lines = []
        
        if not self.system_dependencies:
            return "# No system dependencies required"
        
        lines.append(f"# System dependencies for {self.name}")
        
        if os_type == OSType.MACOS:
            lines.append("# Install with Homebrew:")
            for dep in self.system_dependencies:
                lines.append(f"brew install {dep}")
        elif os_type == OSType.LINUX:
            lines.append("# Install with apt (Ubuntu/Debian):")
            for dep in self.system_dependencies:
                lines.append(f"sudo apt-get install {dep}")
        elif os_type == OSType.WINDOWS:
            lines.append("# Install system dependencies:")
            for dep in self.system_dependencies:
                lines.append(f"# Download and install {dep}")
        
        return "\n".join(lines)
    
    def print_launch_service(self, os_type: OSType) -> str:
        """Generate service launch instructions."""
        lines = []
        
        # Check if service is already running
        service_uri = self.check_service_running()
        if service_uri and isinstance(service_uri, str):
            lines.append(f"# ✅ Service already running at: {service_uri}")
            lines.append(f"# To restart or reconfigure, see instructions below:")
            lines.append("")
        
        # Get OS-specific setup steps
        if os_type in self.setup_steps:
            for step in self.setup_steps[os_type]:
                if step.optional and service_uri:
                    continue  # Skip optional steps if service is running
                
                lines.append(f"# {step.description}")
                if step.note:
                    lines.append(f"# Note: {step.note}")
                lines.append("#" + "-" * 50)
                for cmd in step.commands:
                    lines.append(cmd)
                lines.append("")
        
        # Environment variables
        if self.environment_vars:
            lines.append("# Environment variables:")
            for key, value in self.environment_vars.items():
                current_value = os.getenv(key)
                if current_value:
                    lines.append(f"# ✅ {key} is already set")
                else:
                    lines.append(f"export {key}=\"{value}\"")
            lines.append("")
        
        return "\n".join(lines)
    
    def print_full_installation(self, os_type: OSType) -> str:
        """Generate complete installation instructions."""
        sections = []
        
        # Header
        sections.append(f"# Complete installation for {self.name}")
        sections.append(f"# OS: {os_type.value}")
        if self.documentation_url:
            sections.append(f"# Documentation: {self.documentation_url}")
        if self.installation_url:
            sections.append(f"# Installation guide: {self.installation_url}")
        sections.append("")
        
        # Python packages
        sections.append("# STEP 1: Python packages")
        sections.append("#" + "=" * 50)
        sections.append(self.print_pip_install())
        sections.append("")
        
        # System dependencies
        if self.system_dependencies:
            sections.append("# STEP 2: System dependencies")
            sections.append("#" + "=" * 50)
            sections.append(self.print_system_dependencies(os_type))
            sections.append("")
        
        # Launch service
        sections.append(f"# STEP {3 if self.system_dependencies else 2}: Launch service")
        sections.append("#" + "=" * 50)
        sections.append(self.print_launch_service(os_type))
        
        # Verification
        sections.append("# VERIFICATION")
        sections.append("#" + "=" * 50)
        sections.append(self._get_verification_commands())
        sections.append("")
        
        # Notes
        if self.post_install_notes:
            sections.append("# NOTES")
            sections.append("#" + "=" * 50)
            for note in self.post_install_notes:
                sections.append(f"# • {note}")
        
        return "\n".join(sections)
    
    def _check_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a port is open on a host."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception:
            return False
    
    def _get_verification_commands(self) -> str:
        """Generate verification commands."""
        lines = []
        
        # Check service
        service_uri = self.check_service_running()
        if service_uri and isinstance(service_uri, str):
            lines.append(f"# Service check: ✅ Running at {service_uri}")
        else:
            lines.append(f"# Service check: ❌ Not running")
        
        # Python import test
        lines.append(f"# Test Python import:")
        if self.name == "qdrant":
            lines.append('python -c "from qdrant_client import QdrantClient; print(\'✅ Import successful\')"')
        elif self.name == "chroma":
            lines.append('python -c "import chromadb; print(\'✅ Import successful\')"')
        elif self.name == "mongodb_atlas":
            lines.append('python -c "from pymongo import MongoClient; print(\'✅ Import successful\')"')
        else:
            for package in self.pip_packages:
                base_package = package.split('[')[0].replace('-', '_')
                lines.append(f'python -c "import {base_package}; print(\'✅ {base_package} imported\')"')
        
        return "\n".join(lines)


class VectorDbRegistry:
    """Registry for vector database configurations."""
    
    def __init__(self):
        self._configs: Dict[str, VectorDbConfig] = {}
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default vector database configurations."""
        # Qdrant
        self.register(self._create_qdrant_config())
        
        # Chroma
        self.register(self._create_chroma_config())
        
        # MongoDB Atlas
        self.register(self._create_mongodb_atlas_config())
        
        # Register other databases with basic info
        self._register_basic_configs()
    
    def _create_qdrant_config(self) -> VectorDbConfig:
        """Create Qdrant configuration."""
        return VectorDbConfig(
            name="qdrant",
            aliases={"Qdrant", "qdrant-local", "Qdrant (Local)"},
            langchain_module="langchain_qdrant",
            pip_packages=["qdrant-client", "langchain-qdrant"],
            system_dependencies=["docker"],
            installation_url="https://qdrant.tech/documentation/quick-start/",
            documentation_url="https://qdrant.tech/documentation/",
            default_port=6333,
            setup_steps={
                OSType.MACOS: [
                    InstallStep(
                        description="Run Qdrant with Docker",
                        commands=[
                            "# Pull and run Qdrant container",
                            "docker pull qdrant/qdrant",
                            "docker run -d --name qdrant \\",
                            "  -p 6333:6333 -p 6334:6334 \\",
                            "  -v $(pwd)/qdrant_storage:/qdrant/storage:z \\",
                            "  qdrant/qdrant"
                        ],
                        note="Use 'docker stop qdrant' and 'docker start qdrant' to manage"
                    ),
                    InstallStep(
                        description="Alternative: Install binary directly",
                        commands=[
                            "# Install Qdrant binary",
                            "curl --proto '=https' --tlsv1.2 -sSf https://get.qdrant.io | sh",
                            "./qdrant"
                        ],
                        optional=True
                    )
                ],
                OSType.LINUX: [
                    InstallStep(
                        description="Run Qdrant with Docker",
                        commands=[
                            "# Pull and run Qdrant container",
                            "docker pull qdrant/qdrant",
                            "docker run -d --name qdrant \\",
                            "  -p 6333:6333 -p 6334:6334 \\",
                            "  -v $(pwd)/qdrant_storage:/qdrant/storage:z \\",
                            "  qdrant/qdrant"
                        ]
                    )
                ],
                OSType.WINDOWS: [
                    InstallStep(
                        description="Run Qdrant with Docker Desktop",
                        commands=[
                            "REM Pull and run Qdrant container",
                            "docker pull qdrant/qdrant",
                            "docker run -d --name qdrant ^",
                            "  -p 6333:6333 -p 6334:6334 ^",
                            "  -v %cd%/qdrant_storage:/qdrant/storage:z ^",
                            "  qdrant/qdrant"
                        ]
                    )
                ]
            },
            post_install_notes=[
                "Qdrant dashboard: http://localhost:6333/dashboard",
                "API endpoint: http://localhost:6333",
                "For cloud version: https://cloud.qdrant.io"
            ]
        )
    
    def _create_chroma_config(self) -> VectorDbConfig:
        """Create Chroma configuration."""
        def check_chroma_service():
            # Chroma can work embedded or as server
            if VectorDbConfig._check_port(None, "localhost", 8000):
                return "http://localhost:8000"
            # Check if chromadb is installed for embedded mode
            try:
                import chromadb
                return "embedded://chroma"
            except ImportError:
                return False
        
        return VectorDbConfig(
            name="chroma",
            aliases={"Chroma", "chromadb"},
            langchain_module="langchain_chroma",
            pip_packages=["chromadb", "langchain-chroma"],
            installation_url="https://docs.trychroma.com/getting-started",
            documentation_url="https://docs.trychroma.com/",
            default_port=8000,
            embedded_mode=True,
            service_check=check_chroma_service,
            setup_steps={
                OSType.MACOS: [
                    InstallStep(
                        description="Server mode (optional)",
                        commands=[
                            "# Chroma works embedded after pip install",
                            "# For server mode, choose one:",
                            "",
                            "# Option 1: Docker",
                            "docker pull chromadb/chroma",
                            "docker run -d --name chroma -p 8000:8000 chromadb/chroma",
                            "",
                            "# Option 2: Python server",
                            "# chroma run --host 0.0.0.0 --port 8000"
                        ],
                        optional=True,
                        note="Server mode optional. Embedded mode works after pip install."
                    )
                ],
                OSType.LINUX: [
                    InstallStep(
                        description="Server mode (optional)",
                        commands=[
                            "# Chroma works embedded after pip install",
                            "# For server mode:",
                            "docker pull chromadb/chroma",
                            "docker run -d --name chroma -p 8000:8000 chromadb/chroma"
                        ],
                        optional=True
                    )
                ],
                OSType.WINDOWS: [
                    InstallStep(
                        description="Server mode (optional)",
                        commands=[
                            "REM Chroma works embedded after pip install",
                            "REM For server mode:",
                            "docker pull chromadb/chroma",
                            "docker run -d --name chroma -p 8000:8000 chromadb/chroma"
                        ],
                        optional=True
                    )
                ]
            },
            post_install_notes=[
                "Embedded mode: Works immediately after pip install",
                "Server mode: Optional for distributed applications"
            ]
        )
    
    def _create_mongodb_atlas_config(self) -> VectorDbConfig:
        """Create MongoDB Atlas configuration."""
        def check_mongodb_service():
            uri = os.getenv("MONGODB_ATLAS_URI")
            if uri:
                return uri
            # Check local MongoDB
            if VectorDbConfig._check_port(None, "localhost", 27017):
                return "mongodb://localhost:27017"
            return False
        
        return VectorDbConfig(
            name="mongodb_atlas",
            aliases={"MongoDB Atlas", "mongodb", "mongo"},
            langchain_module="langchain_mongodb",
            pip_packages=["pymongo", "langchain-mongodb"],
            installation_url="https://www.mongodb.com/docs/atlas/getting-started/",
            documentation_url="https://www.mongodb.com/docs/atlas/",
            cloud_service=True,
            service_check=check_mongodb_service,
            environment_vars={
                "MONGODB_ATLAS_URI": "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/"
            },
            setup_steps={
                OSType.MACOS: [
                    InstallStep(
                        description="Set up MongoDB Atlas account",
                        commands=[
                            "# 1. Create account at https://cloud.mongodb.com",
                            "# 2. Create a free M0 cluster",
                            "# 3. Add database user (Database Access)",
                            "# 4. Whitelist IP (Network Access)",
                            "# 5. Get connection string (Connect button)",
                            "# 6. Set environment variable with your connection string:",
                            "export MONGODB_ATLAS_URI='your-connection-string'"
                        ]
                    ),
                    InstallStep(
                        description="Local MongoDB (optional)",
                        commands=[
                            "# For local development:",
                            "brew tap mongodb/brew",
                            "brew install mongodb-community",
                            "brew services start mongodb-community"
                        ],
                        optional=True,
                        note="Local MongoDB won't have Atlas Search features"
                    )
                ],
                OSType.LINUX: [
                    InstallStep(
                        description="Set up MongoDB Atlas account",
                        commands=[
                            "# 1. Create account at https://cloud.mongodb.com",
                            "# 2. Create a free M0 cluster",
                            "# 3. Configure access and get connection string",
                            "export MONGODB_ATLAS_URI='your-connection-string'"
                        ]
                    )
                ],
                OSType.WINDOWS: [
                    InstallStep(
                        description="Set up MongoDB Atlas account",
                        commands=[
                            "REM 1. Create account at https://cloud.mongodb.com",
                            "REM 2. Create cluster and configure access",
                            "set MONGODB_ATLAS_URI=your-connection-string"
                        ]
                    )
                ]
            },
            post_install_notes=[
                "MongoDB Atlas is a cloud service",
                "Free M0 tier available for testing",
                "For vector search, create an Atlas Search index"
            ]
        )
    
    def _register_basic_configs(self) -> None:
        """Register other vector databases with basic configuration."""
        basic_configs = [
            VectorDbConfig(
                name="faiss",
                aliases={"FAISS", "faiss-cpu", "faiss-gpu"},
                langchain_module="langchain_community.vectorstores.faiss",
                pip_packages=["faiss-cpu"],  # or faiss-gpu
                embedded_mode=True,
                documentation_url="https://github.com/facebookresearch/faiss"
            ),
            VectorDbConfig(
                name="pinecone",
                aliases={"Pinecone"},
                langchain_module="langchain_pinecone",
                pip_packages=["pinecone-client", "langchain-pinecone"],
                cloud_service=True,
                documentation_url="https://docs.pinecone.io/",
                environment_vars={"PINECONE_API_KEY": "your-api-key"}
            ),
            VectorDbConfig(
                name="weaviate",
                aliases={"Weaviate"},
                langchain_module="langchain_community.vectorstores.weaviate",
                pip_packages=["weaviate-client"],
                default_port=8080,
                documentation_url="https://weaviate.io/developers/weaviate"
            ),
            VectorDbConfig(
                name="milvus",
                aliases={"Milvus"},
                langchain_module="langchain_community.vectorstores.milvus",
                pip_packages=["pymilvus"],
                default_port=19530,
                documentation_url="https://milvus.io/docs"
            ),
            VectorDbConfig(
                name="redis",
                aliases={"Redis", "RediSearch"},
                langchain_module="langchain_community.vectorstores.redis",
                pip_packages=["redis"],
                default_port=6379,
                documentation_url="https://redis.io/docs/stack/search/reference/vectors/"
            ),
            VectorDbConfig(
                name="elasticsearch",
                aliases={"Elasticsearch", "elastic"},
                langchain_module="langchain_community.vectorstores.elasticsearch",
                pip_packages=["elasticsearch"],
                default_port=9200,
                documentation_url="https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html"
            ),
            VectorDbConfig(
                name="pgvector",
                aliases={"PostgreSQL", "postgres", "pg"},
                langchain_module="langchain_community.vectorstores.pgvector",
                pip_packages=["psycopg2-binary", "pgvector"],
                default_port=5432,
                documentation_url="https://github.com/pgvector/pgvector"
            ),
        ]
        
        for config in basic_configs:
            self.register(config)
    
    def register(self, config: VectorDbConfig, *, replace: bool = True) -> None:
        """
        Register a vector database configuration.
        
        Args:
            config: VectorDbConfig instance to register
            replace: Whether to replace existing configuration
        """
        if not replace and config.name in self._configs:
            raise ValueError(f"Configuration for '{config.name}' already exists")
        
        self._configs[config.name] = config
    
    def get(self, name: str) -> Optional[VectorDbConfig]:
        """
        Get configuration by name or alias.
        
        Args:
            name: Name or alias of the vector database
            
        Returns:
            VectorDbConfig if found, None otherwise
        """
        # Try direct name match
        if name in self._configs:
            return self._configs[name]
        
        # Try lowercase
        lower_name = name.lower()
        if lower_name in self._configs:
            return self._configs[lower_name]
        
        # Try aliases
        for config in self._configs.values():
            if name in config.aliases or lower_name in {a.lower() for a in config.aliases}:
                return config
        
        return None
    
    def list_all(self) -> List[str]:
        """List all registered vector database names."""
        return list(self._configs.keys())
    
    def discover_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover status of all registered vector databases.
        
        Returns:
            Dictionary mapping names to status information
        """
        results = {}
        
        for name, config in self._configs.items():
            status = {
                "langchain_available": config.check_langchain_available(),
                "dependencies": {},
                "service": None,
                "metadata": config.get_metadata()
            }
            
            # Check dependencies
            deps_met, installed, missing = config.check_dependencies_installed()
            status["dependencies"] = {
                "all_met": deps_met,
                "installed": installed,
                "missing": missing
            }
            
            # Check service
            service_result = config.check_service_running()
            if isinstance(service_result, str):
                status["service"] = {"running": True, "uri": service_result}
            else:
                status["service"] = {"running": False}
            
            # Overall readiness
            status["is_ready"] = (
                status["langchain_available"] and
                status["dependencies"]["all_met"] and
                (status["service"]["running"] or config.embedded_mode)
            )
            
            results[name] = status
        
        return results


def _detect_os() -> OSType:
    """Detect the current operating system."""
    system = platform.system().lower()
    if system == "darwin":
        return OSType.MACOS
    elif system == "linux":
        return OSType.LINUX
    elif system == "windows":
        return OSType.WINDOWS
    else:
        return OSType.LINUX


def help_me_install(
    vectordb: Optional[str] = None,
    *,
    what: str = "all",  # "all", "pip", "system", "launch"
    copy_to_clipboard: bool = True,
    print_instructions: bool = True
) -> Optional[str]:
    """
    Generate installation instructions for a vector database.
    
    Args:
        vectordb: Name or alias of the vector database
        what: What to install - "all", "pip", "system", or "launch"
        copy_to_clipboard: Copy instructions to clipboard if available
        print_instructions: Print instructions to console
        
    Returns:
        Installation instructions as string, or None if not found
        
    Example:
        >>> # Show available databases
        >>> help_me_install()  # doctest: +SKIP
        
        >>> # Install everything for Qdrant
        >>> help_me_install("qdrant")  # doctest: +SKIP
        
        >>> # Just show pip install for Chroma
        >>> help_me_install("chroma", what="pip")  # doctest: +SKIP
    """
    registry = VectorDbRegistry()
    
    # Show available if no database specified
    if not vectordb:
        available = registry.list_all()
        print("Available vector databases:")
        for name in available:
            config = registry.get(name)
            aliases = f" (aliases: {', '.join(config.aliases)})" if config.aliases else ""
            print(f"  • {name}{aliases}")
        print(f"\nUsage: help_me_install('{available[0]}')")
        return None
    
    # Get configuration
    config = registry.get(vectordb)
    if not config:
        print(f"❌ Unknown vector database: '{vectordb}'")
        print(f"Available: {', '.join(registry.list_all())}")
        return None
    
    # Detect OS
    os_type = _detect_os()
    
    # Generate instructions based on what's requested
    if what == "all":
        instructions = config.print_full_installation(os_type)
    elif what == "pip":
        instructions = config.print_pip_install()
    elif what == "system":
        instructions = config.print_system_dependencies(os_type)
    elif what == "launch":
        instructions = config.print_launch_service(os_type)
    else:
        print(f"❌ Invalid 'what' parameter: {what}")
        print("Valid options: 'all', 'pip', 'system', 'launch'")
        return None
    
    # Print if requested
    if print_instructions:
        print(f"📦 Installation instructions for {config.name}")
        print("=" * 60)
        print(instructions)
        print("=" * 60)
    
    # Copy to clipboard if requested
    if copy_to_clipboard and HAS_CLIPBOARD:
        try:
            pyperclip.copy(instructions)
            print("\n✅ Instructions copied to clipboard!")
        except Exception as e:
            print(f"\n⚠️  Could not copy to clipboard: {e}")
    
    return instructions


def discover_vectordbs(*, detailed: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Discover status of all vector databases.
    
    Args:
        detailed: Include detailed status information
        
    Returns:
        Status dictionary for all databases
        
    Example:
        >>> status = discover_vectordbs()
        >>> ready = [name for name, info in status.items() if info["is_ready"]]
    """
    registry = VectorDbRegistry()
    return registry.discover_status()


def print_discovery_report() -> None:
    """
    Print a formatted discovery report of all vector databases.
    
    Example:
        >>> print_discovery_report()  # doctest: +SKIP
    """
    status = discover_vectordbs(detailed=True)
    
    ready = []
    needs_deps = []
    needs_service = []
    not_available = []
    
    for name, info in status.items():
        if info["is_ready"]:
            ready.append(name)
        elif info["langchain_available"] and info["dependencies"]["all_met"]:
            needs_service.append(name)
        elif info["langchain_available"]:
            needs_deps.append(name)
        else:
            not_available.append(name)
    
    print("🔍 Vector Database Discovery Report")
    print("=" * 50)
    
    if ready:
        print(f"\n✅ Ready to Use ({len(ready)}):")
        for name in ready:
            info = status[name]
            service_info = ""
            if info["service"]["running"]:
                service_info = f" at {info['service']['uri']}"
            print(f"  • {name}{service_info}")
    
    if needs_deps:
        print(f"\n📦 Missing Dependencies ({len(needs_deps)}):")
        for name in needs_deps:
            info = status[name]
            missing = info["dependencies"]["missing"]
            print(f"  • {name}: pip install {' '.join(missing)}")
    
    if needs_service:
        print(f"\n🚀 Service Not Running ({len(needs_service)}):")
        for name in needs_service:
            info = status[name]
            port = info["metadata"].get("default_port", "")
            port_info = f" (port {port})" if port else ""
            print(f"  • {name}{port_info}")
    
    if not_available:
        print(f"\n❌ Not Available ({len(not_available)}):")
        for name in not_available:
            print(f"  • {name}: LangChain integration not installed")
    
    print(f"\nSummary: {len(ready)} ready, {len(needs_deps)} need deps, "
          f"{len(needs_service)} need service, {len(not_available)} not available")


# Convenience functions for specific databases
def install_qdrant(**kwargs) -> Optional[str]:
    """
    Generate installation instructions for Qdrant.
    
    Example:
        >>> instructions = install_qdrant(what="pip", print_instructions=False)  # doctest: +SKIP
    """
    return help_me_install("qdrant", **kwargs)


def install_chroma(**kwargs) -> Optional[str]:
    """
    Generate installation instructions for Chroma.
    
    Example:
        >>> instructions = install_chroma(what="launch", print_instructions=False)  # doctest: +SKIP
    """
    return help_me_install("chroma", **kwargs)


def install_mongodb_atlas(**kwargs) -> Optional[str]:
    """
    Generate installation instructions for MongoDB Atlas.
    
    Example:
        >>> instructions = install_mongodb_atlas(print_instructions=False)  # doctest: +SKIP
    """
    return help_me_install("mongodb_atlas", **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        db_name = sys.argv[1]
        what = sys.argv[2] if len(sys.argv) > 2 else "all"
        help_me_install(db_name, what=what)
    else:
        # Show discovery report
        print_discovery_report()
        print("\n" + "=" * 50)
        print("Usage: python vectordb_installer.py <database> [all|pip|system|launch]")
        print("Example: python vectordb_installer.py qdrant pip")