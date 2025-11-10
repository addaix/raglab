"""
LangChain Vector Database Discovery Tool

Discovers available vector databases that can be used with LangChain on the current system.
Checks for package availability, dependencies, and service connectivity.
"""

import importlib
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections.abc import Callable, Iterable
from pathlib import Path
import os


@dataclass
class VectorDbStatus:
    """Status information for a vector database integration."""

    name: str
    langchain_available: bool = False
    dependencies_met: bool = False
    service_accessible: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Check if the vector database is ready to use."""
        return (
            self.langchain_available
            and self.dependencies_met
            and self.service_accessible
        )

    def to_dict(self):
        """Returns a dict containing the attributes"""

        return {attr_name: getattr(self, attr_name) for attr_name in self.__dataclass_fields__}



@dataclass
class VectorDbConfig:
    """Configuration for checking a vector database."""

    name: str
    langchain_module: str
    dependencies: list[str] = field(default_factory=list)
    service_check: Callable[[], bool] | None = None
    metadata_collector: Callable[[], dict[str, Any]] | None = None


class VectorDbDiscovery:
    """Discovers available LangChain vector databases."""

    def __init__(self):
        self._configs = self._get_vectordb_configs()

    def _get_vectordb_configs(self) -> list[VectorDbConfig]:
        """Get configurations for all known LangChain vector databases."""
        return [
            # Local/Embedded Databases
            VectorDbConfig(
                name="FAISS",
                langchain_module="langchain_community.vectorstores.faiss",
                dependencies=["faiss-cpu", "faiss-gpu"],  # Either one works
                metadata_collector=self._get_faiss_metadata,
            ),
            VectorDbConfig(
                name="Chroma",
                langchain_module="langchain_chroma",
                dependencies=["chromadb"],
                service_check=self._check_chroma_service,
                metadata_collector=self._get_chroma_metadata,
            ),
            VectorDbConfig(
                name="Qdrant (Local)",
                langchain_module="langchain_qdrant",
                dependencies=["qdrant-client"],
                service_check=lambda: self._check_service_port("localhost", 6333),
                metadata_collector=lambda: {"default_port": 6333, "type": "local"},
            ),
            # Cloud Services
            VectorDbConfig(
                name="Pinecone",
                langchain_module="langchain_pinecone",
                dependencies=["pinecone-client"],
                service_check=self._check_pinecone_auth,
                metadata_collector=self._get_pinecone_metadata,
            ),
            VectorDbConfig(
                name="Weaviate",
                langchain_module="langchain_community.vectorstores.weaviate",
                dependencies=["weaviate-client"],
                service_check=self._check_weaviate_service,
                metadata_collector=self._get_weaviate_metadata,
            ),
            # Database Extensions
            VectorDbConfig(
                name="PostgreSQL (pgvector)",
                langchain_module="langchain_community.vectorstores.pgvector",
                dependencies=["psycopg2-binary", "pgvector"],
                service_check=lambda: self._check_service_port("localhost", 5432),
                metadata_collector=lambda: {
                    "default_port": 5432,
                    "extension": "pgvector",
                },
            ),
            VectorDbConfig(
                name="Redis",
                langchain_module="langchain_community.vectorstores.redis",
                dependencies=["redis"],
                service_check=lambda: self._check_service_port("localhost", 6379),
                metadata_collector=lambda: {"default_port": 6379},
            ),
            # Additional Databases
            VectorDbConfig(
                name="Milvus",
                langchain_module="langchain_community.vectorstores.milvus",
                dependencies=["pymilvus"],
                service_check=lambda: self._check_service_port("localhost", 19530),
                metadata_collector=lambda: {"default_port": 19530},
            ),
            VectorDbConfig(
                name="Elasticsearch",
                langchain_module="langchain_community.vectorstores.elasticsearch",
                dependencies=["elasticsearch"],
                service_check=lambda: self._check_service_port("localhost", 9200),
                metadata_collector=lambda: {"default_port": 9200},
            ),
            VectorDbConfig(
                name="MongoDB Atlas",
                langchain_module="langchain_community.vectorstores.mongodb_atlas",
                dependencies=["pymongo"],
                metadata_collector=lambda: {"type": "cloud", "atlas_required": True},
            ),
            VectorDbConfig(
                name="Neo4j",
                langchain_module="langchain_community.vectorstores.neo4j_vector",
                dependencies=["neo4j"],
                service_check=lambda: self._check_service_port("localhost", 7687),
                metadata_collector=lambda: {"default_port": 7687, "bolt_port": 7687},
            ),
        ]

    def discover_available(
        self, *, check_services: bool = True
    ) -> dict[str, VectorDbStatus]:
        """
        Discover all available vector databases.

        Args:
            check_services: Whether to check if services are running (can be slow)

        Returns:
            Dictionary mapping database names to their status

        Example:

            >>> discovery = VectorDbDiscovery()
            >>> available = discovery.discover_available()
            >>> ready_dbs = {name: status for name, status in available.items() if status.is_ready}
        """
        results = {}

        for config in self._configs:
            status = self._check_vectordb(config, check_services=check_services)
            results[config.name] = status

        return results

    def get_ready_databases(
        self, *, check_services: bool = True
    ) -> dict[str, VectorDbStatus]:
        """Get only the databases that are ready to use."""
        all_dbs = self.discover_available(check_services=check_services)
        return {name: status for name, status in all_dbs.items() if status.is_ready}

    def get_installable_databases(self) -> dict[str, VectorDbStatus]:
        """Get databases where LangChain integration exists but dependencies are missing."""
        all_dbs = self.discover_available(check_services=False)
        return {
            name: status
            for name, status in all_dbs.items()
            if status.langchain_available and not status.dependencies_met
        }

    def _check_vectordb(
        self, config: VectorDbConfig, *, check_services: bool
    ) -> VectorDbStatus:
        """Check the status of a specific vector database."""
        status = VectorDbStatus(name=config.name)

        # Check LangChain integration
        status.langchain_available = self._check_import(config.langchain_module)
        if not status.langchain_available:
            status.errors.append(
                f"LangChain module '{config.langchain_module}' not available"
            )

        # Check dependencies
        if config.dependencies:
            missing_deps = []
            available_deps = []

            for dep in config.dependencies:
                if self._check_package(dep):
                    available_deps.append(dep)
                else:
                    missing_deps.append(dep)

            # Some packages have alternatives (like faiss-cpu vs faiss-gpu)
            status.dependencies_met = len(available_deps) > 0

            if missing_deps and not available_deps:
                status.errors.append(f"Missing dependencies: {missing_deps}")
            elif missing_deps:
                status.warnings.append(
                    f"Alternative dependencies available: {missing_deps}"
                )
                status.metadata["available_dependencies"] = available_deps
                status.metadata["missing_dependencies"] = missing_deps
        else:
            status.dependencies_met = True

        # Check service availability
        if check_services and config.service_check:
            try:
                status.service_accessible = config.service_check()
                if not status.service_accessible:
                    status.warnings.append(
                        "Service not accessible (may need to be started)"
                    )
            except Exception as e:
                status.service_accessible = False
                status.errors.append(f"Service check failed: {str(e)}")
        else:
            status.service_accessible = True  # Assume accessible if not checking

        # Collect metadata
        if config.metadata_collector:
            try:
                status.metadata.update(config.metadata_collector())
            except Exception as e:
                status.warnings.append(f"Could not collect metadata: {str(e)}")

        return status

    def _check_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def _check_package(self, package_name: str) -> bool:
        """Check if a package is installed."""
        try:
            importlib.import_module(package_name.replace('-', '_'))
            return True
        except ImportError:
            return False

    def _check_service_port(
        self, host: str, port: int, *, timeout: float = 2.0
    ) -> bool:
        """Check if a service is running on a specific port."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception:
            return False

    # Metadata collectors
    def _get_faiss_metadata(self) -> dict[str, Any]:
        """Get FAISS-specific metadata."""
        metadata = {"type": "local", "gpu_support": False}
        try:
            import faiss

            metadata["version"] = (
                faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
            )
            # Check for GPU support
            try:
                faiss.StandardGpuResources()
                metadata["gpu_support"] = True
            except:
                pass
        except ImportError:
            pass
        return metadata

    def _get_chroma_metadata(self) -> dict[str, Any]:
        """Get Chroma-specific metadata."""
        metadata = {"type": "local", "default_port": 8000}
        try:
            import chromadb

            metadata["version"] = (
                chromadb.__version__ if hasattr(chromadb, '__version__') else "unknown"
            )
        except ImportError:
            pass
        return metadata

    def _get_pinecone_metadata(self) -> dict[str, Any]:
        """Get Pinecone-specific metadata."""
        metadata = {"type": "cloud"}
        has_api_key = bool(os.getenv("PINECONE_API_KEY"))
        metadata["api_key_configured"] = has_api_key
        if not has_api_key:
            metadata["setup_hint"] = "Set PINECONE_API_KEY environment variable"
        return metadata

    def _get_weaviate_metadata(self) -> dict[str, Any]:
        """Get Weaviate-specific metadata."""
        return {
            "type": "hybrid",
            "default_port": 8080,
            "supports_cloud": True,
            "supports_embedded": True,
        }

    # Service checkers
    def _check_chroma_service(self) -> bool:
        """Check if Chroma server is accessible."""
        # Chroma can run embedded or as a server
        return (
            self._check_service_port("localhost", 8000) or True
        )  # Embedded always works

    def _check_pinecone_auth(self) -> bool:
        """Check if Pinecone credentials are available."""
        return bool(os.getenv("PINECONE_API_KEY"))

    def _check_weaviate_service(self) -> bool:
        """Check if Weaviate is accessible."""
        return self._check_service_port("localhost", 8080) or bool(  # Local server
            os.getenv("WEAVIATE_URL")
        )  # Cloud instance


def print_discovery_results(results: dict[str, VectorDbStatus]) -> None:
    """
    Print a formatted report of discovery results.

    Args:
        results: Results from VectorDbDiscovery.discover_available()

    Example:

        >>> discovery = VectorDbDiscovery()
        >>> results = discovery.discover_available()
        >>> print_discovery_results(results)  # doctest: +SKIP
    """
    ready = []
    needs_deps = []
    needs_service = []
    unavailable = []

    for name, status in results.items():
        if status.is_ready:
            ready.append((name, status))
        elif status.langchain_available and status.dependencies_met:
            needs_service.append((name, status))
        elif status.langchain_available:
            needs_deps.append((name, status))
        else:
            unavailable.append((name, status))

    print("🔍 LangChain Vector Database Discovery Results")
    print("=" * 50)

    if ready:
        print(f"\n✅ Ready to Use ({len(ready)}):")
        for name, status in ready:
            metadata_str = ""
            if status.metadata:
                key_info = []
                if "type" in status.metadata:
                    key_info.append(f"type: {status.metadata['type']}")
                if "version" in status.metadata:
                    key_info.append(f"v{status.metadata['version']}")
                if key_info:
                    metadata_str = f" ({', '.join(key_info)})"
            print(f"  • {name}{metadata_str}")

    if needs_deps:
        print(f"\n📦 Missing Dependencies ({len(needs_deps)}):")
        for name, status in needs_deps:
            missing = status.metadata.get("missing_dependencies", [])
            if missing:
                print(f"  • {name}: pip install {' '.join(missing)}")
            else:
                print(f"  • {name}: check dependencies")

    if needs_service:
        print(f"\n🚀 Service Not Running ({len(needs_service)}):")
        for name, status in needs_service:
            hints = []
            if "default_port" in status.metadata:
                hints.append(f"port {status.metadata['default_port']}")
            if "setup_hint" in status.metadata:
                hints.append(status.metadata["setup_hint"])
            hint_str = f" ({', '.join(hints)})" if hints else ""
            print(f"  • {name}{hint_str}")

    if unavailable:
        print(f"\n❌ Not Available ({len(unavailable)}):")
        for name, status in unavailable:
            print(f"  • {name}: LangChain integration not installed")

    print(
        f"\nSummary: {len(ready)} ready, {len(needs_deps)} need deps, {len(needs_service)} need service"
    )


# Main interface functions
def get_available_vectordbs(
    *, check_services: bool = True
) -> dict[str, VectorDbStatus]:
    """
    Get all available vector databases with their status.

    Args:
        check_services: Whether to check if database services are running

    Returns:
        Dictionary mapping database names to their status information

    Example:

        >>> available = get_available_vectordbs()
        >>> ready_dbs = {name: status for name, status in available.items() if status.is_ready}
        >>> print(f"Found {len(ready_dbs)} ready databases")  # doctest: +SKIP
        
    """
    discovery = VectorDbDiscovery()
    return discovery.discover_available(check_services=check_services)


def get_ready_vectordbs(*, check_services: bool = True) -> list[str]:
    """
    Get names of vector databases that are ready to use.

    Args:
        check_services: Whether to check if database services are running

    Returns:
        List of database names that are ready to use

    Example:

        >>> ready = get_ready_vectordbs()
        >>> print(f"Ready databases: {', '.join(ready)}")  # doctest: +SKIP

    """
    discovery = VectorDbDiscovery()
    ready_dbs = discovery.get_ready_databases(check_services=check_services)
    return list(ready_dbs.keys())


if __name__ == "__main__":
    # Example usage
    print("Discovering LangChain vector databases...")

    discovery = VectorDbDiscovery()
    results = discovery.discover_available(check_services=True)

    print_discovery_results(results)

    # Show just the ready ones
    ready = discovery.get_ready_databases()
    if ready:
        print(f"\n🎯 Quick Start - Ready Databases:")
        for name in ready.keys():
            print(
                f"  from langchain_community.vectorstores import {name.lower().replace(' ', '_')}"
            )
