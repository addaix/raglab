"""
LangChain Vector Database Installation Helper

Provides installation scripts and setup instructions for various vector databases
that can be used with LangChain. Generates copy-pasteable terminal commands.
"""

import platform
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Callable, Any, get_args
from enum import Enum
import shutil

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
    requires_confirmation: bool = True
    optional: bool = False
    note: Optional[str] = None


@dataclass
class VectorDBInstaller:
    """Installation configuration for a vector database."""
    name: str
    pip_packages: List[str] = field(default_factory=list)
    setup_steps: Dict[OSType, List[InstallStep]] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    post_install_notes: List[str] = field(default_factory=list)
    verification_commands: List[str] = field(default_factory=list)


class VectorDBInstallerRegistry:
    """Registry for vector database installers."""
    
    def __init__(self):
        self._installers: Dict[str, VectorDBInstaller] = {}
        self._register_default_installers()
    
    def _register_default_installers(self) -> None:
        """Register the default set of vector database installers."""
        # Qdrant
        self.register(self._create_qdrant_installer())
        
        # Chroma
        self.register(self._create_chroma_installer())
        
        # MongoDB Atlas
        self.register(self._create_mongodb_atlas_installer())
    
    def _create_qdrant_installer(self) -> VectorDBInstaller:
        """Create installer configuration for Qdrant."""
        return VectorDBInstaller(
            name="Qdrant",
            pip_packages=["qdrant-client", "langchain-qdrant"],
            setup_steps={
                OSType.MACOS: [
                    InstallStep(
                        description="Install Qdrant using Docker",
                        commands=[
                            "# Make sure Docker is installed and running first",
                            "# Download from: https://docs.docker.com/desktop/mac/install/",
                            "",
                            "# Pull Qdrant image",
                            "docker pull qdrant/qdrant",
                            "",
                            "# Run Qdrant (creates qdrant_storage folder in current directory)",
                            "docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant",
                            "",
                            "# Check if running",
                            "docker ps | grep qdrant"
                        ],
                        note="This runs Qdrant in Docker. Use 'docker stop qdrant' to stop, 'docker start qdrant' to restart."
                    ),
                    InstallStep(
                        description="Alternative: Install Qdrant binary directly (no Docker)",
                        commands=[
                            "# Download and install Qdrant binary",
                            "curl --proto '=https' --tlsv1.2 -sSf https://get.qdrant.io | sh",
                            "",
                            "# Run Qdrant (in a separate terminal or background)",
                            "./qdrant"
                        ],
                        optional=True,
                        note="This installs Qdrant binary in current directory"
                    )
                ],
                OSType.LINUX: [
                    InstallStep(
                        description="Install Qdrant using Docker",
                        commands=[
                            "# Make sure Docker is installed first",
                            "# Install Docker if needed: curl -fsSL https://get.docker.com | sh",
                            "",
                            "# Pull Qdrant image",
                            "docker pull qdrant/qdrant",
                            "",
                            "# Run Qdrant (creates qdrant_storage folder in current directory)",
                            "docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant",
                            "",
                            "# Check if running",
                            "docker ps | grep qdrant"
                        ],
                        note="Use 'docker stop qdrant' to stop, 'docker start qdrant' to restart."
                    ),
                    InstallStep(
                        description="Alternative: Download and run Qdrant binary",
                        commands=[
                            "# Download Qdrant binary",
                            "curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-musl.tar.gz -o qdrant.tar.gz",
                            "tar -xzf qdrant.tar.gz",
                            "",
                            "# Run Qdrant (in a separate terminal or background)",
                            "./qdrant"
                        ],
                        optional=True
                    )
                ],
                OSType.WINDOWS: [
                    InstallStep(
                        description="Install Qdrant using Docker Desktop",
                        commands=[
                            "REM Make sure Docker Desktop is installed and running",
                            "REM Download from: https://docs.docker.com/desktop/windows/install/",
                            "",
                            "REM Pull Qdrant image",
                            "docker pull qdrant/qdrant",
                            "",
                            "REM Run Qdrant",
                            "docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_storage:/qdrant/storage:z qdrant/qdrant",
                            "",
                            "REM Check if running",
                            "docker ps"
                        ]
                    )
                ]
            },
            post_install_notes=[
                "Qdrant will be available at http://localhost:6333",
                "Dashboard available at http://localhost:6333/dashboard",
                "To use Qdrant Cloud instead, sign up at https://cloud.qdrant.io"
            ],
            verification_commands=[
                "# Check if Qdrant is running",
                "curl http://localhost:6333/health",
                "",
                "# Test Python connection",
                "python -c \"from qdrant_client import QdrantClient; client = QdrantClient('localhost', port=6333); print('✅ Qdrant connection successful!')\""
            ]
        )
    
    def _create_chroma_installer(self) -> VectorDBInstaller:
        """Create installer configuration for Chroma."""
        return VectorDBInstaller(
            name="Chroma",
            pip_packages=["chromadb", "langchain-chroma"],
            setup_steps={
                OSType.MACOS: [
                    InstallStep(
                        description="Optional: Run Chroma as a server (not required for embedded mode)",
                        commands=[
                            "# Chroma works in embedded mode after pip install (no server needed)",
                            "# But if you want client-server mode, you can run:",
                            "",
                            "# Option 1: Run server with Docker",
                            "docker pull chromadb/chroma",
                            "docker run -d --name chroma -p 8000:8000 chromadb/chroma",
                            "",
                            "# Option 2: Run server with Python (after pip install)",
                            "# chroma run --host localhost --port 8000"
                        ],
                        optional=True,
                        note="Server mode is optional. Chroma works embedded after pip install."
                    )
                ],
                OSType.LINUX: [
                    InstallStep(
                        description="Optional: Run Chroma as a server (not required for embedded mode)",
                        commands=[
                            "# Chroma works in embedded mode after pip install (no server needed)",
                            "# But if you want client-server mode, you can run:",
                            "",
                            "# Option 1: Run server with Docker",
                            "docker pull chromadb/chroma",
                            "docker run -d --name chroma -p 8000:8000 chromadb/chroma",
                            "",
                            "# Option 2: Run server with Python (after pip install)",
                            "# chroma run --host localhost --port 8000"
                        ],
                        optional=True,
                        note="Server mode is optional. Chroma works embedded after pip install."
                    )
                ],
                OSType.WINDOWS: [
                    InstallStep(
                        description="Optional: Run Chroma as a server (not required for embedded mode)",
                        commands=[
                            "REM Chroma works in embedded mode after pip install (no server needed)",
                            "REM But if you want client-server mode, you can run:",
                            "",
                            "REM Option 1: Run server with Docker Desktop",
                            "docker pull chromadb/chroma",
                            "docker run -d --name chroma -p 8000:8000 chromadb/chroma",
                            "",
                            "REM Option 2: Run server with Python (after pip install)",
                            "REM chroma run --host localhost --port 8000"
                        ],
                        optional=True
                    )
                ]
            },
            post_install_notes=[
                "Chroma is ready to use in embedded mode after pip install!",
                "For client-server mode, run 'chroma run' or use Docker",
                "Embedded mode: Works immediately, no server needed",
                "Server mode: Run server, then connect clients to http://localhost:8000"
            ],
            verification_commands=[
                "# Test embedded mode (should work after pip install)",
                "python -c \"import chromadb; client = chromadb.Client(); print('✅ Chroma embedded mode works!')\"",
                "",
                "# If using server mode, test connection:",
                "# curl http://localhost:8000/api/v1/heartbeat"
            ]
        )
    
    def _create_mongodb_atlas_installer(self) -> VectorDBInstaller:
        """Create installer configuration for MongoDB Atlas."""
        return VectorDBInstaller(
            name="MongoDB Atlas",
            pip_packages=["pymongo", "langchain-mongodb"],
            environment_vars={
                "MONGODB_ATLAS_URI": "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
            },
            setup_steps={
                OSType.MACOS: [
                    InstallStep(
                        description="Set up MongoDB Atlas Cloud Account (Required)",
                        commands=[
                            "# MongoDB Atlas is a cloud service. Setup steps:",
                            "# ",
                            "# 1. Go to: https://www.mongodb.com/cloud/atlas/register",
                            "# 2. Create a free account (M0 tier available for testing)",
                            "# 3. Create a cluster (choose a region close to you)",
                            "# 4. Set up database access:",
                            "#    - Click 'Database Access' in left menu",
                            "#    - Add a database user with password",
                            "# 5. Set up network access:",
                            "#    - Click 'Network Access' in left menu", 
                            "#    - Add your IP address (or 0.0.0.0/0 for development)",
                            "# 6. Get connection string:",
                            "#    - Click 'Connect' on your cluster",
                            "#    - Choose 'Connect your application'",
                            "#    - Copy the connection string",
                            "# 7. Replace <username>, <password>, and <cluster> in the connection string"
                        ],
                        note="Atlas account required. Free tier available for testing."
                    ),
                    InstallStep(
                        description="Optional: Install local MongoDB for development",
                        commands=[
                            "# For local development (won't have Atlas Search features):",
                            "brew tap mongodb/brew",
                            "brew install mongodb-community",
                            "brew services start mongodb-community",
                            "",
                            "# Local connection string: mongodb://localhost:27017/"
                        ],
                        optional=True,
                        note="Local MongoDB doesn't support Atlas Vector Search"
                    )
                ],
                OSType.LINUX: [
                    InstallStep(
                        description="Set up MongoDB Atlas Cloud Account (Required)",
                        commands=[
                            "# MongoDB Atlas is a cloud service. Setup steps:",
                            "# ",
                            "# 1. Go to: https://www.mongodb.com/cloud/atlas/register",
                            "# 2. Create a free account (M0 tier available for testing)",
                            "# 3. Create a cluster (choose a region close to you)",
                            "# 4. Set up database access:",
                            "#    - Click 'Database Access' in left menu",
                            "#    - Add a database user with password",
                            "# 5. Set up network access:",
                            "#    - Click 'Network Access' in left menu", 
                            "#    - Add your IP address (or 0.0.0.0/0 for development)",
                            "# 6. Get connection string:",
                            "#    - Click 'Connect' on your cluster",
                            "#    - Choose 'Connect your application'",
                            "#    - Copy the connection string",
                            "# 7. Replace <username>, <password>, and <cluster> in the connection string"
                        ],
                        note="Atlas account required. Free tier available for testing."
                    ),
                    InstallStep(
                        description="Optional: Install local MongoDB for development",
                        commands=[
                            "# For Ubuntu/Debian (won't have Atlas Search features):",
                            "wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -",
                            "echo \"deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/7.0 multiverse\" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list",
                            "sudo apt-get update",
                            "sudo apt-get install -y mongodb-org",
                            "sudo systemctl start mongod",
                            "",
                            "# Local connection string: mongodb://localhost:27017/"
                        ],
                        optional=True,
                        note="Local MongoDB doesn't support Atlas Vector Search"
                    )
                ],
                OSType.WINDOWS: [
                    InstallStep(
                        description="Set up MongoDB Atlas Cloud Account (Required)",
                        commands=[
                            "REM MongoDB Atlas setup instructions:",
                            "REM ",
                            "REM 1. Go to: https://www.mongodb.com/cloud/atlas/register",
                            "REM 2. Create a free account",
                            "REM 3. Create a cluster",
                            "REM 4. Set up database access (create user)",
                            "REM 5. Set up network access (add IP)",
                            "REM 6. Get connection string from Connect button",
                            "REM 7. Replace username, password, cluster in connection string"
                        ]
                    )
                ]
            },
            post_install_notes=[
                "MongoDB Atlas requires cloud account setup",
                "Free M0 tier available for testing",
                "For vector search, create an Atlas Search index",
                "Set MONGODB_ATLAS_URI environment variable with your connection string"
            ],
            verification_commands=[
                "# Test connection (set MONGODB_ATLAS_URI first)",
                "export MONGODB_ATLAS_URI='your-connection-string-here'",
                "python -c \"from pymongo import MongoClient; import os; client = MongoClient(os.getenv('MONGODB_ATLAS_URI')); print('✅ MongoDB connection successful!')\"",
            ]
        )
    
    def register(self, installer: VectorDBInstaller) -> None:
        """
        Register a new vector database installer.
        
        Args:
            installer: VectorDBInstaller configuration
            
        Example:
            >>> registry = VectorDBInstallerRegistry()
            >>> custom_installer = VectorDBInstaller(name="CustomDB", pip_packages=["customdb"])
            >>> registry.register(custom_installer)
        """
        self._installers[installer.name.lower()] = installer
    
    def get_installer(self, name: str) -> Optional[VectorDBInstaller]:
        """Get installer by name (case-insensitive)."""
        return self._installers.get(name.lower())
    
    def list_available(self) -> List[str]:
        """Get list of available installer names."""
        return [installer.name for installer in self._installers.values()]


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
        # Default to Linux for unknown systems
        return OSType.LINUX


def _generate_install_commands(installer: VectorDBInstaller, os_type: OSType) -> str:
    """
    Generate simple, copy-pasteable installation commands.
    
    Args:
        installer: VectorDBInstaller configuration
        os_type: Target operating system
        
    Returns:
        Installation commands as string
    """
    lines = []
    
    # Header
    lines.append(f"# Installation commands for {installer.name}")
    lines.append(f"# OS: {os_type.value}")
    lines.append(f"# Copy and run these commands in your terminal")
    lines.append("")
    
    # Step 1: Python packages
    if installer.pip_packages:
        lines.append("# STEP 1: Install Python packages")
        lines.append(f"pip install {' '.join(installer.pip_packages)}")
        lines.append("")
    
    # Step 2: OS-specific setup
    if os_type in installer.setup_steps:
        step_num = 2
        for step in installer.setup_steps[os_type]:
            lines.append(f"# STEP {step_num}: {step.description}")
            if step.optional:
                lines.append("# (Optional - only if needed)")
            if step.note:
                lines.append(f"# Note: {step.note}")
            lines.append("#" + "-" * 50)
            
            for cmd in step.commands:
                lines.append(cmd)
            
            lines.append("")
            step_num += 1
    
    # Environment variables
    if installer.environment_vars:
        lines.append("# ENVIRONMENT VARIABLES")
        lines.append("# Add to your shell config (.bashrc, .zshrc, etc.):")
        lines.append("#" + "-" * 50)
        for key, value in installer.environment_vars.items():
            lines.append(f"export {key}=\"{value}\"")
        lines.append("")
    
    # Verification
    if installer.verification_commands:
        lines.append("# VERIFICATION")
        lines.append("# Run these to verify installation:")
        lines.append("#" + "-" * 50)
        for cmd in installer.verification_commands:
            lines.append(cmd)
        lines.append("")
    
    # Post-install notes
    if installer.post_install_notes:
        lines.append("# NOTES")
        lines.append("#" + "-" * 50)
        for note in installer.post_install_notes:
            lines.append(f"# • {note}")
    
    return "\n".join(lines)


def help_me_install_vectordb(
    vectordb: Optional[str] = None,
    *,
    copy_instructions_to_clipboard: bool = True,
    print_instructions: bool = True,
    check_current_status: bool = True
) -> Optional[str]:
    """
    Generate installation instructions for a vector database.
    
    Args:
        vectordb: Name of the vector database to install (case-insensitive).
                 If None or invalid, shows available options.
        copy_instructions_to_clipboard: Copy the commands to clipboard if available
        print_instructions: Print the installation commands to console
        check_current_status: Check current installation status first
        
    Returns:
        Installation commands as string, or None if vectordb not found
        
    Example:
        >>> # Show available databases
        >>> help_me_install_vectordb()  # doctest: +SKIP
        
        >>> # Get installation commands for Qdrant
        >>> commands = help_me_install_vectordb("qdrant", print_instructions=False)  # doctest: +SKIP
    """
    registry = VectorDBInstallerRegistry()
    
    # If no vectordb specified or invalid, show available options
    if not vectordb:
        available = registry.list_available()
        print("❓ No vector database specified. Available options:")
        for db_name in available:
            print(f"  • {db_name}")
        print(f"\nUsage: help_me_install_vectordb('{available[0].lower()}')")
        return None
    
    installer = registry.get_installer(vectordb)
    if not installer:
        available = registry.list_available()
        print(f"❌ Unknown vector database: '{vectordb}'")
        print(f"Available options: {', '.join(available)}")
        print(f"\nUsage: help_me_install_vectordb('{available[0].lower()}')")
        return None
    
    # Check current status if requested
    if check_current_status:
        try:
            from vectordb_discovery import VectorDbDiscovery
            discovery = VectorDbDiscovery()
            all_status = discovery.discover_available(check_services=True)
            
            # Find matching status
            for name, status in all_status.items():
                if name.lower() == installer.name.lower() or name.lower().startswith(installer.name.lower()):
                    print(f"📊 Current status of {installer.name}:")
                    print(f"  • LangChain available: {'✅' if status.langchain_available else '❌'}")
                    print(f"  • Dependencies met: {'✅' if status.dependencies_met else '❌'}")
                    print(f"  • Service accessible: {'✅' if status.service_accessible else '❌'}")
                    if status.errors:
                        print(f"  • Errors: {', '.join(status.errors)}")
                    if status.warnings:
                        print(f"  • Warnings: {', '.join(status.warnings)}")
                    print()
                    
                    # If already ready, note that
                    if status.is_ready:
                        print("✅ This vector database appears to be ready to use!")
                        print()
                    break
        except ImportError:
            print("ℹ️  Could not check current status (vectordb_discovery module not available)")
            print()
    
    # Detect OS
    os_type = _detect_os()
    
    # Generate commands
    commands = _generate_install_commands(installer, os_type)
    
    # Print instructions
    if print_instructions:
        print(f"🔧 Installation Instructions for {installer.name}")
        print("=" * 60)
        print("\n📋 Copy and run these commands in your terminal:\n")
        print(commands)
        print("\n" + "=" * 60)
        
        # Quick start reminder
        if installer.pip_packages:
            print(f"\n💡 Quick start: First run the pip install command:")
            print(f"   pip install {' '.join(installer.pip_packages)}")
            print(f"\nThen follow the additional setup steps above if needed.")
    
    # Copy to clipboard if available and requested
    if copy_instructions_to_clipboard:
        if HAS_CLIPBOARD:
            try:
                pyperclip.copy(commands)
                print("\n✅ Commands copied to clipboard!")
            except Exception as e:
                print(f"\n⚠️  Could not copy to clipboard: {e}")
        else:
            print("\n💡 Tip: Install 'pyperclip' to enable clipboard support")
            print("   pip install pyperclip")
    
    return commands


# Convenience functions for specific databases
def install_qdrant(**kwargs) -> Optional[str]:
    """
    Generate installation instructions for Qdrant.
    
    Example:
        >>> commands = install_qdrant(print_instructions=False)  # doctest: +SKIP
    """
    return help_me_install_vectordb("qdrant", **kwargs)


def install_chroma(**kwargs) -> Optional[str]:
    """
    Generate installation instructions for Chroma.
    
    Example:
        >>> commands = install_chroma(print_instructions=False)  # doctest: +SKIP
    """
    return help_me_install_vectordb("chroma", **kwargs)


def install_mongodb_atlas(**kwargs) -> Optional[str]:
    """
    Generate installation instructions for MongoDB Atlas.
    
    Example:
        >>> commands = install_mongodb_atlas(print_instructions=False)  # doctest: +SKIP
    """
    return help_me_install_vectordb("mongodb atlas", **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        db_name = sys.argv[1]
        help_me_install_vectordb(db_name)
    else:
        # Show available options
        print("Vector Database Installation Helper")
        print("=" * 40)
        help_me_install_vectordb()
        print("\nUsage: python vectordb_installer.py <database_name>")
        print("Example: python vectordb_installer.py qdrant")