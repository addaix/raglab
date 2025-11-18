"""Command-line interface for RAG module."""

import click
import json
import yaml
from pathlib import Path
from typing import Optional
import logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """RAG module command-line interface."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument('folder_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.option('--output', '-o', type=click.Path(), help='Output directory for index')
def init(folder_path, config, output):
    """Initialize a new RAG index."""
    click.echo(f"Initializing RAG index for: {folder_path}")

    # Create default config if not exists
    if not config:
        config_path = Path(folder_path) / 'rag_config.yaml'
        if not config_path.exists():
            default_config = {
                'sources': [
                    {
                        'type': 'folder',
                        'path': folder_path,
                        'exclude_patterns': [r'\.git', r'__pycache__', r'\.pyc$'],
                        'extensions': ['.md', '.txt', '.py'],
                        'recursive': True,
                    }
                ],
                'pipeline': {
                    'chunk_size': 400,
                    'chunk_overlap': 100,
                    'use_async': True,
                },
            }

            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)

            click.echo(f"Created default config: {config_path}")
            config = str(config_path)

    click.echo(f"Configuration: {config}")
    click.echo("Index initialized successfully!")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
@click.option('--force', '-f', is_flag=True, help='Force reindex all content')
@click.option('--progress', '-p', is_flag=True, help='Show progress bar')
def index(config, force, progress):
    """Index content according to configuration."""
    click.echo(f"Indexing from config: {config}")

    from raglab.rag.config import load_config
    from raglab.rag.pipeline import RAGPipeline

    # Load configuration
    cfg = load_config(config)

    # Create pipeline from config
    pipeline = RAGPipeline.from_config(cfg)

    # Progress callback
    def progress_callback(current, total):
        if progress:
            pct = int(100 * current / total)
            click.echo(f"Progress: {current}/{total} ({pct}%)")

    # Run indexing
    stats = pipeline.run(
        force_refresh=force,
        progress_callback=progress_callback if progress else None
    )

    # Display stats
    click.echo("\nIndexing complete!")
    click.echo(f"  Files: {stats['content_keys']}")
    click.echo(f"  Segments: {stats['segments']}")
    click.echo(f"  Vectors: {stats['vectors']}")
    click.echo(f"  New: {stats['new']}")
    click.echo(f"  Modified: {stats['modified']}")
    click.echo(f"  Deleted: {stats['deleted']}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
def update(config):
    """Update index with new/modified content."""
    click.echo(f"Updating index from config: {config}")

    from raglab.rag.config import load_config
    from raglab.rag.pipeline import RAGPipeline

    # Load configuration
    cfg = load_config(config)

    # Create pipeline
    pipeline = RAGPipeline.from_config(cfg)

    # Run update (incremental)
    stats = pipeline.run(force_refresh=False)

    # Display stats
    click.echo("\nUpdate complete!")
    if stats['refreshed'] == 0:
        click.echo("  No changes detected.")
    else:
        click.echo(f"  Refreshed: {stats['refreshed']}")
        click.echo(f"  New: {stats['new']}")
        click.echo(f"  Modified: {stats['modified']}")
        click.echo(f"  Deleted: {stats['deleted']}")


@cli.command()
@click.argument('query')
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
@click.option('-k', default=5, help='Number of results')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='text', help='Output format')
def search(query, config, k, format):
    """Search the index."""
    from raglab.rag.config import load_config
    from raglab.rag.pipeline import RAGPipeline

    # Load configuration
    cfg = load_config(config)

    # Create pipeline
    pipeline = RAGPipeline.from_config(cfg)

    # Search
    results = pipeline.search(query, k=k)

    # Display results
    if format == 'json':
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo(f"\nSearch results for: '{query}'")
        click.echo("=" * 60)
        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. Score: {result.get('score', 0):.4f}")
            click.echo(f"   Source: {result.get('key', 'unknown')}")
            click.echo(f"   Text: {result.get('text', '')[:200]}...")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
def stats(config):
    """Show index statistics."""
    from raglab.rag.config import load_config
    from raglab.rag.pipeline import RAGPipeline

    # Load configuration
    cfg = load_config(config)

    # Create pipeline
    pipeline = RAGPipeline.from_config(cfg)

    # Get stats
    stats = pipeline.get_stats()

    # Display
    click.echo("\nIndex Statistics")
    click.echo("=" * 60)
    click.echo(f"Total segments: {stats.get('total_segments', 0)}")
    click.echo(f"Total vectors: {stats.get('total_vectors', 0)}")
    click.echo(f"Indexed content keys: {stats.get('previous_content_keys', 0)}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
def validate(config):
    """Validate configuration file."""
    click.echo(f"Validating config: {config}")

    try:
        from raglab.rag.config import load_config, validate_config

        cfg = load_config(config)
        errors = validate_config(cfg)

        if not errors:
            click.echo("✓ Configuration is valid!")
        else:
            click.echo("✗ Configuration has errors:")
            for error in errors:
                click.echo(f"  - {error}")

    except Exception as e:
        click.echo(f"✗ Error loading config: {e}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
def clear(config):
    """Clear the index."""
    from raglab.rag.config import load_config
    from raglab.rag.pipeline import RAGPipeline

    if not click.confirm("Are you sure you want to clear the index?"):
        return

    # Load configuration
    cfg = load_config(config)

    # Create pipeline
    pipeline = RAGPipeline.from_config(cfg)

    # Clear
    if hasattr(pipeline.vector_store, 'clear'):
        pipeline.vector_store.clear()
        click.echo("Index cleared successfully!")
    else:
        click.echo("Vector store does not support clearing.")


@cli.command()
@click.argument('output_path', type=click.Path())
def export_config(output_path):
    """Export a sample configuration file."""
    sample_config = {
        'sources': [
            {
                'type': 'folder',
                'path': './docs',
                'exclude_patterns': [
                    r'\.git',
                    r'__pycache__',
                    r'\.pyc$',
                    r'node_modules',
                ],
                'extensions': ['.md', '.txt', '.py', '.js'],
                'recursive': True,
            }
        ],
        'pipeline': {
            'chunk_size': 400,
            'chunk_overlap': 100,
            'use_async': True,
            'max_workers': 4,
        },
        'embedder': {
            'type': 'openai',
            'model': 'text-embedding-3-small',
            'dimensions': 384,
        },
        'vector_store': {
            'type': 'chroma',
            'collection_name': 'rag_collection',
            'persist_directory': './chroma_db',
        },
        'refresh': {
            'strategy': 'simple',
        },
    }

    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Sample configuration exported to: {output_path}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output file for state')
def save_state(config, output):
    """Save pipeline state."""
    from raglab.rag.config import load_config
    from raglab.rag.pipeline import RAGPipeline

    # Load configuration
    cfg = load_config(config)

    # Create pipeline
    pipeline = RAGPipeline.from_config(cfg)

    # Save state
    pipeline.save_state(output)

    click.echo(f"State saved to: {output}")


@cli.command()
@click.option('--state', '-s', type=click.Path(exists=True), required=True, help='State file')
def load_state(state):
    """Load pipeline state."""
    from raglab.rag.pipeline import RAGPipeline

    # Load pipeline
    pipeline = RAGPipeline.load_state(state)

    # Show stats
    stats = pipeline.get_stats()
    click.echo(f"State loaded from: {state}")
    click.echo(f"  Segments: {stats.get('total_segments', 0)}")
    click.echo(f"  Vectors: {stats.get('total_vectors', 0)}")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
