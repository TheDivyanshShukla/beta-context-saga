"""
Command-line interface for ContextSaga.
This module provides a command-line interface for interacting with ContextSaga.
"""

import argparse
import os
import sys

from rich.console import Console

from beta_context_saga.core import ContextSaga
from beta_context_saga.memory_db import MemoryDatabase
from beta_context_saga.rich_display import (
    display_chunks_info,
    display_memory_actions,
    display_memory_items,
    display_processing_stats,
    display_search_results,
    print_welcome,
)

console = Console()


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="ContextSaga - AI-powered memory management system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Process file command
    process_parser = subparsers.add_parser("process", help="Process a file or text")
    process_parser.add_argument("--file", "-f", help="File to process")
    process_parser.add_argument("--text", "-t", help="Text to process")
    process_parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from checkpoint"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memory database")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit", "-l", type=int, default=10, help="Maximum number of results"
    )

    # Show memory command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument(
        "--limit", "-l", type=int, default=10, help="Maximum number of memories to list"
    )

    # Delete memory command
    delete_parser = subparsers.add_parser("delete", help="Delete memory by ID")
    delete_parser.add_argument("id", help="Memory ID to delete")

    return parser


def process_command(args: argparse.Namespace) -> None:
    """Handle the process command."""
    saga = ContextSaga()

    if args.file:
        if not os.path.exists(args.file):
            console.print(f"[bold red]Error:[/bold red] File not found: {args.file}")
            return

        console.print(f"[bold]Processing file:[/bold] {args.file}")
        resume = not args.no_resume
        results = saga.process_file(args.file, resume=resume)

        display_processing_stats(results)

        if results["actions"]:
            display_memory_actions(results["actions"])

    elif args.text:
        console.print("[bold]Processing text[/bold]")

        # Check if text is long enough to split into chunks
        token_count = saga.token_counter.num_tokens_from_string(args.text)
        if token_count > saga.max_tokens_per_chunk:
            chunks = saga.token_counter.split_text_by_tokens(
                args.text, max_tokens=saga.max_tokens_per_chunk, overlap_tokens=saga.token_overlap
            )
            display_chunks_info(chunks)
            results = saga.process_large_content(args.text)
        else:
            results = saga.process_content(args.text)

        display_processing_stats(results)

        if results["actions"]:
            display_memory_actions(results["actions"])

    else:
        console.print("[bold red]Error:[/bold red] Either --file or --text must be specified")


def search_command(args: argparse.Namespace) -> None:
    """Handle the search command."""
    db = MemoryDatabase()

    console.print(f"[bold]Searching for:[/bold] {args.query}")
    memory_results = db.search_by_text(args.query, limit=args.limit)

    results = []
    for memory, score in memory_results:
        results.append(
            {
                "id": memory.id,
                "content": memory.content,
                "importance": memory.importance,
                "tags": memory.tags,
                "score": score,
                "created_at": memory.created_at.isoformat(),
                "source": memory.source,
            }
        )

    if results:
        display_search_results(results)
    else:
        console.print("[yellow]No results found[/yellow]")


def list_command(args: argparse.Namespace) -> None:
    """Handle the list command."""
    db = MemoryDatabase()

    memories = db.get_all_memories(limit=args.limit)

    if memories:
        display_memory_items(memories)
    else:
        console.print("[yellow]No memories found in database[/yellow]")


def delete_command(args: argparse.Namespace) -> None:
    """Handle the delete command."""
    db = MemoryDatabase()

    success = db.delete_memory(args.id)

    if success:
        console.print(f"[green]Successfully deleted memory with ID:[/green] {args.id}")
    else:
        console.print(f"[bold red]Error:[/bold red] Failed to delete memory with ID: {args.id}")


def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    print_welcome()

    if args.command == "process":
        process_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "list":
        list_command(args)
    elif args.command == "delete":
        delete_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)
