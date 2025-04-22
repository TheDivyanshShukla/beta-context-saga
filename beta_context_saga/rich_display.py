"""
Rich display utilities for ContextSaga data.
This module provides functions to display ContextSaga data in nicely formatted tables.
"""

import datetime
from typing import Any

from models import MemoryItem
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def display_memory_items(memories: list[MemoryItem], title: str = "Memory Items") -> None:
    """
    Display a list of memory items in a nicely formatted table.

    Args:
        memories: List of MemoryItem objects to display
        title: Title for the table
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("ID", style="dim")
    table.add_column("Content", style="green")
    table.add_column("Tags", style="cyan")
    table.add_column("Importance", style="yellow")
    table.add_column("Created", style="blue")
    table.add_column("Source", style="magenta")

    for memory in memories:
        table.add_row(
            str(memory.id),
            memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
            ", ".join(memory.tags) if memory.tags else "",
            str(memory.importance),
            memory.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            memory.source,
        )

    console.print(table)


def display_search_results(results: list[dict[str, Any]], title: str = "Search Results") -> None:
    """
    Display search results in a nicely formatted table.

    Args:
        results: List of search result dictionaries
        title: Title for the table
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("ID", style="dim")
    table.add_column("Content", style="green")
    table.add_column("Tags", style="cyan")
    table.add_column("Similarity", style="yellow")
    table.add_column("Created", style="blue")
    table.add_column("Source", style="magenta")

    for result in results:
        # Convert ISO format date string to datetime for formatting
        try:
            created_at = datetime.datetime.fromisoformat(result.get("created_at", ""))
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            created_at_str = "Unknown"

        table.add_row(
            str(result.get("id", "N/A")),
            result.get("content", "")[:100] + "..."
            if len(result.get("content", "")) > 100
            else result.get("content", ""),
            ", ".join(result.get("tags", [])),
            f"{result.get('score', 0):.3f}",
            created_at_str,
            result.get("source", "Unknown"),
        )

    console.print(table)


def display_processing_stats(results: dict[str, Any], title: str = "Processing Results") -> None:
    """
    Display processing statistics in a nicely formatted panel.

    Args:
        results: Dictionary containing processing statistics
        title: Title for the panel
    """
    # Create a table for the statistics
    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Search Results", str(len(results.get("search_results", []))))
    table.add_row("Actions", str(len(results.get("actions", []))))

    # CRUD operations
    created = len(results.get("results", {}).get("created", []))
    updated = len(results.get("results", {}).get("updated", []))
    deleted = len(results.get("results", {}).get("deleted", []))

    table.add_row("Items Created", str(created))
    table.add_row("Items Updated", str(updated))
    table.add_row("Items Deleted", str(deleted))

    panel = Panel(table, title=title, border_style="blue")
    console.print(panel)


def create_progress_bar(description: str = "Processing") -> Progress:
    """
    Create a rich progress bar for tracking progress.

    Args:
        description: Description for the progress bar

    Returns:
        Progress: A Rich Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )


def display_chunks_info(chunks: list[str], title: str = "Content Chunks") -> None:
    """
    Display information about content chunks.

    Args:
        chunks: List of content chunks
        title: Title for the table
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Chunk #", style="dim")
    table.add_column("Length (chars)", style="cyan")
    table.add_column("Preview", style="green")

    for i, chunk in enumerate(chunks):
        table.add_row(str(i + 1), str(len(chunk)), chunk[:50] + "..." if len(chunk) > 50 else chunk)

    console.print(table)


def display_memory_actions(actions: list[dict[str, Any]], title: str = "Memory Actions") -> None:
    """
    Display memory actions in a nicely formatted table.

    Args:
        actions: List of memory action dictionaries
        title: Title for the table
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Action", style="bold cyan")
    table.add_column("ID", style="dim")
    table.add_column("Details", style="green")

    for action in actions:
        action_type = action.get("action", "unknown")
        action_id = action.get("id", "N/A")

        if action_type == "create":
            item = action.get("item", {})
            content = item.get("content", "")
            details = f"Content: {content[:50]}..." if len(content) > 50 else f"Content: {content}"
            details += f"\nTags: {', '.join(item.get('tags', []))}"
            details += f"\nImportance: {item.get('importance', 0)}"
        elif action_type == "update":
            item = action.get("item", {})
            fields = [f"{key}: {value}" for key, value in item.items()]
            details = "\n".join(fields)
        elif action_type == "delete":
            details = "Delete memory item"
        else:
            details = "Unknown action"

        table.add_row(action_type.upper(), str(action_id), details)

    console.print(table)


def print_welcome() -> None:
    """Print a welcome message with ContextSaga info."""
    console.print(
        Panel(
            "[bold green]ContextSaga[/bold green] - An AI-powered memory management system\n\n"
            "This tool helps process and organize textual information into a searchable memory\
database.",
            title="Welcome",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    from .memory_db import MemoryDatabase

    db = MemoryDatabase()
    memories = db.list_memories()
    display_memory_items(memories)
