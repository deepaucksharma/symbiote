#!/usr/bin/env python3
"""
Main CLI for Symbiote - Cognitive Prosthetic.

Usage:
    sym "text"              - Capture a thought/task
    sym search "query"      - Search for context
    sym suggest             - Get suggestions
    sym review              - Review suggestions
    sym daemon start        - Start the daemon
    sym daemon stop         - Stop the daemon
    sym daemon status       - Check daemon status
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from loguru import logger

console = Console()

# Default daemon URL
DAEMON_URL = "http://localhost:8765"


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Symbiote - Cognitive Prosthetic CLI."""
    if ctx.invoked_subcommand is None:
        # If no subcommand, treat the argument as capture text
        if len(sys.argv) > 1:
            text = " ".join(sys.argv[1:])
            asyncio.run(capture_text(text))
        else:
            click.echo(ctx.get_help())


@cli.command(name="capture")
@click.argument("text")
@click.option("--type", "-t", type=click.Choice(["task", "note", "question"]), default="note")
@click.option("--project", "-p", help="Project to associate with")
@click.option("--voice", is_flag=True, help="Indicate this is from voice input")
def capture(text: str, type: str, project: Optional[str], voice: bool):
    """Capture a thought, task, or note."""
    asyncio.run(capture_text(text, type, project, "voice" if voice else "text"))


async def capture_text(
    text: str,
    type: str = "note",
    project: Optional[str] = None,
    source: str = "text"
):
    """Send capture request to daemon."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DAEMON_URL}/capture",
                json={
                    "text": text,
                    "type": type,
                    "source": source,
                    "project": project
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✓[/green] Captured: {data.get('id', 'unknown')}")
            else:
                console.print(f"[red]Failed to capture[/red]: {response.text}")
    
    except httpx.ConnectError:
        console.print("[red]Cannot connect to daemon. Is it running?[/red]")
        console.print("Start with: [cyan]sym daemon start[/cyan]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command()
@click.argument("query")
@click.option("--project", "-p", help="Filter by project")
@click.option("--limit", "-l", default=10, help="Max results")
def search(query: str, project: Optional[str], limit: int):
    """Search for context."""
    asyncio.run(search_context(query, project, limit))


async def search_context(query: str, project: Optional[str], limit: int):
    """Send search request to daemon."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            progress.add_task(description="Searching...", total=None)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{DAEMON_URL}/context",
                    params={
                        "q": query,
                        "project": project,
                        "limit": limit
                    },
                    timeout=5.0
                )
        
        if response.status_code == 200:
            data = response.json()
            display_search_results(data)
        else:
            console.print(f"[red]Search failed:[/red] {response.text}")
    
    except httpx.ConnectError:
        console.print("[red]Cannot connect to daemon[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


def display_search_results(data: dict):
    """Display search results in a nice table."""
    results = data.get("results", [])
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = Table(title=f"Search Results ({data.get('latency_ms', 0):.1f}ms)")
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Type", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Snippet", no_wrap=False)
    
    for r in results:
        table.add_row(
            r.get("title", "Untitled"),
            r.get("source", "unknown"),
            f"{r.get('score', 0):.2f}",
            r.get("snippet", "")[:100] + "..."
        )
    
    console.print(table)
    
    # Show quick actions if available
    actions = data.get("quick_actions", [])
    if actions:
        console.print("\n[bold]Quick Actions:[/bold]")
        for i, action in enumerate(actions, 1):
            console.print(f"  {i}. {action.get('label', 'Unknown action')}")
    
    # Show suggestions if available
    suggestions = data.get("suggestions", [])
    if suggestions:
        console.print("\n[bold]Suggestions:[/bold]")
        for s in suggestions:
            console.print(f"  • {s}")


@cli.command()
@click.option("--all", is_flag=True, help="Show all suggestions")
def suggest(all: bool):
    """Get current suggestions."""
    asyncio.run(get_suggestions(all))


async def get_suggestions(show_all: bool):
    """Get suggestions from daemon."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DAEMON_URL}/suggestions",
                params={"all": show_all},
                timeout=5.0
            )
        
        if response.status_code == 200:
            data = response.json()
            display_suggestions(data)
        else:
            console.print(f"[red]Failed to get suggestions:[/red] {response.text}")
    
    except httpx.ConnectError:
        console.print("[red]Cannot connect to daemon[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


def display_suggestions(data: dict):
    """Display suggestions."""
    suggestions = data.get("suggestions", [])
    
    if not suggestions:
        console.print("[green]No pending suggestions[/green]")
        return
    
    console.print("[bold]Current Suggestions:[/bold]\n")
    
    for i, s in enumerate(suggestions, 1):
        kind = s.get("kind", "unknown")
        text = s.get("text", "")
        confidence = s.get("confidence", "medium")
        
        # Color code by confidence
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "white")
        
        console.print(f"{i}. [{conf_color}]{kind.upper()}[/{conf_color}]: {text}")
        
        # Show receipts if verbose
        if s.get("receipts_id"):
            console.print(f"   [dim]Receipt: {s['receipts_id']}[/dim]")


@cli.command()
@click.argument("suggestion_id", required=False)
@click.option("--accept", "-a", is_flag=True, help="Accept suggestion")
@click.option("--reject", "-r", is_flag=True, help="Reject suggestion")
def review(suggestion_id: Optional[str], accept: bool, reject: bool):
    """Review and act on suggestions."""
    if accept and reject:
        console.print("[red]Cannot both accept and reject[/red]")
        return
    
    asyncio.run(review_suggestion(suggestion_id, accept, reject))


async def review_suggestion(
    suggestion_id: Optional[str],
    accept: bool,
    reject: bool
):
    """Review a suggestion."""
    try:
        async with httpx.AsyncClient() as client:
            if suggestion_id and (accept or reject):
                # Act on specific suggestion
                response = await client.post(
                    f"{DAEMON_URL}/suggestions/{suggestion_id}",
                    json={"action": "accept" if accept else "reject"},
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    action = "Accepted" if accept else "Rejected"
                    console.print(f"[green]{action} suggestion {suggestion_id}[/green]")
                else:
                    console.print(f"[red]Failed:[/red] {response.text}")
            else:
                # Show suggestions for review
                response = await client.get(
                    f"{DAEMON_URL}/suggestions",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    display_suggestions(data)
                    console.print("\n[dim]Use 'sym review <id> --accept' or '--reject' to act[/dim]")
    
    except httpx.ConnectError:
        console.print("[red]Cannot connect to daemon[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.group()
def daemon():
    """Manage the Symbiote daemon."""
    pass


@daemon.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
def start(config: Optional[str]):
    """Start the Symbiote daemon."""
    console.print("[cyan]Starting Symbiote daemon...[/cyan]")
    
    # Import here to avoid circular dependencies
    from ..daemon.main import main as daemon_main
    
    try:
        asyncio.run(daemon_main(config))
    except KeyboardInterrupt:
        console.print("\n[yellow]Daemon stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Daemon error:[/red] {e}")
        logger.exception("Daemon crashed")


@daemon.command()
def stop():
    """Stop the Symbiote daemon."""
    asyncio.run(stop_daemon())


async def stop_daemon():
    """Send stop signal to daemon."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DAEMON_URL}/shutdown",
                timeout=5.0
            )
            
            if response.status_code == 200:
                console.print("[green]Daemon stopped[/green]")
            else:
                console.print(f"[red]Failed to stop daemon:[/red] {response.text}")
    
    except httpx.ConnectError:
        console.print("[yellow]Daemon not running[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@daemon.command()
def status():
    """Check daemon status."""
    asyncio.run(check_status())


async def check_status():
    """Check if daemon is running and get stats."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DAEMON_URL}/status",
                timeout=2.0
            )
            
            if response.status_code == 200:
                data = response.json()
                console.print("[green]✓ Daemon is running[/green]")
                
                # Display stats
                if data.get("stats"):
                    stats = data["stats"]
                    console.print(f"\nUptime: {stats.get('uptime', 'unknown')}")
                    console.print(f"Captures: {stats.get('capture_count', 0)}")
                    console.print(f"Searches: {stats.get('search_count', 0)}")
                    console.print(f"Memory: {stats.get('memory_mb', 0):.1f} MB")
            else:
                console.print("[red]Daemon error[/red]")
    
    except httpx.ConnectError:
        console.print("[red]✗ Daemon is not running[/red]")
        console.print("Start with: [cyan]sym daemon start[/cyan]")
    except Exception as e:
        console.print(f"[red]Error checking status:[/red] {e}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()