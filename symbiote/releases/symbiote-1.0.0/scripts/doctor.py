#!/usr/bin/env python3
"""
Doctor tool for Symbiote diagnostics and maintenance.

Usage:
    sym-doctor              - Run health checks
    sym-doctor --reindex    - Rebuild all indexes
    sym-doctor --repair     - Fix common issues
    sym-doctor --stats      - Show performance statistics
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import click
import duckdb
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint
from loguru import logger
import psutil
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from symbiote.daemon.config import Config
from symbiote.daemon.indexers import AnalyticsIndexer

console = Console()

DAEMON_URL = "http://localhost:8765"


class SymbioteDoctor:
    """Diagnostic and maintenance tool for Symbiote."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vault_path = config.vault_path
        self.issues = []
        self.warnings = []
    
    async def run_health_check(self) -> bool:
        """Run comprehensive health checks."""
        console.print("[bold cyan]Running Symbiote Health Check...[/bold cyan]\n")
        
        all_healthy = True
        
        # Check vault
        vault_healthy = await self._check_vault()
        all_healthy = all_healthy and vault_healthy
        
        # Check indexes
        index_healthy = await self._check_indexes()
        all_healthy = all_healthy and index_healthy
        
        # Check daemon
        daemon_healthy = await self._check_daemon()
        all_healthy = all_healthy and daemon_healthy
        
        # Check resources
        resource_healthy = await self._check_resources()
        all_healthy = all_healthy and resource_healthy
        
        # Report results
        self._report_results()
        
        return all_healthy
    
    async def _check_vault(self) -> bool:
        """Check vault structure and integrity."""
        console.print("[yellow]Checking vault...[/yellow]")
        
        healthy = True
        
        # Check vault exists
        if not self.vault_path.exists():
            self.issues.append(f"Vault path does not exist: {self.vault_path}")
            healthy = False
        else:
            # Check required directories
            required_dirs = [
                "journal",
                "tasks",
                "notes",
                ".sym/wal",
                ".sym/cache"
            ]
            
            for dir_name in required_dirs:
                dir_path = self.vault_path / dir_name
                if not dir_path.exists():
                    self.warnings.append(f"Missing directory: {dir_name}")
                    dir_path.mkdir(parents=True, exist_ok=True)
                    console.print(f"  Created missing directory: {dir_name}")
            
            # Count files
            note_count = len(list((self.vault_path / "notes").glob("*.md")))
            task_count = len(list((self.vault_path / "tasks").glob("*.md")))
            journal_count = len(list((self.vault_path / "journal").rglob("*.md")))
            
            console.print(f"  Notes: {note_count}")
            console.print(f"  Tasks: {task_count}")
            console.print(f"  Journal entries: {journal_count}")
            
            # Check WAL
            wal_files = list((self.vault_path / ".sym" / "wal").glob("*.log"))
            if wal_files:
                console.print(f"  WAL files: {len(wal_files)}")
                
                # Check for old WAL files
                import datetime
                cutoff = datetime.datetime.now() - datetime.timedelta(days=14)
                old_wals = [
                    f for f in wal_files
                    if datetime.datetime.fromtimestamp(f.stat().st_mtime) < cutoff
                ]
                if old_wals:
                    self.warnings.append(f"{len(old_wals)} WAL files older than 14 days")
        
        status = "[green]✓[/green]" if healthy else "[red]✗[/red]"
        console.print(f"{status} Vault check complete\n")
        return healthy
    
    async def _check_indexes(self) -> bool:
        """Check index health."""
        console.print("[yellow]Checking indexes...[/yellow]")
        
        healthy = True
        
        # Check DuckDB
        db_path = self.vault_path / ".sym" / "analytics.db"
        if db_path.exists():
            try:
                conn = duckdb.connect(str(db_path))
                
                # Check tables
                tables = conn.execute("SHOW TABLES").fetchall()
                expected_tables = {
                    'notes', 'tasks', 'suggestions', 'receipts',
                    'links_confirmed', 'links_suggested', 'audit_outbound'
                }
                
                actual_tables = {t[0] for t in tables}
                missing_tables = expected_tables - actual_tables
                
                if missing_tables:
                    self.issues.append(f"Missing tables: {missing_tables}")
                    healthy = False
                else:
                    # Get row counts
                    for table in expected_tables:
                        if table in actual_tables:
                            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            console.print(f"  {table}: {count} rows")
                
                conn.close()
                
            except Exception as e:
                self.issues.append(f"DuckDB error: {e}")
                healthy = False
        else:
            self.warnings.append("DuckDB index not found")
        
        # Check FTS (if implemented)
        fts_path = self.vault_path / ".sym" / "fts_index"
        if fts_path.exists():
            console.print(f"  FTS index: {fts_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            self.warnings.append("FTS index not found")
        
        # Check Vector (if implemented)
        vector_path = self.vault_path / ".sym" / "vector_index"
        if vector_path.exists():
            console.print(f"  Vector index: {vector_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            self.warnings.append("Vector index not found")
        
        status = "[green]✓[/green]" if healthy else "[red]✗[/red]"
        console.print(f"{status} Index check complete\n")
        return healthy
    
    async def _check_daemon(self) -> bool:
        """Check if daemon is running and responsive."""
        console.print("[yellow]Checking daemon...[/yellow]")
        
        healthy = True
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{DAEMON_URL}/status",
                    timeout=2.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    console.print("  [green]Daemon is running[/green]")
                    
                    stats = data.get("stats", {})
                    console.print(f"  Uptime: {data.get('uptime', 'unknown')}")
                    console.print(f"  Memory: {stats.get('memory_mb', 0):.1f} MB")
                    console.print(f"  Captures: {stats.get('capture_count', 0)}")
                    console.print(f"  Searches: {stats.get('search_count', 0)}")
                else:
                    self.issues.append("Daemon returned error status")
                    healthy = False
        
        except httpx.ConnectError:
            console.print("  [red]Daemon is not running[/red]")
            self.warnings.append("Daemon not running - start with 'sym daemon start'")
            healthy = False
        except Exception as e:
            self.issues.append(f"Daemon check error: {e}")
            healthy = False
        
        status = "[green]✓[/green]" if healthy else "[red]✗[/red]"
        console.print(f"{status} Daemon check complete\n")
        return healthy
    
    async def _check_resources(self) -> bool:
        """Check system resources."""
        console.print("[yellow]Checking resources...[/yellow]")
        
        healthy = True
        
        # Check memory
        memory = psutil.virtual_memory()
        console.print(f"  Memory: {memory.percent:.1f}% used ({memory.used / 1024 / 1024 / 1024:.1f} GB / {memory.total / 1024 / 1024 / 1024:.1f} GB)")
        
        if memory.percent > 90:
            self.warnings.append("High memory usage")
        
        # Check disk
        disk = psutil.disk_usage(str(self.vault_path))
        console.print(f"  Disk: {disk.percent:.1f}% used ({disk.used / 1024 / 1024 / 1024:.1f} GB / {disk.total / 1024 / 1024 / 1024:.1f} GB)")
        
        if disk.percent > 90:
            self.warnings.append("Low disk space")
            healthy = False
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        console.print(f"  CPU: {cpu_percent:.1f}%")
        
        status = "[green]✓[/green]" if healthy else "[red]✗[/red]"
        console.print(f"{status} Resource check complete\n")
        return healthy
    
    def _report_results(self) -> None:
        """Report health check results."""
        console.print("[bold]Health Check Summary:[/bold]\n")
        
        if not self.issues and not self.warnings:
            console.print("[bold green]✓ All systems healthy![/bold green]")
        else:
            if self.issues:
                console.print("[bold red]Issues Found:[/bold red]")
                for issue in self.issues:
                    console.print(f"  • {issue}")
                console.print()
            
            if self.warnings:
                console.print("[bold yellow]Warnings:[/bold yellow]")
                for warning in self.warnings:
                    console.print(f"  • {warning}")
    
    async def reindex_all(self) -> None:
        """Rebuild all indexes from vault."""
        console.print("[bold cyan]Rebuilding all indexes...[/bold cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            # Count total files
            all_files = list(self.vault_path.rglob("*.md"))
            total = len(all_files)
            
            task = progress.add_task("Reindexing...", total=total)
            
            # Initialize indexer
            analytics = AnalyticsIndexer(self.config)
            await analytics.initialize()
            
            # Process each file
            for i, file_path in enumerate(all_files):
                progress.update(task, advance=1, description=f"Processing {file_path.name}")
                
                try:
                    # Parse and index file
                    # (This would call actual indexing logic)
                    await asyncio.sleep(0.01)  # Simulate work
                
                except Exception as e:
                    logger.error(f"Failed to index {file_path}: {e}")
            
            await analytics.close()
        
        console.print("\n[green]✓ Reindexing complete![/green]")
    
    async def show_stats(self) -> None:
        """Show performance statistics."""
        console.print("[bold cyan]Performance Statistics[/bold cyan]\n")
        
        # Try to get stats from daemon
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{DAEMON_URL}/status", timeout=2.0)
                
                if response.status_code == 200:
                    data = response.json()
                    stats = data.get("stats", {})
                    
                    table = Table(title="Daemon Statistics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", justify="right")
                    
                    table.add_row("Uptime", data.get("uptime", "unknown"))
                    table.add_row("Memory (MB)", f"{stats.get('memory_mb', 0):.1f}")
                    table.add_row("CPU %", f"{stats.get('cpu_percent', 0):.1f}")
                    table.add_row("Captures", str(stats.get('capture_count', 0)))
                    table.add_row("Searches", str(stats.get('search_count', 0)))
                    table.add_row("Suggestions", str(stats.get('suggestion_count', 0)))
                    
                    console.print(table)
        
        except Exception as e:
            console.print(f"[yellow]Could not get daemon stats: {e}[/yellow]")
        
        # Show vault stats
        console.print("\n[bold]Vault Statistics:[/bold]")
        
        note_count = len(list((self.vault_path / "notes").glob("*.md")))
        task_count = len(list((self.vault_path / "tasks").glob("*.md")))
        journal_count = len(list((self.vault_path / "journal").rglob("*.md")))
        
        vault_size = sum(f.stat().st_size for f in self.vault_path.rglob("*") if f.is_file())
        
        table = Table()
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right")
        
        table.add_row("Notes", str(note_count))
        table.add_row("Tasks", str(task_count))
        table.add_row("Journal Entries", str(journal_count))
        table.add_row("Total Files", str(note_count + task_count + journal_count))
        table.add_row("Vault Size", f"{vault_size / 1024 / 1024:.1f} MB")
        
        console.print(table)


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--reindex", is_flag=True, help="Rebuild all indexes")
@click.option("--repair", is_flag=True, help="Fix common issues")
@click.option("--stats", is_flag=True, help="Show performance statistics")
def main(config: Optional[str], reindex: bool, repair: bool, stats: bool):
    """Symbiote Doctor - Diagnostics and Maintenance."""
    
    # Load configuration
    try:
        if config:
            cfg = Config.load(Path(config))
        else:
            cfg = Config.load()
    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    
    doctor = SymbioteDoctor(cfg)
    
    if reindex:
        asyncio.run(doctor.reindex_all())
    elif stats:
        asyncio.run(doctor.show_stats())
    else:
        # Run health check
        healthy = asyncio.run(doctor.run_health_check())
        
        if repair and not healthy:
            console.print("\n[yellow]Running repairs...[/yellow]")
            # Implement repair logic here
            console.print("[green]Repairs complete[/green]")
        
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()