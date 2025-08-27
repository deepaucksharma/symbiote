#!/usr/bin/env python3
"""Setup script to initialize Symbiote vault with example data."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import frontmatter
import ulid
import click
from rich.console import Console

console = Console()


def create_example_notes(vault_path: Path):
    """Create example notes in the vault."""
    notes_dir = vault_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    examples = [
        {
            "title": "Q3 Strategy Planning",
            "content": """# Q3 Strategy Planning

## Key Objectives
- Improve system performance by 30%
- Launch new feature set
- Expand team by 2 engineers

## Metrics to Track
- User engagement
- System latency
- Feature adoption rate""",
            "tags": ["strategy", "q3", "planning"]
        },
        {
            "title": "WebRTC Implementation Notes",
            "content": """# WebRTC Implementation Notes

## Architecture Decisions
- Use TURN server for NAT traversal
- Implement adaptive bitrate
- Support H.264 and VP9 codecs

## Open Questions
- How to handle mobile network transitions?
- Fallback strategy for unsupported browsers?""",
            "tags": ["webrtc", "architecture", "implementation"]
        },
        {
            "title": "API Design Principles",
            "content": """# API Design Principles

## REST Best Practices
- Use proper HTTP status codes
- Version APIs in URL path
- Implement pagination for lists
- Use consistent naming conventions

## Authentication
- JWT tokens with refresh mechanism
- Rate limiting per API key
- OAuth2 for third-party integrations""",
            "tags": ["api", "design", "best-practices"]
        }
    ]
    
    for i, example in enumerate(examples):
        note_id = str(ulid.ULID())
        slug = example["title"].lower().replace(" ", "-")[:30]
        note_path = notes_dir / f"{slug}-{note_id}.md"
        
        post = frontmatter.Post(
            content=example["content"],
            metadata={
                "id": note_id,
                "type": "note",
                "captured": (datetime.utcnow() - timedelta(days=i)).isoformat() + "Z",
                "title": example["title"],
                "project": None,
                "tags": example["tags"],
                "links": []
            }
        )
        
        with open(note_path, 'w') as f:
            f.write(frontmatter.dumps(post))
        
        console.print(f"  Created note: {example['title']}")


def create_example_tasks(vault_path: Path):
    """Create example tasks in the vault."""
    tasks_dir = vault_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    examples = [
        {
            "title": "Review Q3 roadmap with team",
            "status": "next",
            "energy": "deep",
            "effort_min": 60,
            "tags": ["meeting", "planning"]
        },
        {
            "title": "Fix memory leak in search module",
            "status": "inbox",
            "energy": "deep",
            "effort_min": 120,
            "tags": ["bug", "performance"]
        },
        {
            "title": "Email Priya about API changes",
            "status": "next",
            "energy": "shallow",
            "effort_min": 15,
            "tags": ["communication", "api"]
        },
        {
            "title": "Update documentation for v2.0",
            "status": "waiting",
            "energy": "shallow",
            "effort_min": 45,
            "tags": ["documentation"]
        }
    ]
    
    for i, example in enumerate(examples):
        task_id = str(ulid.ULID())
        task_path = tasks_dir / f"task-{task_id}.md"
        
        post = frontmatter.Post(
            content=example["title"],
            metadata={
                "id": task_id,
                "type": "task",
                "captured": (datetime.utcnow() - timedelta(hours=i*2)).isoformat() + "Z",
                "status": example["status"],
                "title": example["title"],
                "project": None,
                "energy": example["energy"],
                "effort_min": example["effort_min"],
                "due": None,
                "tags": example["tags"],
                "evidence": {
                    "source": "text",
                    "context": None,
                    "heuristics": []
                },
                "receipts_id": None
            }
        )
        
        with open(task_path, 'w') as f:
            f.write(frontmatter.dumps(post))
        
        console.print(f"  Created task: {example['title']}")


def create_example_journal(vault_path: Path):
    """Create example journal entries."""
    today = datetime.utcnow()
    
    for days_ago in range(3):
        date = today - timedelta(days=days_ago)
        journal_dir = vault_path / "journal" / f"{date.year:04d}" / f"{date.month:02d}"
        journal_dir.mkdir(parents=True, exist_ok=True)
        
        journal_path = journal_dir / f"{date.day:02d}.md"
        
        content = f"""# Journal - {date.strftime('%Y-%m-%d')}

## 09:00:00 - Note
Started working on the search optimization. The current implementation is too slow for large vaults.

## 10:30:00 - Task
Need to profile the indexing pipeline and find bottlenecks.
*Context: VSCode main.py*

## 14:15:00 - Note
Found that the vector similarity calculation is the main bottleneck. Consider using approximate methods.

## 16:45:00 - Question
Should we implement caching at the query level or result level?
"""
        
        with open(journal_path, 'w') as f:
            f.write(content)
        
        console.print(f"  Created journal: {date.strftime('%Y-%m-%d')}")


@click.command()
@click.option("--vault", "-v", type=click.Path(), help="Vault path to initialize")
@click.option("--force", is_flag=True, help="Overwrite existing files")
def main(vault: str, force: bool):
    """Initialize Symbiote vault with example data."""
    console.print("[bold cyan]Symbiote Setup[/bold cyan]\n")
    
    # Determine vault path
    if vault:
        vault_path = Path(vault)
    else:
        vault_path = Path.cwd() / "vault"
    
    vault_path = vault_path.resolve()
    
    # Check if vault exists
    if vault_path.exists() and not force:
        console.print(f"[yellow]Vault already exists at {vault_path}[/yellow]")
        if not click.confirm("Continue anyway?"):
            return
    
    # Create vault structure
    console.print(f"Initializing vault at: {vault_path}\n")
    
    vault_path.mkdir(parents=True, exist_ok=True)
    (vault_path / ".sym" / "wal").mkdir(parents=True, exist_ok=True)
    (vault_path / ".sym" / "cache").mkdir(parents=True, exist_ok=True)
    (vault_path / "projects").mkdir(parents=True, exist_ok=True)
    (vault_path / "decisions").mkdir(parents=True, exist_ok=True)
    (vault_path / "insights").mkdir(parents=True, exist_ok=True)
    
    # Create example data
    console.print("Creating example data...")
    create_example_notes(vault_path)
    create_example_tasks(vault_path)
    create_example_journal(vault_path)
    
    console.print(f"\n[green]✓ Vault initialized successfully![/green]")
    console.print(f"\nUpdate your symbiote.yaml:")
    console.print(f"  vault_path: {vault_path}")
    console.print(f"\nThen start the daemon:")
    console.print(f"  sym daemon start")


if __name__ == "__main__":
    main()