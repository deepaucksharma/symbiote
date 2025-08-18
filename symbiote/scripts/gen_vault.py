#!/usr/bin/env python3
"""
Synthetic vault generator for testing and benchmarking.
Generates realistic notes and tasks with varied tags, projects, and timestamps.
"""

import random
import json
from pathlib import Path
from datetime import datetime, timedelta
import click
import ulid
import frontmatter
from faker import Faker
from loguru import logger

fake = Faker()


class VaultGenerator:
    """Generate synthetic vault data for testing."""
    
    # Realistic project names
    PROJECTS = [
        "Remote Support Feature",
        "Q3 Strategy Planning",
        "API Redesign",
        "Performance Optimization",
        "Mobile App MVP",
        "Data Pipeline",
        "Customer Portal",
        "Security Audit",
        "Documentation Update",
        "Team Expansion"
    ]
    
    # Common tags
    TAGS = [
        "strategy", "planning", "implementation", "bug", "feature",
        "documentation", "meeting", "review", "urgent", "blocked",
        "research", "design", "testing", "deployment", "maintenance",
        "api", "frontend", "backend", "database", "infrastructure",
        "performance", "security", "ux", "analytics", "ml"
    ]
    
    # Task templates
    TASK_TEMPLATES = [
        "Review {thing} for {project}",
        "Fix {issue} in {component}",
        "Implement {feature} for {user_type}",
        "Update {doc_type} documentation",
        "Schedule meeting about {topic}",
        "Research {technology} for {use_case}",
        "Test {feature} on {platform}",
        "Deploy {component} to {environment}",
        "Optimize {metric} in {system}",
        "Document {api_endpoint} endpoint"
    ]
    
    # Note content generators
    NOTE_TOPICS = [
        ("Architecture Decision", "technical"),
        ("Meeting Notes", "meeting"),
        ("Research Findings", "research"),
        ("Bug Investigation", "debugging"),
        ("Feature Specification", "spec"),
        ("Performance Analysis", "performance"),
        ("Security Review", "security"),
        ("User Feedback", "feedback"),
        ("Weekly Update", "status"),
        ("Retrospective", "process")
    ]
    
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.vault_path / "journal").mkdir(exist_ok=True)
        (self.vault_path / "tasks").mkdir(exist_ok=True)
        (self.vault_path / "notes").mkdir(exist_ok=True)
        (self.vault_path / "projects").mkdir(exist_ok=True)
        (self.vault_path / ".sym" / "wal").mkdir(parents=True, exist_ok=True)
    
    def generate_task(self, created_at: datetime) -> dict:
        """Generate a realistic task."""
        template = random.choice(self.TASK_TEMPLATES)
        
        # Fill template with fake data
        title = template.format(
            thing=fake.catch_phrase(),
            project=random.choice(self.PROJECTS),
            issue=f"#{fake.random_int(100, 999)}",
            component=fake.word(),
            feature=fake.bs(),
            user_type=fake.job(),
            doc_type=random.choice(["API", "User", "Admin", "Developer"]),
            topic=fake.bs(),
            technology=fake.word(),
            use_case=fake.catch_phrase(),
            platform=random.choice(["iOS", "Android", "Web", "Desktop"]),
            environment=random.choice(["staging", "production", "dev"]),
            metric=random.choice(["latency", "throughput", "memory", "CPU"]),
            system=fake.word(),
            api_endpoint=f"/{fake.word()}/{fake.word()}"
        )
        
        # Generate metadata
        task_id = str(ulid.ULID())
        status = random.choice(["inbox", "next", "waiting", "done", "cancelled"])
        project = random.choice(self.PROJECTS + [None, None])  # 2/3 chance of project
        energy = random.choice(["deep", "shallow", "recovery", "unknown"])
        effort_min = random.choice([15, 30, 45, 60, 90, 120])
        tags = random.sample(self.TAGS, k=random.randint(1, 4))
        
        # Add due date for some tasks
        due = None
        if random.random() < 0.3:  # 30% have due dates
            due = (created_at + timedelta(days=random.randint(1, 30))).date().isoformat()
        
        return {
            "id": task_id,
            "type": "task",
            "captured": created_at.isoformat() + "Z",
            "status": status,
            "title": title,
            "project": project,
            "energy": energy,
            "effort_min": effort_min,
            "due": due,
            "tags": tags,
            "evidence": {
                "source": random.choice(["text", "voice"]),
                "context": f"{fake.word()}:{fake.file_name()}",
                "heuristics": []
            },
            "receipts_id": None,
            "content": title + "\n\n" + fake.paragraph()
        }
    
    def generate_note(self, created_at: datetime) -> dict:
        """Generate a realistic note."""
        topic, topic_type = random.choice(self.NOTE_TOPICS)
        
        # Generate title
        if topic_type == "meeting":
            title = f"{topic}: {fake.company()} - {fake.date()}"
        elif topic_type == "technical":
            title = f"{topic}: {fake.catch_phrase()}"
        else:
            title = f"{topic} - {fake.bs()}"
        
        # Generate content
        content = f"# {title}\n\n"
        
        # Add sections
        num_sections = random.randint(2, 5)
        for i in range(num_sections):
            section_title = fake.sentence(nb_words=4)[:-1]
            content += f"## {section_title}\n\n"
            
            # Add paragraphs or bullet points
            if random.random() < 0.5:
                # Paragraphs
                for _ in range(random.randint(1, 3)):
                    content += fake.paragraph() + "\n\n"
            else:
                # Bullet points
                for _ in range(random.randint(3, 7)):
                    content += f"- {fake.sentence()}\n"
                content += "\n"
        
        # Add code block for technical notes
        if topic_type in ["technical", "debugging", "performance"]:
            content += "```python\n"
            content += f"def {fake.word()}():\n"
            content += f"    # {fake.sentence()}\n"
            content += f"    return {fake.word()}\n"
            content += "```\n\n"
        
        # Generate metadata
        note_id = str(ulid.ULID())
        project = random.choice(self.PROJECTS + [None, None])
        tags = random.sample(self.TAGS, k=random.randint(2, 5))
        
        # Add some wikilinks
        links = []
        if random.random() < 0.7:  # 70% have links
            num_links = random.randint(1, 3)
            for _ in range(num_links):
                links.append(f"[[{fake.sentence(nb_words=3)[:-1]}]]")
        
        return {
            "id": note_id,
            "type": "note",
            "captured": created_at.isoformat() + "Z",
            "title": title,
            "project": project,
            "tags": tags,
            "links": links,
            "content": content
        }
    
    def generate_journal_entry(self, date: datetime) -> str:
        """Generate journal content for a specific date."""
        content = f"# Journal - {date.strftime('%Y-%m-%d')}\n\n"
        
        # Add entries throughout the day
        num_entries = random.randint(3, 8)
        entry_times = sorted([
            date.replace(
                hour=random.randint(8, 20),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            for _ in range(num_entries)
        ])
        
        for entry_time in entry_times:
            entry_type = random.choice(["Note", "Task", "Question", "Idea"])
            timestamp = entry_time.strftime("%H:%M:%S")
            
            content += f"\n## {timestamp} - {entry_type}\n"
            content += fake.paragraph() + "\n"
            
            if random.random() < 0.3:
                content += f"*Context: {fake.word()}:{fake.file_name()}*\n"
        
        return content
    
    def generate(
        self,
        num_notes: int,
        num_tasks: int,
        days_back: int = 90,
        create_journal: bool = True
    ) -> dict:
        """Generate a complete synthetic vault."""
        stats = {
            "notes": 0,
            "tasks": 0,
            "journal_days": 0,
            "total_size_kb": 0
        }
        
        # Generate timestamps spread over the time period
        now = datetime.utcnow()
        start_date = now - timedelta(days=days_back)
        
        # Generate tasks
        logger.info(f"Generating {num_tasks} tasks...")
        for i in range(num_tasks):
            # Random timestamp within the period
            created_at = start_date + timedelta(
                seconds=random.randint(0, days_back * 86400)
            )
            
            task = self.generate_task(created_at)
            task_path = self.vault_path / "tasks" / f"task-{task['id']}.md"
            
            # Write task file
            post = frontmatter.Post(
                content=task.pop("content"),
                metadata=task
            )
            
            with open(task_path, 'w') as f:
                f.write(frontmatter.dumps(post))
            
            stats["tasks"] += 1
            stats["total_size_kb"] += len(frontmatter.dumps(post)) / 1024
            
            if (i + 1) % 100 == 0:
                logger.debug(f"  Generated {i + 1} tasks")
        
        # Generate notes
        logger.info(f"Generating {num_notes} notes...")
        for i in range(num_notes):
            created_at = start_date + timedelta(
                seconds=random.randint(0, days_back * 86400)
            )
            
            note = self.generate_note(created_at)
            slug = note["title"].lower().replace(" ", "-")[:30]
            note_path = self.vault_path / "notes" / f"{slug}-{note['id']}.md"
            
            # Write note file
            post = frontmatter.Post(
                content=note.pop("content"),
                metadata=note
            )
            
            with open(note_path, 'w') as f:
                f.write(frontmatter.dumps(post))
            
            stats["notes"] += 1
            stats["total_size_kb"] += len(frontmatter.dumps(post)) / 1024
            
            if (i + 1) % 100 == 0:
                logger.debug(f"  Generated {i + 1} notes")
        
        # Generate journal entries
        if create_journal:
            logger.info(f"Generating journal entries for {days_back} days...")
            current_date = start_date
            while current_date <= now:
                # Skip some days randomly
                if random.random() < 0.8:  # 80% of days have entries
                    journal_dir = (
                        self.vault_path / "journal" /
                        f"{current_date.year:04d}" /
                        f"{current_date.month:02d}"
                    )
                    journal_dir.mkdir(parents=True, exist_ok=True)
                    
                    journal_path = journal_dir / f"{current_date.day:02d}.md"
                    content = self.generate_journal_entry(current_date)
                    
                    with open(journal_path, 'w') as f:
                        f.write(content)
                    
                    stats["journal_days"] += 1
                    stats["total_size_kb"] += len(content) / 1024
                
                current_date += timedelta(days=1)
        
        # Create project files
        for project in self.PROJECTS[:5]:  # Create 5 project files
            project_slug = project.lower().replace(" ", "-")
            project_path = self.vault_path / "projects" / project_slug / "project.md"
            project_path.parent.mkdir(parents=True, exist_ok=True)
            
            post = frontmatter.Post(
                content=f"# {project}\n\n{fake.paragraph()}\n\n## Goals\n{fake.paragraph()}",
                metadata={
                    "id": f"proj_{ulid.ULID()}",
                    "type": "project",
                    "title": project,
                    "status": random.choice(["active", "paused", "done"]),
                    "owner": "me",
                    "tags": random.sample(self.TAGS, k=3)
                }
            )
            
            with open(project_path, 'w') as f:
                f.write(frontmatter.dumps(post))
        
        return stats


@click.command()
@click.option("--notes", default=100, help="Number of notes to generate")
@click.option("--tasks", default=50, help="Number of tasks to generate")
@click.option("--days", default=90, help="Days of history to generate")
@click.option("--out", type=click.Path(), required=True, help="Output vault path")
@click.option("--no-journal", is_flag=True, help="Skip journal generation")
@click.option("--seed", type=int, help="Random seed for reproducibility")
def main(notes: int, tasks: int, days: int, out: str, no_journal: bool, seed: int):
    """Generate a synthetic vault for testing and benchmarking."""
    if seed:
        random.seed(seed)
        Faker.seed(seed)
    
    vault_path = Path(out)
    logger.info(f"Generating synthetic vault at {vault_path}")
    logger.info(f"Configuration: {notes} notes, {tasks} tasks, {days} days of history")
    
    generator = VaultGenerator(vault_path)
    stats = generator.generate(
        num_notes=notes,
        num_tasks=tasks,
        days_back=days,
        create_journal=not no_journal
    )
    
    logger.success(f"Vault generated successfully!")
    logger.info(f"Statistics:")
    logger.info(f"  Notes: {stats['notes']}")
    logger.info(f"  Tasks: {stats['tasks']}")
    logger.info(f"  Journal days: {stats['journal_days']}")
    logger.info(f"  Total size: {stats['total_size_kb']:.1f} KB")
    
    # Write stats file
    stats_path = vault_path / ".sym" / "vault_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()