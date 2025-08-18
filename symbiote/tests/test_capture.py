"""Tests for capture service."""

import asyncio
import tempfile
from pathlib import Path
import pytest
import ulid

from symbiote.daemon.config import Config
from symbiote.daemon.capture import CaptureService, CaptureEntry


@pytest.fixture
def temp_vault():
    """Create a temporary vault for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir) / "test_vault"
        vault_path.mkdir()
        (vault_path / ".sym" / "wal").mkdir(parents=True)
        yield vault_path


@pytest.fixture
def test_config(temp_vault):
    """Create test configuration."""
    return Config(vault_path=temp_vault)


@pytest.mark.asyncio
async def test_capture_text(test_config):
    """Test basic text capture."""
    service = CaptureService(test_config)
    await service.initialize()
    
    # Capture a note
    entry = await service.capture(
        text="Test note content",
        type="note",
        source="text"
    )
    
    assert entry.id
    assert entry.type == "note"
    assert entry.text == "Test note content"
    assert entry.source == "text"
    
    # Check WAL was written
    wal_files = list((test_config.vault_path / ".sym" / "wal").glob("*.log"))
    assert len(wal_files) == 1
    
    await service.close()


@pytest.mark.asyncio
async def test_capture_task(test_config):
    """Test task capture."""
    service = CaptureService(test_config)
    await service.initialize()
    
    # Capture a task
    entry = await service.capture(
        text="Review pull request",
        type="task",
        source="text",
        context="GitHub PR #123"
    )
    
    assert entry.type == "task"
    assert entry.context == "GitHub PR #123"
    
    # Give time for materialization
    await asyncio.sleep(0.1)
    
    # Check task file was created
    task_files = list((test_config.vault_path / "tasks").glob("*.md"))
    assert len(task_files) == 1
    
    await service.close()


@pytest.mark.asyncio
async def test_wal_replay(test_config):
    """Test WAL replay on restart."""
    # First capture
    service1 = CaptureService(test_config)
    await service1.initialize()
    
    entry = await service1.capture(
        text="Test for replay",
        type="note"
    )
    original_id = entry.id
    
    # Close without materializing
    await service1.close()
    
    # Start new service and replay
    service2 = CaptureService(test_config)
    await service2.initialize()
    
    # Give time for replay
    await asyncio.sleep(0.1)
    
    # Check note was materialized from WAL
    note_files = list((test_config.vault_path / "notes").glob(f"*-{original_id}.md"))
    assert len(note_files) == 1
    
    await service2.close()


def test_capture_entry_creation():
    """Test CaptureEntry dataclass."""
    entry = CaptureEntry(
        id="",
        type="task",
        text="Test task"
    )
    
    # Should auto-generate ID
    assert entry.id
    assert len(entry.id) == 26  # ULID length
    
    # Should have timestamp
    assert entry.captured_at is not None