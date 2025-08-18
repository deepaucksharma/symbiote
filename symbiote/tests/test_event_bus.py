"""Tests for event bus."""

import asyncio
import pytest

from symbiote.daemon.bus import EventBus, Event


@pytest.mark.asyncio
async def test_event_emit_and_subscribe():
    """Test basic pub/sub functionality."""
    bus = EventBus()
    await bus.start()
    
    received_events = []
    
    async def handler(event: Event):
        received_events.append(event)
    
    # Subscribe to capture events
    bus.subscribe("capture.*", handler)
    
    # Emit event
    await bus.emit(Event(
        type="capture.written",
        data={"id": "test123"}
    ))
    
    # Give time for processing
    await asyncio.sleep(0.1)
    
    assert len(received_events) == 1
    assert received_events[0].type == "capture.written"
    assert received_events[0].data["id"] == "test123"
    
    await bus.stop()


@pytest.mark.asyncio
async def test_wildcard_subscription():
    """Test wildcard pattern matching."""
    bus = EventBus()
    await bus.start()
    
    all_events = []
    capture_events = []
    
    async def all_handler(event: Event):
        all_events.append(event)
    
    async def capture_handler(event: Event):
        capture_events.append(event)
    
    # Subscribe with different patterns
    bus.subscribe("*", all_handler)
    bus.subscribe("capture.*", capture_handler)
    
    # Emit various events
    await bus.emit(Event(type="capture.written", data={}))
    await bus.emit(Event(type="search.request", data={}))
    await bus.emit(Event(type="capture.failed", data={}))
    
    await asyncio.sleep(0.1)
    
    assert len(all_events) == 3
    assert len(capture_events) == 2
    
    await bus.stop()


@pytest.mark.asyncio
async def test_event_queue_full():
    """Test behavior when event queue is full."""
    bus = EventBus()
    bus._event_queue = asyncio.Queue(maxsize=2)
    await bus.start()
    
    # Fill the queue
    await bus.emit(Event(type="test.1", data={}))
    await bus.emit(Event(type="test.2", data={}))
    
    # This should be dropped
    await bus.emit(Event(type="test.3", data={}))
    
    stats = bus.get_stats()
    assert stats['dropped'] == 1
    
    await bus.stop()


def test_pattern_matching():
    """Test pattern matching logic."""
    bus = EventBus()
    
    # Exact match
    assert bus._matches_pattern("capture.written", "capture.written")
    assert not bus._matches_pattern("capture.written", "capture.failed")
    
    # Wildcard
    assert bus._matches_pattern("capture.written", "capture.*")
    assert bus._matches_pattern("search.request", "search.*")
    assert not bus._matches_pattern("capture.written", "search.*")
    
    # Global wildcard
    assert bus._matches_pattern("anything", "*")
    assert bus._matches_pattern("capture.written", "*")