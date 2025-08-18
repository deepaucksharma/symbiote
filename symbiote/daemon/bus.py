"""Async event bus for inter-module communication."""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import weakref
from loguru import logger


@dataclass
class Event:
    """Base event class."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    correlation_id: Optional[str] = None


class EventBus:
    """
    Async pub/sub event bus for in-process communication.
    
    Event types follow pattern: category.action
    Examples: capture.request, capture.written, search.request, indexer.update
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[weakref.ref]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._stats = defaultdict(int)
    
    def subscribe(self, event_pattern: str, handler: Callable[[Event], Any]) -> None:
        """
        Subscribe to events matching pattern.
        Pattern can use wildcards: 'capture.*' matches all capture events.
        """
        # Use weak references to avoid memory leaks
        handler_ref = weakref.ref(handler)
        self._subscribers[event_pattern].append(handler_ref)
        logger.debug(f"Subscribed handler to pattern: {event_pattern}")
    
    def unsubscribe(self, event_pattern: str, handler: Callable[[Event], Any]) -> None:
        """Unsubscribe handler from event pattern."""
        self._subscribers[event_pattern] = [
            ref for ref in self._subscribers[event_pattern]
            if ref() is not None and ref() != handler
        ]
    
    async def emit(self, event: Event) -> None:
        """Emit an event to the bus."""
        if self._event_queue.full():
            logger.warning(f"Event queue full, dropping event: {event.type}")
            self._stats['dropped'] += 1
            return
        
        await self._event_queue.put(event)
        self._stats['emitted'] += 1
        logger.debug(f"Emitted event: {event.type}")
    
    def emit_nowait(self, event: Event) -> bool:
        """
        Emit an event without waiting (non-async).
        Returns True if successful, False if queue is full.
        """
        try:
            self._event_queue.put_nowait(event)
            self._stats['emitted'] += 1
            logger.debug(f"Emitted event (nowait): {event.type}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.type}")
            self._stats['dropped'] += 1
            return False
    
    async def start(self) -> None:
        """Start the event processor."""
        if self._running:
            logger.warning("Event bus already running")
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event processor."""
        self._running = False
        if self._processor_task:
            await self._processor_task
        logger.info("Event bus stopped")
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                # Wait for event with timeout to allow checking _running flag
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                # Find matching subscribers
                handlers = []
                for pattern, refs in self._subscribers.items():
                    if self._matches_pattern(event.type, pattern):
                        # Clean up dead weak references
                        valid_refs = []
                        for ref in refs:
                            handler = ref()
                            if handler is not None:
                                handlers.append(handler)
                                valid_refs.append(ref)
                        self._subscribers[pattern] = valid_refs
                
                # Call handlers concurrently
                if handlers:
                    tasks = []
                    for handler in handlers:
                        if asyncio.iscoroutinefunction(handler):
                            tasks.append(asyncio.create_task(handler(event)))
                        else:
                            # Wrap sync handlers
                            tasks.append(
                                asyncio.create_task(
                                    asyncio.to_thread(handler, event)
                                )
                            )
                    
                    # Wait for all handlers with timeout
                    results = await asyncio.gather(
                        *tasks, 
                        return_exceptions=True
                    )
                    
                    # Log any handler errors
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Handler error for event {event.type}: {result}"
                            )
                            self._stats['handler_errors'] += 1
                
                self._stats['processed'] += 1
                
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self._stats['processing_errors'] += 1
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches subscription pattern."""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix + ".")
        return event_type == pattern
    
    def get_stats(self) -> Dict[str, int]:
        """Get event bus statistics."""
        return dict(self._stats)
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats.clear()


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus