"""Event hooks and callbacks for RAG pipeline."""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    """Types of events in the RAG pipeline."""
    FILE_DISCOVERED = "file_discovered"
    FILE_PROCESSED = "file_processed"
    FILE_FAILED = "file_failed"
    SEGMENT_CREATED = "segment_created"
    VECTOR_GENERATED = "vector_generated"
    VECTOR_STORED = "vector_stored"
    BATCH_START = "batch_start"
    BATCH_COMPLETE = "batch_complete"
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    ERROR = "error"


@dataclass
class Event:
    """Represents an event in the pipeline."""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[Event], None]


class EventHookManager:
    """Manages event hooks and callbacks."""

    def __init__(self):
        """Initialize hook manager."""
        self._handlers: Dict[EventType, List[EventHandler]] = {}

    def register(self, event_type: EventType, handler: EventHandler) -> None:
        """Register an event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unregister(self, event_type: EventType, handler: EventHandler) -> None:
        """Unregister an event handler."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        if event.type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    # Emit error event
                    error_event = Event(
                        type=EventType.ERROR,
                        data={'original_event': event, 'error': str(e)}
                    )
                    self._emit_error(error_event)

    def _emit_error(self, event: Event) -> None:
        """Emit error event safely."""
        if EventType.ERROR in self._handlers:
            for handler in self._handlers[EventType.ERROR]:
                try:
                    handler(event)
                except Exception:
                    pass  # Don't cascade errors


# Predefined useful handlers

def logging_handler(event: Event) -> None:
    """Log events."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Event: {event.type.value} - {event.data}")


def progress_bar_handler(total: int) -> EventHandler:
    """Create a progress bar handler."""
    from collections import defaultdict
    counter = defaultdict(int)

    def handler(event: Event) -> None:
        if event.type == EventType.FILE_PROCESSED:
            counter['processed'] += 1
            pct = int(100 * counter['processed'] / total)
            print(f"\rProgress: {counter['processed']}/{total} ({pct}%)", end='')

    return handler


def statistics_collector() -> tuple[EventHandler, Callable[[], Dict[str, Any]]]:
    """Create a statistics collection handler."""
    stats = {
        'files_processed': 0,
        'files_failed': 0,
        'segments_created': 0,
        'vectors_generated': 0,
        'errors': [],
    }

    def handler(event: Event) -> None:
        if event.type == EventType.FILE_PROCESSED:
            stats['files_processed'] += 1
        elif event.type == EventType.FILE_FAILED:
            stats['files_failed'] += 1
        elif event.type == EventType.SEGMENT_CREATED:
            stats['segments_created'] += 1
        elif event.type == EventType.VECTOR_GENERATED:
            stats['vectors_generated'] += 1
        elif event.type == EventType.ERROR:
            stats['errors'].append(event.data)

    def get_stats() -> Dict[str, Any]:
        return dict(stats)

    return handler, get_stats
