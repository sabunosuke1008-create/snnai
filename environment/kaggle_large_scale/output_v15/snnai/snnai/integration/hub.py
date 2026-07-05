"""Event-driven hub for routing spikes between biological modules."""
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class SpikeEvent:
    """Common spike event format for module communication.

    Parameters
    ----------
    timestamp : int
        Simulation time step at which the spike occurred.
    source : str
        Name of the module that emitted the spike.
    neuron_id : int
        Index of the emitting neuron within the source module.
    strength : float
        Spike strength (default 1.0).
    """

    timestamp: int
    source: str
    neuron_id: int
    strength: float = 1.0


class Hub:
    """Simple publish/subscribe event bus for spike events.

    Modules subscribe to events from specific sources. When a module
    publishes events, the hub delivers them to all matching subscribers.
    """

    def __init__(self):
        self.subscribers: dict[str, list[Callable[[SpikeEvent], None]]] = {}
        self.events: list[SpikeEvent] = []

    def subscribe(self, source: str, callback: Callable[[SpikeEvent], None]):
        """Register a callback for events from ``source``."""
        self.subscribers.setdefault(source, []).append(callback)

    def publish(self, event: SpikeEvent):
        """Publish a spike event to all subscribers."""
        self.events.append(event)
        for callback in self.subscribers.get(event.source, []):
            callback(event)

    def reset(self):
        """Clear stored events and subscribers."""
        self.events.clear()
        self.subscribers.clear()
