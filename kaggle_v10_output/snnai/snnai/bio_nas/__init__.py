"""Bio-inspired Neural Architecture Search (Bio-NAS)."""
from .evolution_search import EvolutionSearch
from .search_space import Architecture, hub_architecture, parallel_architecture, serial_architecture

__all__ = [
    "Architecture",
    "serial_architecture",
    "hub_architecture",
    "parallel_architecture",
    "EvolutionSearch",
]
