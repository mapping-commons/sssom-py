"""Additional SSSOM object models."""

from dataclasses import dataclass
from typing import Dict

from curies import Converter
from sssom_schema import MappingSet

__all__ = [
    "MappingSetDocument",
]


@dataclass()
class MappingSetDocument:
    """
    Represents a single SSSOM document.

    A document is simply a holder for a MappingSet object plus a CURIE map
    """

    mapping_set: MappingSet
    """
    The main part of the document: a set of mappings plus metadata
    """

    converter: Converter

    @property
    def prefix_map(self) -> Dict[str, str]:
        """Get a prefix map."""
        return dict(self.converter.bimap)
