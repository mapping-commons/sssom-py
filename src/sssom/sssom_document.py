"""Additional SSSOM object models."""

import warnings
from dataclasses import dataclass

from curies import Converter
from sssom_schema import MappingSet

from .typehints import PrefixMap

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
    def prefix_map(self) -> PrefixMap:
        """Get a prefix map."""
        warnings.warn(
            "use MappingSetDocument.converter.bimap directly", DeprecationWarning, stacklevel=2
        )
        return dict(self.converter.bimap)
