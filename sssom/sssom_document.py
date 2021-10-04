"""Additional SSSOM object models."""

from dataclasses import dataclass

from .sssom_datamodel import MappingSet
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

    prefix_map: PrefixMap
    """
    Mappings between ID prefixes and URI Bases, used to map CURIEs to URIs.
    Note that the prefix map is not part of the core SSSOM model, hence it belongs here in the document
    object
    """

    @classmethod
    def empty(cls, prefix_map: PrefixMap) -> "MappingSetDocument":
        """Get an empty mapping set document with the given prefix map."""
        return cls(prefix_map=prefix_map, mapping_set=MappingSet(mappings=[]))
