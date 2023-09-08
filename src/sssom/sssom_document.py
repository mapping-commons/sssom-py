"""Additional SSSOM object models."""

from dataclasses import dataclass

from curies import Converter
from sssom_schema import MappingSet

from sssom.context import DEFAULT_LICENSE, DEFAULT_MAPPING_SET_ID

__all__ = [
    "MappingSetDocument",
]


@dataclass()
class MappingSetDocument:
    """Represents a single SSSOM document."""

    mapping_set: MappingSet
    converter: Converter

    @classmethod
    def empty(cls, converter: Converter) -> "MappingSetDocument":
        """Get an empty mapping set document with the given prefix map."""
        mapping_set = MappingSet(mapping_set_id=DEFAULT_MAPPING_SET_ID, license=DEFAULT_LICENSE)
        return cls(converter=converter, mapping_set=mapping_set)
