# -*- coding: utf-8 -*-

"""Type hints for SSSOM."""

from typing import Any, Dict, NamedTuple

from curies import Converter

__all__ = [
    "PrefixMap",
    "MetadataType",
    "Metadata",
]

PrefixMap = Dict[str, str]

#: TODO replace this with something more specific
MetadataType = Dict[str, Any]


class Metadata(NamedTuple):
    """A pair of a prefix map and associated metadata."""

    converter: Converter
    metadata: MetadataType

    @property
    def prefix_map(self) -> PrefixMap:
        """Get the prefix bimap out of the converter."""
        return get_bimap(self.converter)


def get_bimap(converter: Converter) -> PrefixMap:
    """Get a bidirectional prefix map."""
    return {record.prefix: record.uri_prefix for record in converter.records}
