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
