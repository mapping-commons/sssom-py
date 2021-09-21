from typing import Any, Dict, NamedTuple

__all__ = [
    "PrefixMap",
    "MetadataType",
    "Metadata",
    "ModeLiteral",
]

PrefixMap = Dict[str, str]

#: TODO replace this with something more specific
MetadataType = Dict[str, Any]


class Metadata(NamedTuple):
    prefix_map: PrefixMap
    metadata: MetadataType
