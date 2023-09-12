# -*- coding: utf-8 -*-

"""Type hints for SSSOM."""

import uuid
from typing import Any, Dict, NamedTuple

from sssom.constants import DEFAULT_LICENSE, SSSOM_URI_PREFIX

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

    prefix_map: PrefixMap
    metadata: MetadataType

    @classmethod
    def default(cls):
        """Get default metadata."""
        from .context import get_extended_prefix_map

        return cls(
            prefix_map=get_extended_prefix_map(),
            metadata=get_default_metadata(),
        )


def generate_mapping_set_id() -> str:
    """Getnerate a mapping set ID."""
    return f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"


def get_default_metadata() -> MetadataType:
    """Get default metadata."""
    return {
        "mapping_set_id": generate_mapping_set_id(),
        "license": DEFAULT_LICENSE,
    }
