# -*- coding: utf-8 -*-

"""Type hints for SSSOM."""

import uuid
from typing import Any, Dict, NamedTuple

from curies import Converter

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

    converter: Converter
    metadata: MetadataType

    @classmethod
    def default(cls):
        """Get default metadata."""
        from .context import get_converter

        return cls(
            converter=get_converter(),
            metadata=get_default_metadata(),
        )


def generate_mapping_set_id() -> str:
    """Generate a mapping set ID."""
    return f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"


def get_default_metadata() -> MetadataType:
    """Get default metadata."""
    return {
        "mapping_set_id": generate_mapping_set_id(),
        "license": DEFAULT_LICENSE,
    }
