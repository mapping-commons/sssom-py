# -*- coding: utf-8 -*-

"""Type hints for SSSOM."""

import uuid
from collections import ChainMap
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, TYPE_CHECKING, Union

import yaml

import curies
from curies import Converter

from sssom.constants import DEFAULT_LICENSE, SSSOM_URI_PREFIX, CURIE_MAP, PREFIX_MAP_MODE_METADATA_ONLY, \
    PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY, PREFIX_MAP_MODE_MERGED

if TYPE_CHECKING:
    from .context import HINT, get_converter

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
    def prefix_map(self):
        """Get the bimap."""
        return self.converter.bimap

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


def _get_prefix_map_and_metadata(
    prefix_map: "HINT" = None, meta: Optional[MetadataType] = None
) -> Metadata:
    from sssom.context import _get_built_in_prefix_map, ensure_converter

    if meta is None:
        meta = get_default_metadata()
    converter = curies.chain(
        [
            _get_built_in_prefix_map(),
            Converter.from_prefix_map(meta.pop(CURIE_MAP, {})),
            ensure_converter(prefix_map, use_defaults=False),
        ]
    )
    return Metadata(converter=converter, metadata=meta)


def get_metadata_and_prefix_map(
    metadata_path: Union[None, str, Path] = None, prefix_map_mode: Optional[str] = None
) -> Metadata:
    """
    Load SSSOM metadata from a file, and then augments it with default prefixes.

    :param metadata_path: The metadata file in YAML format
    :param prefix_map_mode: one of metadata_only, sssom_default_only, merged
    :return: a prefix map dictionary and a metadata object dictionary
    """
    if metadata_path is None:
        return Metadata.default()

    with Path(metadata_path).resolve().open() as file:
        metadata = yaml.safe_load(file)

    metadata = dict(ChainMap(metadata, get_default_metadata()))

    converter = Converter.from_prefix_map(metadata.pop(CURIE_MAP, {}))
    converter = _merge_converter(converter, prefix_map_mode=prefix_map_mode)

    return Metadata(converter=converter, metadata=metadata)


def _merge_converter(converter: Converter, prefix_map_mode: str = None) -> Converter:
    """Merge the metadata's converter with the default converter."""
    if prefix_map_mode is None or prefix_map_mode == PREFIX_MAP_MODE_METADATA_ONLY:
        return converter
    if prefix_map_mode == PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY:
        return get_converter()
    if prefix_map_mode == PREFIX_MAP_MODE_MERGED:
        return curies.chain([converter, get_converter()])
    raise ValueError(f"Invalid prefix map mode: {prefix_map_mode}")
