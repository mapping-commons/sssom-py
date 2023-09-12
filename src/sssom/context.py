"""Utilities for loading JSON-LD contexts."""

from functools import lru_cache
from typing import Mapping, Union

import curies
from curies import Converter
from rdflib.namespace import is_ncname

from .constants import (
    EXTENDED_PREFIX_MAP,
    PREFIX_MAP_MODE_MERGED,
    PREFIX_MAP_MODE_METADATA_ONLY,
    PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY,
)

ConverterHint = Union[Mapping[str, str], None, Converter]


@lru_cache(1)
def get_converter() -> Converter:
    """Get prefix map from bioregistry (obo.epm.json)."""
    converter = Converter.from_extended_prefix_map(EXTENDED_PREFIX_MAP)
    records = []
    for record in converter.records:
        if not is_ncname(record.prefix):
            continue
        record.prefix_synonyms = [s for s in record.prefix_synonyms if is_ncname(s)]
        records.append(record)
    return Converter(records)


def ensure_converter(converter: ConverterHint = None) -> Converter:
    """Ensure that a converter with default content is available.

    :param converter: A custom prefix map
    :return: A prefix map
    """
    if converter is None:
        return get_converter()
    if isinstance(converter, Converter):
        converter = converter
    else:
        converter = Converter.from_prefix_map(converter)
    return curies.chain([converter, get_converter()])


def merge_converter(converter: Converter, prefix_map_mode: str = None) -> Converter:
    """Merge the metadata's converter with the default converter."""
    if prefix_map_mode is None or prefix_map_mode == PREFIX_MAP_MODE_METADATA_ONLY:
        return converter
    if prefix_map_mode == PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY:
        return get_converter()
    if prefix_map_mode == PREFIX_MAP_MODE_MERGED:
        return curies.chain([converter, get_converter()])
    raise ValueError(f"Invalid prefix map mode: {prefix_map_mode}")
