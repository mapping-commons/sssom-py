"""Utilities for loading JSON-LD contexts."""

import logging
import uuid
from functools import lru_cache
from typing import Union

import curies
from curies import Converter
from rdflib.namespace import is_ncname

from .constants import (
    EXTENDED_PREFIX_MAP,
    PREFIX_MAP_MODE_MERGED,
    PREFIX_MAP_MODE_METADATA_ONLY,
    PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY,
)
from .typehints import Metadata, PrefixMap

SSSOM_URI_PREFIX = "https://w3id.org/sssom/"
SSSOM_BUILT_IN_PREFIXES = ("sssom", "owl", "rdf", "rdfs", "skos", "semapv")
DEFAULT_MAPPING_SET_ID = f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"
DEFAULT_LICENSE = f"{SSSOM_URI_PREFIX}license/unspecified"

ConverterHint = Union[PrefixMap, None, Converter]


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
    """Add built-in prefix map from the sssom_context variable in the auto-generated 'internal_context.py' file.

    :param converter: A custom prefix map
    :raises ValueError: If there is a prefix map mismatch.
    :return: A prefix map
    """
    if converter is None:
        return get_converter()
    if isinstance(converter, Converter):
        converter = converter
    else:
        converter = Converter.from_prefix_map(converter)
    return curies.chain([converter, get_converter()])


def get_default_metadata() -> Metadata:
    """Get @context property value from the sssom_context variable in the auto-generated 'internal_context.py' file.

    :return: Metadata
    """
    return Metadata(
        converter=get_converter(),
        metadata={
            "mapping_set_id": DEFAULT_MAPPING_SET_ID,
            "license": DEFAULT_LICENSE,
        },
    )


def set_default_mapping_set_id(meta: Metadata) -> Metadata:
    """Provide a default mapping_set_id if absent in the MappingSetDataFrame.

    :param meta: Metadata without mapping_set_id
    :return: Metadata with a default mapping_set_id
    """
    if ("mapping_set_id" not in meta.metadata) or (meta.metadata["mapping_set_id"] is None):
        meta.metadata["mapping_set_id"] = DEFAULT_MAPPING_SET_ID
    return meta


def set_default_license(meta: Metadata) -> Metadata:
    """Provide a default license if absent in the MappingSetDataFrame.

    :param meta: Metadata without license
    :return: Metadata with a default license
    """
    if ("license" not in meta.metadata) or (meta.metadata["license"] is None):
        meta.metadata["license"] = DEFAULT_LICENSE
        logging.warning(f"No License provided, using {DEFAULT_LICENSE}")
    return meta


def merge_converter(metadata: Metadata, prefix_map_mode: str = None) -> Converter:
    """Merge the metadata's converter with the default converter."""
    if prefix_map_mode is None or prefix_map_mode == PREFIX_MAP_MODE_METADATA_ONLY:
        return metadata.converter
    if prefix_map_mode == PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY:
        return get_converter()
    if prefix_map_mode == PREFIX_MAP_MODE_MERGED:
        return curies.chain([metadata.converter, get_converter()])
    raise ValueError(f"Invalid prefix map mode: {prefix_map_mode}")
