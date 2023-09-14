"""Utilities for loading JSON-LD contexts."""

import json
from functools import lru_cache
from typing import Optional

import curies
import pkg_resources
from curies import Converter

from .constants import EXTENDED_PREFIX_MAP
from .typehints import PrefixMap

SSSOM_BUILT_IN_PREFIXES = ("sssom", "owl", "rdf", "rdfs", "skos", "semapv")
SSSOM_CONTEXT = pkg_resources.resource_filename(
    "sssom_schema", "context/sssom_schema.context.jsonld"
)


@lru_cache(1)
def get_converter() -> Converter:
    """Get a converter."""
    return Converter.from_extended_prefix_map(EXTENDED_PREFIX_MAP)


@lru_cache(1)
def get_extended_prefix_map():
    """Get prefix map from bioregistry (obo.epm.json).

    :return: Prefix map.
    """
    converter = get_converter()
    return {record.prefix: record.uri_prefix for record in converter.records}


@lru_cache(1)
def _get_built_in_prefix_map() -> Converter:
    """Get URI prefixes for built-in prefixes."""
    with open(SSSOM_CONTEXT) as file:
        context = json.load(file, strict=False)
    prefix_map = {
        prefix: uri_prefix
        for prefix, uri_prefix in context["@context"].items()
        if prefix in SSSOM_BUILT_IN_PREFIXES and isinstance(uri_prefix, str)
    }
    return Converter.from_prefix_map(prefix_map)


def add_built_in_prefixes_to_prefix_map(
    prefix_map: Optional[PrefixMap] = None,
) -> PrefixMap:
    """Add built-in prefix map from the sssom_context variable in the auto-generated 'internal_context.py' file.

    :param prefix_map: A custom prefix map
    :raises ValueError: If there is a prefix map mismatch.
    :return: A prefix map
    """
    default_converter = _get_built_in_prefix_map()
    if not prefix_map:
        return dict(default_converter.bimap)
    new_converter = curies.chain([default_converter, Converter.from_prefix_map(prefix_map)])
    return dict(new_converter.bimap)
