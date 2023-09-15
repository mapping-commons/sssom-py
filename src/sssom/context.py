"""Utilities for loading JSON-LD contexts."""

import json
from functools import lru_cache
from typing import Union

import curies
import pkg_resources
from curies import Converter
from rdflib.namespace import is_ncname

from .constants import EXTENDED_PREFIX_MAP
from .typehints import PrefixMap

SSSOM_BUILT_IN_PREFIXES = ("sssom", "owl", "rdf", "rdfs", "skos", "semapv")
SSSOM_CONTEXT = pkg_resources.resource_filename(
    "sssom_schema", "context/sssom_schema.context.jsonld"
)


@lru_cache(1)
def get_converter() -> Converter:
    """Get a converter."""
    return curies.chain([_get_built_in_prefix_map(), _get_default_converter()])


@lru_cache(1)
def _get_default_converter() -> Converter:
    converter = Converter.from_extended_prefix_map(EXTENDED_PREFIX_MAP)
    records = []
    for record in converter.records:
        if not is_ncname(record.prefix):
            continue
        record.prefix_synonyms = [s for s in record.prefix_synonyms if is_ncname(s)]
        records.append(record)
    return Converter(records)


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


HINT = Union[None, PrefixMap, Converter]


def ensure_converter(prefix_map: HINT = None) -> Converter:
    """Ensure a converter is available."""
    if not prefix_map:
        return get_converter()
    if not isinstance(prefix_map, Converter):
        prefix_map = Converter.from_prefix_map(prefix_map)
    return curies.chain([_get_built_in_prefix_map(), prefix_map])
