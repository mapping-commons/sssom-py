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


def ensure_converter(prefix_map: HINT = None, *, use_defaults: bool = True) -> Converter:
    """Ensure a converter is available.

    :param prefix_map: One of the following:

        1. An empty dictionary or ``None``. This results in using the default
           extended prefix map (currently based on a variant of the Bioregistry)
           if ``use_defaults`` is set to true, otherwise just the builtin prefix
           map including the prefixes in :data:`SSSOM_BUILT_IN_PREFIXES`
        2. A non-empty dictionary representing a prefix map. This is loaded as a
           converter with :meth:`Converter.from_prefix_map`. It is chained
           behind the builtin prefix map to ensure none of the
           :data:`SSSOM_BUILT_IN_PREFIXES` are overwritten with non-default values
        3. A pre-instantiated :class:`curies.Converter`. Similarly to a prefix
           map passed into this function, this is chained behind the builtin prefix
           map
    :param use_defaults: If an empty dictionary or None is passed to this function,
        this parameter chooses if the extended prefix map (currently based on a
        variant of the Bioregistry) gets loaded.
    :returns: A re-usable converter
    """
    if not prefix_map:
        if use_defaults:
            return get_converter()
        else:
            return _get_built_in_prefix_map()
    if not isinstance(prefix_map, Converter):
        prefix_map = Converter.from_prefix_map(prefix_map)
    return curies.chain([_get_built_in_prefix_map(), prefix_map])
