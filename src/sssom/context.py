"""Utilities for loading JSON-LD contexts."""

import json
import uuid
from typing import Optional

import pkg_resources
from curies import Converter

from .constants import EXTENDED_PREFIX_MAP, SSSOM_URI_PREFIX
from .typehints import Metadata, PrefixMap

SSSOM_BUILT_IN_PREFIXES = ("sssom", "owl", "rdf", "rdfs", "skos", "semapv")
DEFAULT_MAPPING_SET_ID = f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"
SSSOM_CONTEXT = pkg_resources.resource_filename(
    "sssom_schema", "context/sssom_schema.context.jsonld"
)


def get_jsonld_context():
    """Get JSON-LD form of sssom_context variable from auto-generated 'internal_context.py' file.

    [Auto generated from sssom.yaml by jsonldcontextgen.py]

    :return: JSON-LD context
    """
    with open(SSSOM_CONTEXT, "r") as c:
        context = json.load(c, strict=False)

    return context


def get_converter() -> Converter:
    """Get a converter."""
    return Converter.from_extended_prefix_map(EXTENDED_PREFIX_MAP)


def get_extended_prefix_map():
    """Get prefix map from bioregistry (obo.epm.json).

    :return: Prefix map.
    """
    converter = get_converter()
    return {record.prefix: record.uri_prefix for record in converter.records}


def get_built_in_prefix_map() -> PrefixMap:
    """Get built-in prefix map from the sssom_context variable in the auto-generated 'internal_context.py' file.

    [Auto generated from sssom.yaml by jsonldcontextgen.py]

    :return: Prefix map
    """
    contxt = get_jsonld_context()
    prefix_map = {}
    for key in contxt["@context"]:
        if key in list(SSSOM_BUILT_IN_PREFIXES):
            v = contxt["@context"][key]
            if isinstance(v, str):
                prefix_map[key] = v
    return prefix_map


def add_built_in_prefixes_to_prefix_map(
    prefix_map: Optional[PrefixMap] = None,
) -> PrefixMap:
    """Add built-in prefix map from the sssom_context variable in the auto-generated 'internal_context.py' file.

    [Auto generated from sssom.yaml by jsonldcontextgen.py]

    :param prefix_map: A custom prefix map
    :raises ValueError: If there is a prefix map mismatch.
    :return: A prefix map
    """
    builtinmap = get_built_in_prefix_map()
    if not prefix_map:
        prefix_map = builtinmap
    else:
        for k, v in builtinmap.items():
            if k not in prefix_map and v not in prefix_map.values():
                prefix_map[k] = v
            elif builtinmap[k] != prefix_map[k]:
                raise ValueError(
                    f"Built-in prefix {k} is specified ({prefix_map[k]}) but differs from default ({builtinmap[k]})"
                )
    return prefix_map


def get_default_metadata() -> Metadata:
    """Get the default metadata."""
    return Metadata.default()
