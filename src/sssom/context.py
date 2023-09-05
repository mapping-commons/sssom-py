"""Utilities for loading JSON-LD contexts."""

import json
import logging
import uuid
from typing import Optional

import pkg_resources
from curies import Converter

from sssom.constants import EXTENDED_PREFIX_MAP

from .typehints import Metadata, MetadataType, PrefixMap

# HERE = pathlib.Path(__file__).parent.resolve()
# DEFAULT_CONTEXT_PATH = HERE / "sssom.context.jsonld"
# EXTERNAL_CONTEXT_PATH = HERE / "obo.epm.json"

SSSOM_URI_PREFIX = "https://w3id.org/sssom/"
SSSOM_BUILT_IN_PREFIXES = ("sssom", "owl", "rdf", "rdfs", "skos", "semapv")
DEFAULT_MAPPING_SET_ID = f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"
DEFAULT_LICENSE = f"{SSSOM_URI_PREFIX}license/unspecified"
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


def get_extended_prefix_map():
    """Get prefix map from bioregistry (obo.epm.json).

    :return: Prefix map.
    """
    converter = Converter.from_extended_prefix_map(EXTENDED_PREFIX_MAP)
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
    """Get @context property value from the sssom_context variable in the auto-generated 'internal_context.py' file.

    [Auto generated from sssom.yaml by jsonldcontextgen.py]

    :return: Metadata
    """
    contxt = get_jsonld_context()
    contxt_external = get_extended_prefix_map()
    prefix_map = {}
    metadata_dict: MetadataType = {}
    for key in contxt["@context"]:
        v = contxt["@context"][key]
        if isinstance(v, str):
            prefix_map[key] = v
        elif isinstance(v, dict):
            if "@id" in v and "@prefix" in v:
                if v["@prefix"]:
                    prefix_map[key] = v["@id"]
    del prefix_map["@vocab"]

    prefix_map.update({(k, v) for k, v in contxt_external.items() if k not in prefix_map})
    _raise_on_invalid_prefix_map(prefix_map)

    metadata = Metadata(prefix_map=prefix_map, metadata=metadata_dict)
    metadata.metadata["mapping_set_id"] = DEFAULT_MAPPING_SET_ID
    metadata.metadata["license"] = DEFAULT_LICENSE
    return metadata


def _raise_on_invalid_prefix_map(prefix_map):
    """Raise an exception if the prefix map is not bijective.

    This uses :meth:`curies.Converter.from_prefix_map` to try and load a
    prefix map. If there are any duplicate values (i.e., it is _not_ bijective)
    then it throws a value error.
    """
    Converter.from_prefix_map(prefix_map)


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
