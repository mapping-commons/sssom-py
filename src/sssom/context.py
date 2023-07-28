"""Utilities for loading JSON-LD contexts."""

import json
import logging
import uuid
from typing import Any, Mapping, Optional, Union

import curies
import pkg_resources
from curies import Converter

from sssom.constants import EXTENDED_PREFIX_MAP

from .typehints import Metadata, PrefixMap

SSSOM_URI_PREFIX = "https://w3id.org/sssom/"
SSSOM_BUILT_IN_PREFIXES = ("sssom", "owl", "rdf", "rdfs", "skos", "semapv")
DEFAULT_MAPPING_SET_ID = f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"
DEFAULT_LICENSE = f"{SSSOM_URI_PREFIX}license/unspecified"
SSSOM_CONTEXT = pkg_resources.resource_filename(
    "sssom_schema", "context/sssom_schema.context.jsonld"
)


def _get_jsonld_context():
    """Get JSON-LD form of sssom_context variable from auto-generated 'internal_context.py' file."""
    with open(SSSOM_CONTEXT, "r") as c:
        context = json.load(c, strict=False)
    return context


def get_internal_converter() -> Converter:
    """Get a converter from the SSSOM internal context."""
    context = _get_jsonld_context()
    prefix_map = {}
    for key in context["@context"]:
        v = context["@context"][key]
        if isinstance(v, str):
            prefix_map[key] = v
        elif isinstance(v, dict):
            if "@id" in v and "@prefix" in v:
                if v["@prefix"]:
                    prefix_map[key] = v["@id"]
    del prefix_map["@vocab"]
    return Converter.from_prefix_map(prefix_map)


def get_external_converter() -> Converter:
    """Get prefix map from bioregistry (obo.epm.json)."""
    return Converter.from_extended_prefix_map(EXTENDED_PREFIX_MAP)


def get_built_in_converter() -> Converter:
    """Get built-in prefix map from the sssom_context variable in the auto-generated 'internal_context.py' file."""
    context = _get_jsonld_context()
    prefix_map = {}
    for key in context["@context"]:
        if key in list(SSSOM_BUILT_IN_PREFIXES):
            v = context["@context"][key]
            if isinstance(v, str):
                prefix_map[key] = v
    return Converter.from_prefix_map(prefix_map)


def add_built_in_prefixes_to_prefix_map(
    prefix_map: Union[Converter, PrefixMap, None] = None,
) -> Converter:
    """Add built-in prefix map from the sssom_context variable in the auto-generated 'internal_context.py' file.

    :param prefix_map: A custom prefix map
    :raises ValueError: If there is a prefix map mismatch.
    :return: A prefix map
    """
    if prefix_map is None:
        return get_built_in_converter()
    if isinstance(prefix_map, Converter):
        converter = prefix_map
    else:
        converter = Converter.from_prefix_map(prefix_map)
    return curies.chain([converter, get_built_in_converter()])


def get_default_metadata() -> Metadata:
    """Get @context property value from the sssom_context variable in the auto-generated 'internal_context.py' file.

    :return: Metadata
    """
    converter_internal = get_internal_converter()
    converter_external = get_external_converter()
    converter = curies.chain([converter_internal, converter_external])
    return Metadata(
        converter=converter,
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


def prepare_context(
    prefix_map: Optional[PrefixMap] = None,
) -> Mapping[str, Any]:
    """Prepare a JSON-LD context from a prefix map."""
    context = _get_jsonld_context()
    if prefix_map is None:
        prefix_map = get_default_metadata().prefix_map

    for k, v in prefix_map.items():
        if isinstance(v, str):
            if k not in context["@context"]:
                context["@context"][k] = v
            else:
                if context["@context"][k] != v:
                    logging.info(
                        f"{k} namespace is already in the context, ({context['@context'][k]}, "
                        f"but with a different value than {v}. Overwriting!"
                    )
                    context["@context"][k] = v
    return context
