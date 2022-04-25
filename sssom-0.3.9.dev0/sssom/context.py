"""Utilities for loading JSON-LD contexts."""

import json
import logging
import uuid
from typing import Optional

from .external_context import sssom_external_context
from .internal_context import sssom_context
from .typehints import Metadata, MetadataType, PrefixMap

# HERE = pathlib.Path(__file__).parent.resolve()
# DEFAULT_CONTEXT_PATH = HERE / "sssom.context.jsonld"
# EXTERNAL_CONTEXT_PATH = HERE / "sssom.external.context.jsonld"

SSSOM_URI_PREFIX = "http://w3id.org/sssom/"
SSSOM_BUILT_IN_PREFIXES = ["sssom", "owl", "rdf", "rdfs", "skos"]
DEFAULT_MAPPING_SET_ID = f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"
DEFAULT_LICENSE = f"{SSSOM_URI_PREFIX}license/unspecified"


def get_jsonld_context():
    """Get JSON-LD form of sssom_context variable from auto-generated 'internal_context.py' file.

    [Auto generated from sssom.yaml by jsonldcontextgen.py]

    :return: JSON-LD context
    """
    return json.loads(sssom_context, strict=False)


def get_external_jsonld_context():
    """Get JSON-LD form of sssom_external_context variable from auto-generated 'external_context.py' file.

    :return: JSON-LD context
    """
    return json.loads(sssom_external_context, strict=False)


def get_built_in_prefix_map() -> PrefixMap:
    """Get built-in prefix map from the sssom_context variable in the auto-generated 'internal_context.py' file.

    [Auto generated from sssom.yaml by jsonldcontextgen.py]

    :return: Prefix map
    """
    contxt = get_jsonld_context()
    prefix_map = {}
    for key in contxt["@context"]:
        if key in SSSOM_BUILT_IN_PREFIXES:
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
    :return: A prefix map
    """
    builtinmap = get_built_in_prefix_map()
    if not prefix_map:
        prefix_map = builtinmap
    else:
        for k, v in builtinmap.items():
            if k not in prefix_map:
                prefix_map[k] = v
            elif builtinmap[k] != prefix_map[k]:
                logging.warning(
                    f"Built-in prefix {k} is specified ({prefix_map[k]}) but differs from default ({builtinmap[k]})"
                )
    return prefix_map


def get_default_metadata() -> Metadata:
    """Get @context property value from the sssom_context variable in the auto-generated 'internal_context.py' file.

    [Auto generated from sssom.yaml by jsonldcontextgen.py]

    :return: Metadata
    """
    contxt = get_jsonld_context()
    contxt_external = get_external_jsonld_context()
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
    for key in contxt_external["@context"]:
        v = contxt_external["@context"][key]
        if isinstance(v, str):
            if key not in prefix_map:
                prefix_map[key] = v
            else:
                if prefix_map[key] != v:
                    logging.warning(
                        f"{key} is already in prefix map ({prefix_map[key]}, but with a different value than {v}"
                    )

    metadata = Metadata(prefix_map=prefix_map, metadata=metadata_dict)
    metadata = set_default_mapping_set_id(metadata)
    metadata = set_default_license(metadata)
    return metadata


def set_default_mapping_set_id(meta: Metadata) -> Metadata:
    """Provide a default mapping_set_id if absent in the MappingSetDataFrame.

    :param meta: Metadata without mapping_set_id
    :return: Metadata with a default mapping_set_id
    """
    if ("mapping_set_id" not in meta.metadata) or (
        meta.metadata["mapping_set_id"] is None
    ):
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
