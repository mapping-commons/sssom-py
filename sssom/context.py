import json
import logging

from .external_context import sssom_external_context
from .internal_context import sssom_context

# HERE = pathlib.Path(__file__).parent.resolve()
# DEFAULT_CONTEXT_PATH = HERE / "sssom.context.jsonld"
# EXTERNAL_CONTEXT_PATH = HERE / "sssom.external.context.jsonld"
SSSOM_BUILT_IN_PREFIXES = ["sssom", "owl", "rdf", "rdfs", "skos"]


def get_jsonld_context():
    return json.loads(sssom_context, strict=False)


def get_external_jsonld_context():
    return json.loads(sssom_external_context, strict=False)


def get_built_in_prefix_map():
    contxt = get_jsonld_context()
    curie_map = {}
    for key in contxt["@context"]:
        if key in SSSOM_BUILT_IN_PREFIXES:
            v = contxt["@context"][key]
            if isinstance(v, str):
                curie_map[key] = v
    return curie_map


def add_built_in_prefixes_to_prefix_map(prefixmap):
    builtinmap = get_built_in_prefix_map()
    if not prefixmap:
        prefixmap = builtinmap
    else:
        for k, v in builtinmap.items():
            if k not in prefixmap:
                prefixmap[k] = v
            elif builtinmap[k] != prefixmap[k]:
                logging.warning(
                    f"Built-in prefix {k} is specified ({prefixmap[k]}) but differs from default ({builtinmap[k]})"
                )
    return prefixmap


def get_default_metadata():
    contxt = get_jsonld_context()
    contxt_external = get_external_jsonld_context()
    curie_map = {}
    meta = {}
    for key in contxt["@context"]:
        v = contxt["@context"][key]
        if isinstance(v, str):
            curie_map[key] = v
        elif isinstance(v, dict):
            if "@id" in v and "@prefix" in v:
                if v["@prefix"]:
                    curie_map[key] = v["@id"]
    for key in contxt_external["@context"]:
        v = contxt_external["@context"][key]
        if isinstance(v, str):
            if key not in curie_map:
                curie_map[key] = v
            else:
                if curie_map[key] != v:
                    logging.warning(
                        f"{key} is already in curie map ({curie_map[key]}, but with a different value than {v}"
                    )
    return meta, curie_map
