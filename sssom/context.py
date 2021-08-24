import json
import logging

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    # noinspection PyUnresolvedReferences
    import importlib_resources as pkg_resources

# cwd = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONTEXT_PATH = "sssom.context.jsonld"
EXTERNAL_CONTEXT_PATH = "sssom.external.context.jsonld"


def get_jsonld_context():
    with pkg_resources.open_text(__package__, DEFAULT_CONTEXT_PATH) as json_file:
        contxt = json.load(json_file)
    return contxt


def get_external_jsonld_context():
    with pkg_resources.open_text(__package__, EXTERNAL_CONTEXT_PATH) as json_file:
        contxt = json.load(json_file)
    return contxt


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
