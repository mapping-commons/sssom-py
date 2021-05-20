import os
import json

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


#cwd = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONTEXT_PATH = 'sssom.context.jsonld'

def get_jsonld_context():
    with pkg_resources.open_text(__package__, DEFAULT_CONTEXT_PATH) as json_file:
        contxt = json.load(json_file)
    return contxt

def get_default_metadata():
    contxt = get_jsonld_context()
    curie_map = {}
    meta = {}
    for key in contxt["@context"]:
        v = contxt["@context"][key]
        if isinstance(v, str):
            curie_map[key] = v
    return meta, curie_map
