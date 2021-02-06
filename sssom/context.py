import os
import json

cwd = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONTEXT_PATH = f'{cwd}/../schema/sssom.context.jsonld'


def get_jsonld_context():
    with open(DEFAULT_CONTEXT_PATH) as json_file:
        contxt = json.load(json_file)
    return contxt
