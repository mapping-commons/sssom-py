"""Constants."""

import os
import pathlib

from linkml_runtime.utils.schema_as_dict import schema_as_dict
from linkml_runtime.utils.schemaview import SchemaView

HERE = pathlib.Path(__file__).parent.resolve()
SCHEMA_YAML = os.path.join(HERE, "sssom.yaml")

SCHEMA_VIEW = SchemaView(SCHEMA_YAML)
SCHEMA_DICT = schema_as_dict(SCHEMA_VIEW.schema)
