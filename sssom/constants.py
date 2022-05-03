"""Constants."""

import os
import pathlib

from linkml_runtime.utils.schema_as_dict import schema_as_dict
from linkml_runtime.utils.schemaview import SchemaView

HERE = pathlib.Path(__file__).parent.resolve()
SCHEMA_YAML = os.path.join(HERE, "sssom.yaml")

SCHEMA_VIEW = SchemaView(SCHEMA_YAML)
SCHEMA_DICT = schema_as_dict(SCHEMA_VIEW.schema)
MAPPING_SLOTS = SCHEMA_DICT["classes"]["mapping"]["slots"]
MAPPING_SET_SLOTS = SCHEMA_DICT["classes"]["mapping set"]["slots"]

OWL_EQUIV_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

DEFAULT_MAPPING_PROPERTIES = [
    "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
    "http://www.w3.org/2004/02/skos/core#exactMatch",
    "http://www.w3.org/2004/02/skos/core#broadMatch",
    "http://www.w3.org/2004/02/skos/core#closeMatch",
    "http://www.w3.org/2004/02/skos/core#narrowMatch",
    "http://www.w3.org/2004/02/skos/core#relatedMatch",
    OWL_EQUIV_CLASS,
]
