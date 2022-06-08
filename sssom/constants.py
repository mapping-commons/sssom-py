"""Constants."""

import pathlib

import pkg_resources

from linkml_runtime.utils.schemaview import SchemaView
# from linkml_runtime.utils.introspection import package_schemaview
from linkml_runtime.utils.schema_as_dict import schema_as_dict
import pydantic

HERE = pathlib.Path(__file__).parent.resolve()
# SCHEMA_YAML = os.path.join(HERE, "sssom.yaml")

SCHEMA_YAML = pkg_resources.resource_filename(
    "sssom_schema", "schema/sssom_schema.yaml"
)
SCHEMA_VIEW = SchemaView(SCHEMA_YAML)
# SCHEMA_VIEW = package_schemaview("sssom_schema")
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


PREFIX_MAP_MODE_METADATA_ONLY = "metadata_only"
PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY = "sssom_default_only"
PREFIX_MAP_MODE_MERGED = "merged"
PREFIX_MAP_MODES = [
    PREFIX_MAP_MODE_METADATA_ONLY,
    PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY,
    PREFIX_MAP_MODE_MERGED,
]

multivalued_slots = []
for slot in SCHEMA_DICT['slots']:
    if 'multivalued' in SCHEMA_DICT['slots'][slot].keys() and SCHEMA_DICT['slots'][slot]["multivalued"]:
        multivalued_slots.append(slot)