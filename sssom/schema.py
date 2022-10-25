"""SSSOM Schema import."""
import pkg_resources
from linkml_runtime.utils.schema_as_dict import schema_as_dict
from linkml_runtime.utils.schemaview import SchemaView

SCHEMA_YAML = pkg_resources.resource_filename(
    "sssom_schema", "schema/sssom_schema.yaml"
)
SCHEMA_VIEW: SchemaView = SchemaView(SCHEMA_YAML)
# SCHEMA_VIEW = package_schemaview("sssom_schema")
SCHEMA_DICT = schema_as_dict(SCHEMA_VIEW.schema)
MAPPING_SLOTS = SCHEMA_DICT["classes"]["mapping"]["slots"]
MAPPING_SET_SLOTS = SCHEMA_DICT["classes"]["mapping set"]["slots"]

MULTIVALUED_SLOTS = [
    c for c in SCHEMA_VIEW.all_slots() if SCHEMA_VIEW.get_slot(c).multivalued
]

ENTITY_REFERENCE = "EntityReference"

ENTITY_REFERENCE_SLOTS = [
    c
    for c in SCHEMA_VIEW.all_slots()
    if SCHEMA_VIEW.get_slot(c).range == ENTITY_REFERENCE
]
