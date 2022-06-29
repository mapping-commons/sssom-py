"""Constants."""

import pathlib
from enum import Enum

import pkg_resources
from linkml_runtime.utils.schema_as_dict import schema_as_dict
from linkml_runtime.utils.schemaview import SchemaView

# from linkml_runtime.utils.introspection import package_schemaview

HERE = pathlib.Path(__file__).parent.resolve()

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
ENTITY_REFERENCE = "EntityReference"

MULTIVALUED_SLOTS = [
    c for c in SCHEMA_VIEW.all_slots() if SCHEMA_VIEW.get_slot(c).multivalued
]
ENTITY_REFERENCE_SLOTS = [
    c
    for c in SCHEMA_VIEW.all_slots()
    if SCHEMA_VIEW.get_slot(c).range == ENTITY_REFERENCE
]

# Slot Constants
MIRROR_FROM = "mirror_from"
REGISTRY_CONFIDENCE = "registry_confidence"
LAST_UPDATED = "last_updated"
LOCAL_NAME = "local_name"
MAPPING_SET_REFERENCES = "mapping_set_references"
MAPPING_REGISTRY_ID = "mapping_registry_id"
IMPORTS = "imports"
DOCUMENTATION = "documentation"
HOMEPAGE = "homepage"
MAPPINGS = "mappings"
SUBJECT_ID = "subject_id"
SUBJECT_LABEL = "subject_label"
SUBJECT_CATEGORY = "subject_category"
SUBJECT_TYPE = "subject_type"
PREDICATE_ID = "predicate_id"
PREDICATE_MODIFIER = "predicate_modifier"
PREDICATE_MODIFIER_NOT = "Not"
PREDICATE_LABEL = "predicate_label"
PREDICATE_TYPE = "predicate_type"
OBJECT_ID = "object_id"
OBJECT_LABEL = "object_label"
OBJECT_CATEGORY = "object_category"
MAPPING_JUSTIFICATION = "mapping_justification"
MAPPING_JUSTIFICATION_UNSPECIFIED = "Unspecified"
OBJECT_TYPE = "object_type"
MAPPING_SET_ID = "mapping_set_id"
MAPPING_SET_VERSION = "mapping_set_version"
MAPPING_SET_GROUP = "mapping_set_group"
MAPPING_SET_DESCRIPTION = "mapping_set_description"
CREATOR_ID = "creator_id"
CREATOR_LABEL = "creator_label"
AUTHOR_ID = "author_id"
AUTHOR_LABEL = "author_label"
REVIEWER_ID = "reviewer_id"
REVIEWER_LABEL = "reviewer_label"
LICENSE = "license"
SUBJECT_SOURCE = "subject_source"
SUBJECT_SOURCE_VERSION = "subject_source_version"
OBJECT_SOURCE = "object_source"
OBJECT_SOURCE_VERSION = "object_source_version"
MAPPING_PROVIDER = "mapping_provider"
MAPPING_SET_SOURCE = "mapping_set_source"
MAPPING_SOURCE = "mapping_source"
MAPPING_CARDINALITY = "mapping_cardinality"
MAPPING_TOOL = "mapping_tool"
MAPPING_TOOL_VERSION = "mapping_tool_version"
MAPPING_DATE = "mapping_date"
PBLICATION_DATE = "publication_date"
CONFIDENCE = "confidence"
SUBJECT_MATCH_FIELD = "subject_match_field"
OBJECT_MATCH_FIELD = "object_match_field"
MATCH_STRING = "match_string"
SUBJECT_PREPROCESSING = "subject_preprocessing"
OBJECT_PREPROCESSING = "object_preprocessing"
SEMANTIC_SIMILARITY_SCORE = "semantic_similarity_score"
SEMANTIC_SIMILARITY_MEASURE = "semantic_similarity_measure"
SEE_ALSO = "see_also"
OTHER = "other"
COMMENT = "comment"

CURIE_MAP = "curie_map"
SUBJECT_SOURCE_ID = "subject_source_id"
OBJECT_SOURCE_ID = "object_source_id"


class SEMAPV(Enum):
    """SEMAPV Enum containing different mapping_justification."""

    LexicalMatching = "semapv:LexicalMatching"
    LogicalReasoning = "semapv:LogicalReasoning"
    CompositeMatching = "semapv:CompositeMatching"
    UnspecifiedMatching = "semapv:UnspecifiedMatching"
    SemanticSimilarityThresholdMatching = "semapv:SemanticSimilarityThresholdMatching"
    LexicalSimilarityThresholdMatching = "semapv:LexicalSimilarityThresholdMatching"
    MappingChaining = "semapv:MappingChaining"
    MappingReview = "semapv:MappingReview"
    ManualMappingCuration = "semapv:ManualMappingCuration"


class SchemaValidationType(str, Enum):
    """Schema validation types."""

    JsonSchema = "JsonSchema"
    Shacl = "Shacl"
    PrefixMapCompleteness = "PrefixMapCompleteness"


DEFAULT_VALIDATION_TYPES = [
    SchemaValidationType.JsonSchema,
    SchemaValidationType.PrefixMapCompleteness,
]
