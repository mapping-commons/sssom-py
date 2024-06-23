"""Constants."""

import pathlib
import uuid
from enum import Enum
from functools import cached_property, lru_cache
from typing import Any, Dict, List, Literal, Set

import importlib_resources
import yaml
from linkml_runtime.utils.schema_as_dict import schema_as_dict
from linkml_runtime.utils.schemaview import SchemaView

HERE = pathlib.Path(__file__).parent.resolve()

SCHEMA_YAML = importlib_resources.files("sssom_schema").joinpath("schema/sssom_schema.yaml")
EXTENDED_PREFIX_MAP = HERE / "obo.epm.json"

OWL_EQUIV_CLASS_URI = "http://www.w3.org/2002/07/owl#equivalentClass"
RDFS_SUBCLASS_OF_URI = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDF_TYPE_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SSSOM_SUPERCLASS_OF_URI = "http://w3id.org/sssom/superClassOf"
SKOS_EXACT_MATCH_URI = "http://www.w3.org/2004/02/skos/core#exactMatch"
SKOS_CLOSE_MATCH_URI = "http://www.w3.org/2004/02/skos/core#closeMatch"
SKOS_BROAD_MATCH_URI = "http://www.w3.org/2004/02/skos/core#broadMatch"
SKOS_NARROW_MATCH_URI = "http://www.w3.org/2004/02/skos/core#narrowMatch"
OBO_HAS_DB_XREF_URI = "http://www.geneontology.org/formats/oboInOwl#hasDbXref"
SKOS_RELATED_MATCH_URI = "http://www.w3.org/2004/02/skos/core#relatedMatch"

DEFAULT_MAPPING_PROPERTIES = [
    SKOS_EXACT_MATCH_URI,
    SKOS_CLOSE_MATCH_URI,
    SKOS_BROAD_MATCH_URI,
    SKOS_NARROW_MATCH_URI,
    OBO_HAS_DB_XREF_URI,
    SKOS_RELATED_MATCH_URI,
    OWL_EQUIV_CLASS_URI,
]

UNKNOWN_IRI = "http://w3id.org/sssom/unknown_prefix/"
MergeMode = Literal["metadata_only", "sssom_default_only", "merged"]
PREFIX_MAP_MODE_METADATA_ONLY: MergeMode = "metadata_only"
PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY: MergeMode = "sssom_default_only"
PREFIX_MAP_MODE_MERGED: MergeMode = "merged"
ENTITY_REFERENCE = "EntityReference"

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
MAPPING_JUSTIFICATION_UNSPECIFIED = "semapv:UnspecifiedMatching"
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

# PREDICATES
OWL_EQUIVALENT_CLASS = "owl:equivalentClass"
OWL_EQUIVALENT_PROPERTY = "owl:equivalentProperty"
OWL_DIFFERENT_FROM = "owl:differentFrom"
RDFS_SUBCLASS_OF = "rdfs:subClassOf"
RDFS_SUBPROPERTY_OF = "rdfs:subPropertyOf"
OWL_SAME_AS = "owl:sameAs"
SKOS_EXACT_MATCH = "skos:exactMatch"
SKOS_CLOSE_MATCH = "skos:closeMatch"
SKOS_BROAD_MATCH = "skos:broadMatch"
SKOS_NARROW_MATCH = "skos:narrowMatch"
OBO_HAS_DB_XREF = "oboInOwl:hasDbXref"
SKOS_RELATED_MATCH = "skos:relatedMatch"
CROSS_SPECIES_EXACT_MATCH = "semapv:crossSpeciesExactMatch"
CROSS_SPECIES_NARROW_MATCH = "semapv:crossSpeciesNarrowMatch"
CROSS_SPECIES_BROAD_MATCH = "semapv:crossSpeciesBroadMatch"
RDF_SEE_ALSO = "rdfs:seeAlso"
RDF_TYPE = "rdf:type"
SSSOM_SUPERCLASS_OF = "sssom:superClassOf"

PREDICATE_LIST = [
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
    RDFS_SUBCLASS_OF,
    SSSOM_SUPERCLASS_OF,
    RDFS_SUBPROPERTY_OF,
    OWL_SAME_AS,
    SKOS_EXACT_MATCH,
    SKOS_CLOSE_MATCH,
    SKOS_BROAD_MATCH,
    SKOS_NARROW_MATCH,
    OBO_HAS_DB_XREF,
    SKOS_RELATED_MATCH,
    RDF_SEE_ALSO,
]

with open(HERE / "inverse_map.yaml", "r") as im:
    inverse_map = yaml.safe_load(im)

PREDICATE_INVERT_DICTIONARY = inverse_map["inverse_predicate_map"]

COLUMN_INVERT_DICTIONARY = {
    SUBJECT_ID: OBJECT_ID,
    SUBJECT_LABEL: OBJECT_LABEL,
    SUBJECT_CATEGORY: OBJECT_CATEGORY,
    SUBJECT_MATCH_FIELD: OBJECT_MATCH_FIELD,
    SUBJECT_SOURCE: OBJECT_SOURCE,
    SUBJECT_PREPROCESSING: OBJECT_PREPROCESSING,
    SUBJECT_SOURCE_VERSION: OBJECT_SOURCE_VERSION,
    OBJECT_ID: SUBJECT_ID,
    OBJECT_LABEL: SUBJECT_LABEL,
    OBJECT_CATEGORY: SUBJECT_CATEGORY,
    OBJECT_MATCH_FIELD: SUBJECT_MATCH_FIELD,
    OBJECT_SOURCE: SUBJECT_SOURCE,
    OBJECT_PREPROCESSING: SUBJECT_PREPROCESSING,
    OBJECT_SOURCE_VERSION: SUBJECT_SOURCE_VERSION,
}


class SEMAPV(Enum):
    """SEMAPV Enum containing different mapping_justification.

    See also: https://mapping-commons.github.io/semantic-mapping-vocabulary/#matchingprocess
    """

    LexicalMatching = "semapv:LexicalMatching"
    LogicalReasoning = "semapv:LogicalReasoning"
    CompositeMatching = "semapv:CompositeMatching"
    UnspecifiedMatching = "semapv:UnspecifiedMatching"
    SemanticSimilarityThresholdMatching = "semapv:SemanticSimilarityThresholdMatching"
    LexicalSimilarityThresholdMatching = "semapv:LexicalSimilarityThresholdMatching"
    MappingChaining = "semapv:MappingChaining"
    MappingReview = "semapv:MappingReview"
    ManualMappingCuration = "semapv:ManualMappingCuration"
    MappingInversion = "semapv:MappingInversion"
    CrossSpeciesExactMatch = CROSS_SPECIES_EXACT_MATCH
    CrossSpeciesNarrowMatch = CROSS_SPECIES_NARROW_MATCH
    CrossSpeciesBroadMatch = CROSS_SPECIES_BROAD_MATCH


class SchemaValidationType(str, Enum):
    """Schema validation types."""

    # TODO move this class into validators.py
    JsonSchema = "JsonSchema"
    Shacl = "Shacl"
    Sparql = "Sparql"
    PrefixMapCompleteness = "PrefixMapCompleteness"
    StrictCurieFormat = "StrictCurieFormat"


DEFAULT_VALIDATION_TYPES = [
    SchemaValidationType.JsonSchema,
    SchemaValidationType.PrefixMapCompleteness,
    SchemaValidationType.StrictCurieFormat,
]


class SSSOMSchemaView(object):
    """
    SchemaView class from linkml which is instantiated when necessary.

    Reason for this: https://github.com/mapping-commons/sssom-py/issues/322
    Implemented via PR: https://github.com/mapping-commons/sssom-py/pull/323
    """

    def __new__(cls):
        """Create a instance of the SSSOM schema view if non-existent."""
        if not hasattr(cls, "instance"):
            cls.instance = super(SSSOMSchemaView, cls).__new__(cls)
        return cls.instance

    @cached_property
    def view(self) -> SchemaView:
        """Return SchemaView object."""
        return SchemaView(SCHEMA_YAML)

    @cached_property
    def dict(self) -> dict:
        """Return SchemaView as a dictionary."""
        return schema_as_dict(self.view.schema)

    @cached_property
    def mapping_slots(self) -> List[str]:
        """Return list of mapping slots."""
        return self.view.get_class("mapping").slots

    @cached_property
    def mapping_set_slots(self) -> List[str]:
        """Return list of mapping set slots."""
        return self.view.get_class("mapping set").slots

    @cached_property
    def multivalued_slots(self) -> Set[str]:
        """Return set of multivalued slots."""
        return {c for c in self.view.all_slots() if self.view.get_slot(c).multivalued}

    @cached_property
    def entity_reference_slots(self) -> Set[str]:
        """Return set of entity reference slots."""
        return {c for c in self.view.all_slots() if self.view.get_slot(c).range == ENTITY_REFERENCE}

    @cached_property
    def mapping_enum_keys(self) -> Set[str]:
        """Return a set of mapping enum keys."""
        return set(_get_sssom_schema_object().dict["enums"].keys())

    @cached_property
    def slots(self) -> Dict[str, str]:
        """Return the slots for SSSOMSchemaView object."""
        return self.dict["slots"]

    @cached_property
    def double_slots(self) -> Set[str]:
        """Return the slot names for SSSOMSchemaView object."""
        return {k for k, v in self.dict["slots"].items() if v["range"] == "double"}


@lru_cache(1)
def _get_sssom_schema_object() -> SSSOMSchemaView:
    """Get a view over the SSSOM schema."""
    sssom_sv_object = (
        SSSOMSchemaView.instance if hasattr(SSSOMSchemaView, "instance") else SSSOMSchemaView()
    )
    return sssom_sv_object


SSSOM_URI_PREFIX = "https://w3id.org/sssom/"
DEFAULT_LICENSE = f"{SSSOM_URI_PREFIX}license/unspecified"

#: The type for metadata that gets passed around in many places
MetadataType = Dict[str, Any]


def generate_mapping_set_id() -> str:
    """Generate a mapping set ID."""
    return f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}"


def get_default_metadata() -> MetadataType:
    """Get default metadata.

    :returns: A metadata dictionary containing a default
        license with value :data:`DEFAULT_LICENSE` and an
        auto-generated mapping set ID

    If you want to combine some metadata you loaded
    but ensure that there is also default metadata,
    the best tool is :class:`collections.ChainMap`.
    You can do:

    .. code-block:: python

        my_metadata: dict | None = ...

        from collections import ChainMap
        from sssom import get_default_metadata

        metadata = dict(ChainMap(
            my_metadata or {},
            get_default_metadata()
        ))
    """
    return {
        "mapping_set_id": generate_mapping_set_id(),
        "license": DEFAULT_LICENSE,
    }
