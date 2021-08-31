# Auto generated from sssom.yaml by pythongen.py version: 0.9.0
# Generation date: 2021-08-30 21:03
# Schema: sssom
#
# id: http://w3id.org/sssom/schema/
# description: Datamodel for Simple Standard for Sharing Ontology Mappings (SSSOM)
# license: https://creativecommons.org/publicdomain/zero/1.0/

import dataclasses
import re
import sys
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Union

from jsonasobj2 import JsonObj, as_dict
from linkml_runtime.linkml_model.meta import (
    EnumDefinition,
    PermissibleValue,
    PvFormulaOptions,
)
from linkml_runtime.linkml_model.types import Double, String
from linkml_runtime.utils.curienamespace import CurieNamespace
from linkml_runtime.utils.dataclass_extensions_376 import (
    dataclasses_init_fn_with_kwargs,
)
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from linkml_runtime.utils.formatutils import camelcase, sfx, underscore
from linkml_runtime.utils.metamodelcore import bnode, empty_dict, empty_list
from linkml_runtime.utils.slot import Slot
from linkml_runtime.utils.yamlutils import (
    YAMLRoot,
    extended_float,
    extended_int,
    extended_str,
)
from rdflib import Namespace, URIRef

metamodel_version = "1.7.0"

# Overwrite dataclasses _init_fn to add **kwargs in __init__
dataclasses._init_fn = dataclasses_init_fn_with_kwargs

# Namespaces
ORPHANET = CurieNamespace("Orphanet", "http://www.orpha.net/ORDO/Orphanet_")
DC = CurieNamespace("dc", "http://purl.org/dc/terms/")
DCE = CurieNamespace("dce", "http://purl.org/dc/elements/1.1/")
DCTERMS = CurieNamespace("dcterms", "http://purl.org/dc/terms/")
LINKML = CurieNamespace("linkml", "https://w3id.org/linkml/")
NEURONAME = CurieNamespace(
    "neuroname", "http://braininfo.rprc.washington.edu/centraldirectory.aspx?ID="
)
OBOINOWL = CurieNamespace("oboInOwl", "http://www.geneontology.org/formats/oboInOwl#")
OIO = CurieNamespace("oio", "http://www.geneontology.org/formats/oboInOwl#")
OWL = CurieNamespace("owl", "http://www.w3.org/2002/07/owl#")
RDF = CurieNamespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = CurieNamespace("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
SSSOM = CurieNamespace("sssom", "http://w3id.org/sssom/")
DEFAULT_ = SSSOM


# Types

# Class references
class EntityId(extended_str):
    pass


@dataclass
class MappingSet(YAMLRoot):
    """
    Represents a set of mappings
    """

    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = SSSOM.MappingSet
    class_class_curie: ClassVar[str] = "sssom:MappingSet"
    class_name: ClassVar[str] = "mapping set"
    class_model_uri: ClassVar[URIRef] = SSSOM.MappingSet

    mappings: Optional[
        Union[Union[dict, "Mapping"], List[Union[dict, "Mapping"]]]
    ] = empty_list()
    mapping_set_id: Optional[Union[str, EntityId]] = None
    mapping_set_version: Optional[str] = None
    mapping_set_description: Optional[str] = None
    creator_id: Optional[Union[str, EntityId]] = None
    creator_label: Optional[str] = None
    license: Optional[str] = None
    subject_source: Optional[str] = None
    subject_source_version: Optional[str] = None
    object_source: Optional[str] = None
    object_source_version: Optional[str] = None
    mapping_provider: Optional[str] = None
    mapping_tool: Optional[str] = None
    mapping_date: Optional[str] = None
    subject_match_field: Optional[Union[str, EntityId]] = None
    object_match_field: Optional[Union[str, EntityId]] = None
    subject_preprocessing: Optional[str] = None
    object_preprocessing: Optional[str] = None
    match_term_type: Optional[str] = None
    see_also: Optional[str] = None
    other: Optional[str] = None
    comment: Optional[str] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if not isinstance(self.mappings, list):
            self.mappings = [self.mappings] if self.mappings is not None else []
        self.mappings = [
            v if isinstance(v, Mapping) else Mapping(**as_dict(v))
            for v in self.mappings
        ]

        if self.mapping_set_id is not None and not isinstance(
            self.mapping_set_id, EntityId
        ):
            self.mapping_set_id = EntityId(self.mapping_set_id)

        if self.mapping_set_version is not None and not isinstance(
            self.mapping_set_version, str
        ):
            self.mapping_set_version = str(self.mapping_set_version)

        if self.mapping_set_description is not None and not isinstance(
            self.mapping_set_description, str
        ):
            self.mapping_set_description = str(self.mapping_set_description)

        if self.creator_id is not None and not isinstance(self.creator_id, EntityId):
            self.creator_id = EntityId(self.creator_id)

        if self.creator_label is not None and not isinstance(self.creator_label, str):
            self.creator_label = str(self.creator_label)

        if self.license is not None and not isinstance(self.license, str):
            self.license = str(self.license)

        if self.subject_source is not None and not isinstance(self.subject_source, str):
            self.subject_source = str(self.subject_source)

        if self.subject_source_version is not None and not isinstance(
            self.subject_source_version, str
        ):
            self.subject_source_version = str(self.subject_source_version)

        if self.object_source is not None and not isinstance(self.object_source, str):
            self.object_source = str(self.object_source)

        if self.object_source_version is not None and not isinstance(
            self.object_source_version, str
        ):
            self.object_source_version = str(self.object_source_version)

        if self.mapping_provider is not None and not isinstance(
            self.mapping_provider, str
        ):
            self.mapping_provider = str(self.mapping_provider)

        if self.mapping_tool is not None and not isinstance(self.mapping_tool, str):
            self.mapping_tool = str(self.mapping_tool)

        if self.mapping_date is not None and not isinstance(self.mapping_date, str):
            self.mapping_date = str(self.mapping_date)

        if self.subject_match_field is not None and not isinstance(
            self.subject_match_field, EntityId
        ):
            self.subject_match_field = EntityId(self.subject_match_field)

        if self.object_match_field is not None and not isinstance(
            self.object_match_field, EntityId
        ):
            self.object_match_field = EntityId(self.object_match_field)

        if self.subject_preprocessing is not None and not isinstance(
            self.subject_preprocessing, str
        ):
            self.subject_preprocessing = str(self.subject_preprocessing)

        if self.object_preprocessing is not None and not isinstance(
            self.object_preprocessing, str
        ):
            self.object_preprocessing = str(self.object_preprocessing)

        if self.match_term_type is not None and not isinstance(
            self.match_term_type, str
        ):
            self.match_term_type = str(self.match_term_type)

        if self.see_also is not None and not isinstance(self.see_also, str):
            self.see_also = str(self.see_also)

        if self.other is not None and not isinstance(self.other, str):
            self.other = str(self.other)

        if self.comment is not None and not isinstance(self.comment, str):
            self.comment = str(self.comment)

        super().__post_init__(**kwargs)


@dataclass
class Mapping(YAMLRoot):
    """
    Represents an individual mapping between a pair of entities
    """

    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = OWL.Axiom
    class_class_curie: ClassVar[str] = "owl:Axiom"
    class_name: ClassVar[str] = "mapping"
    class_model_uri: ClassVar[URIRef] = SSSOM.Mapping

    subject_id: Optional[Union[str, EntityId]] = None
    subject_label: Optional[str] = None
    subject_category: Optional[str] = None
    predicate_id: Optional[Union[str, EntityId]] = None
    predicate_label: Optional[str] = None
    object_id: Optional[Union[str, EntityId]] = None
    object_label: Optional[str] = None
    object_category: Optional[str] = None
    match_type: Optional[str] = None
    creator_id: Optional[Union[str, EntityId]] = None
    creator_label: Optional[str] = None
    license: Optional[str] = None
    subject_source: Optional[str] = None
    subject_source_version: Optional[str] = None
    object_source: Optional[str] = None
    object_source_version: Optional[str] = None
    mapping_provider: Optional[str] = None
    mapping_cardinality: Optional[Union[str, "MappingCardinalityEnum"]] = None
    mapping_tool: Optional[str] = None
    mapping_date: Optional[str] = None
    confidence: Optional[float] = None
    subject_match_field: Optional[Union[str, EntityId]] = None
    object_match_field: Optional[Union[str, EntityId]] = None
    match_string: Optional[str] = None
    subject_preprocessing: Optional[str] = None
    object_preprocessing: Optional[str] = None
    match_term_type: Optional[str] = None
    semantic_similarity_score: Optional[float] = None
    information_content_mica_score: Optional[float] = None
    see_also: Optional[str] = None
    other: Optional[str] = None
    comment: Optional[str] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self.subject_id is not None and not isinstance(self.subject_id, EntityId):
            self.subject_id = EntityId(self.subject_id)

        if self.subject_label is not None and not isinstance(self.subject_label, str):
            self.subject_label = str(self.subject_label)

        if self.subject_category is not None and not isinstance(
            self.subject_category, str
        ):
            self.subject_category = str(self.subject_category)

        if self.predicate_id is not None and not isinstance(
            self.predicate_id, EntityId
        ):
            self.predicate_id = EntityId(self.predicate_id)

        if self.predicate_label is not None and not isinstance(
            self.predicate_label, str
        ):
            self.predicate_label = str(self.predicate_label)

        if self.object_id is not None and not isinstance(self.object_id, EntityId):
            self.object_id = EntityId(self.object_id)

        if self.object_label is not None and not isinstance(self.object_label, str):
            self.object_label = str(self.object_label)

        if self.object_category is not None and not isinstance(
            self.object_category, str
        ):
            self.object_category = str(self.object_category)

        if self.match_type is not None and not isinstance(self.match_type, str):
            self.match_type = str(self.match_type)

        if self.creator_id is not None and not isinstance(self.creator_id, EntityId):
            self.creator_id = EntityId(self.creator_id)

        if self.creator_label is not None and not isinstance(self.creator_label, str):
            self.creator_label = str(self.creator_label)

        if self.license is not None and not isinstance(self.license, str):
            self.license = str(self.license)

        if self.subject_source is not None and not isinstance(self.subject_source, str):
            self.subject_source = str(self.subject_source)

        if self.subject_source_version is not None and not isinstance(
            self.subject_source_version, str
        ):
            self.subject_source_version = str(self.subject_source_version)

        if self.object_source is not None and not isinstance(self.object_source, str):
            self.object_source = str(self.object_source)

        if self.object_source_version is not None and not isinstance(
            self.object_source_version, str
        ):
            self.object_source_version = str(self.object_source_version)

        if self.mapping_provider is not None and not isinstance(
            self.mapping_provider, str
        ):
            self.mapping_provider = str(self.mapping_provider)

        if self.mapping_cardinality is not None and not isinstance(
            self.mapping_cardinality, MappingCardinalityEnum
        ):
            self.mapping_cardinality = MappingCardinalityEnum(self.mapping_cardinality)

        if self.mapping_tool is not None and not isinstance(self.mapping_tool, str):
            self.mapping_tool = str(self.mapping_tool)

        if self.mapping_date is not None and not isinstance(self.mapping_date, str):
            self.mapping_date = str(self.mapping_date)

        if self.confidence is not None and not isinstance(self.confidence, float):
            self.confidence = float(self.confidence)

        if self.subject_match_field is not None and not isinstance(
            self.subject_match_field, EntityId
        ):
            self.subject_match_field = EntityId(self.subject_match_field)

        if self.object_match_field is not None and not isinstance(
            self.object_match_field, EntityId
        ):
            self.object_match_field = EntityId(self.object_match_field)

        if self.match_string is not None and not isinstance(self.match_string, str):
            self.match_string = str(self.match_string)

        if self.subject_preprocessing is not None and not isinstance(
            self.subject_preprocessing, str
        ):
            self.subject_preprocessing = str(self.subject_preprocessing)

        if self.object_preprocessing is not None and not isinstance(
            self.object_preprocessing, str
        ):
            self.object_preprocessing = str(self.object_preprocessing)

        if self.match_term_type is not None and not isinstance(
            self.match_term_type, str
        ):
            self.match_term_type = str(self.match_term_type)

        if self.semantic_similarity_score is not None and not isinstance(
            self.semantic_similarity_score, float
        ):
            self.semantic_similarity_score = float(self.semantic_similarity_score)

        if self.information_content_mica_score is not None and not isinstance(
            self.information_content_mica_score, float
        ):
            self.information_content_mica_score = float(
                self.information_content_mica_score
            )

        if self.see_also is not None and not isinstance(self.see_also, str):
            self.see_also = str(self.see_also)

        if self.other is not None and not isinstance(self.other, str):
            self.other = str(self.other)

        if self.comment is not None and not isinstance(self.comment, str):
            self.comment = str(self.comment)

        super().__post_init__(**kwargs)


@dataclass
class Entity(YAMLRoot):
    """
    Represents any entity that can be mapped, such as an OWL class or SKOS concept
    """

    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = SSSOM.Entity
    class_class_curie: ClassVar[str] = "sssom:Entity"
    class_name: ClassVar[str] = "entity"
    class_model_uri: ClassVar[URIRef] = SSSOM.Entity

    id: Union[str, EntityId] = None
    label: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, EntityId):
            self.id = EntityId(self.id)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.category is not None and not isinstance(self.category, str):
            self.category = str(self.category)

        if self.source is not None and not isinstance(self.source, str):
            self.source = str(self.source)

        super().__post_init__(**kwargs)


# Enumerations
class MappingCardinalityEnum(EnumDefinitionImpl):

    _defn = EnumDefinition(
        name="MappingCardinalityEnum",
    )

    @classmethod
    def _addvals(cls):
        setattr(cls, "1:1", PermissibleValue(text="1:1"))
        setattr(cls, "1:n", PermissibleValue(text="1:n"))
        setattr(cls, "n:1", PermissibleValue(text="n:1"))
        setattr(cls, "1:0", PermissibleValue(text="1:0"))
        setattr(cls, "0:1", PermissibleValue(text="0:1"))
        setattr(cls, "n:n", PermissibleValue(text="n:n"))


# Slots
class slots:
    pass


slots.entity_field = Slot(
    uri=SSSOM.entity_field,
    name="entity_field",
    curie=SSSOM.curie("entity_field"),
    model_uri=SSSOM.entity_field,
    domain=None,
    range=Optional[str],
)

slots.subject_field = Slot(
    uri=SSSOM.subject_field,
    name="subject_field",
    curie=SSSOM.curie("subject_field"),
    model_uri=SSSOM.subject_field,
    domain=None,
    range=Optional[str],
)

slots.object_field = Slot(
    uri=SSSOM.object_field,
    name="object_field",
    curie=SSSOM.curie("object_field"),
    model_uri=SSSOM.object_field,
    domain=None,
    range=Optional[str],
)

slots.label = Slot(
    uri=SSSOM.label,
    name="label",
    curie=SSSOM.curie("label"),
    model_uri=SSSOM.label,
    domain=None,
    range=Optional[str],
)

slots.source = Slot(
    uri=SSSOM.source,
    name="source",
    curie=SSSOM.curie("source"),
    model_uri=SSSOM.source,
    domain=None,
    range=Optional[str],
)

slots.category = Slot(
    uri=SSSOM.category,
    name="category",
    curie=SSSOM.curie("category"),
    model_uri=SSSOM.category,
    domain=None,
    range=Optional[str],
)

slots.mappings = Slot(
    uri=SSSOM.mappings,
    name="mappings",
    curie=SSSOM.curie("mappings"),
    model_uri=SSSOM.mappings,
    domain=None,
    range=Optional[Union[Union[dict, Mapping], List[Union[dict, Mapping]]]],
)

slots.id = Slot(
    uri=SSSOM.id,
    name="id",
    curie=SSSOM.curie("id"),
    model_uri=SSSOM.id,
    domain=None,
    range=URIRef,
)

slots.subject_id = Slot(
    uri=OWL.annotatedSource,
    name="subject_id",
    curie=OWL.curie("annotatedSource"),
    model_uri=SSSOM.subject_id,
    domain=None,
    range=Optional[Union[str, EntityId]],
)

slots.subject_label = Slot(
    uri=SSSOM.subject_label,
    name="subject_label",
    curie=SSSOM.curie("subject_label"),
    model_uri=SSSOM.subject_label,
    domain=None,
    range=Optional[str],
)

slots.subject_category = Slot(
    uri=SSSOM.subject_category,
    name="subject_category",
    curie=SSSOM.curie("subject_category"),
    model_uri=SSSOM.subject_category,
    domain=None,
    range=Optional[str],
)

slots.predicate_id = Slot(
    uri=OWL.annotatedProperty,
    name="predicate_id",
    curie=OWL.curie("annotatedProperty"),
    model_uri=SSSOM.predicate_id,
    domain=None,
    range=Optional[Union[str, EntityId]],
)

slots.predicate_label = Slot(
    uri=SSSOM.predicate_label,
    name="predicate_label",
    curie=SSSOM.curie("predicate_label"),
    model_uri=SSSOM.predicate_label,
    domain=None,
    range=Optional[str],
)

slots.object_id = Slot(
    uri=OWL.annotatedTarget,
    name="object_id",
    curie=OWL.curie("annotatedTarget"),
    model_uri=SSSOM.object_id,
    domain=None,
    range=Optional[Union[str, EntityId]],
)

slots.object_label = Slot(
    uri=SSSOM.object_label,
    name="object_label",
    curie=SSSOM.curie("object_label"),
    model_uri=SSSOM.object_label,
    domain=None,
    range=Optional[str],
)

slots.object_category = Slot(
    uri=SSSOM.object_category,
    name="object_category",
    curie=SSSOM.curie("object_category"),
    model_uri=SSSOM.object_category,
    domain=None,
    range=Optional[str],
)

slots.match_type = Slot(
    uri=SSSOM.match_type,
    name="match_type",
    curie=SSSOM.curie("match_type"),
    model_uri=SSSOM.match_type,
    domain=None,
    range=Optional[str],
)

slots.mapping_set_id = Slot(
    uri=SSSOM.mapping_set_id,
    name="mapping_set_id",
    curie=SSSOM.curie("mapping_set_id"),
    model_uri=SSSOM.mapping_set_id,
    domain=None,
    range=Optional[Union[str, EntityId]],
)

slots.mapping_set_version = Slot(
    uri=SSSOM.mapping_set_version,
    name="mapping_set_version",
    curie=SSSOM.curie("mapping_set_version"),
    model_uri=SSSOM.mapping_set_version,
    domain=None,
    range=Optional[str],
    mappings=[OWL.versionInfo],
)

slots.mapping_set_description = Slot(
    uri=SSSOM.mapping_set_description,
    name="mapping_set_description",
    curie=SSSOM.curie("mapping_set_description"),
    model_uri=SSSOM.mapping_set_description,
    domain=None,
    range=Optional[str],
)

slots.creator_id = Slot(
    uri=SSSOM.creator_id,
    name="creator_id",
    curie=SSSOM.curie("creator_id"),
    model_uri=SSSOM.creator_id,
    domain=None,
    range=Optional[Union[str, EntityId]],
    mappings=[DC.creator],
)

slots.creator_label = Slot(
    uri=SSSOM.creator_label,
    name="creator_label",
    curie=SSSOM.curie("creator_label"),
    model_uri=SSSOM.creator_label,
    domain=None,
    range=Optional[str],
)

slots.license = Slot(
    uri=SSSOM.license,
    name="license",
    curie=SSSOM.curie("license"),
    model_uri=SSSOM.license,
    domain=None,
    range=Optional[str],
    mappings=[DC.license],
)

slots.subject_source = Slot(
    uri=SSSOM.subject_source,
    name="subject_source",
    curie=SSSOM.curie("subject_source"),
    model_uri=SSSOM.subject_source,
    domain=None,
    range=Optional[str],
)

slots.subject_source_version = Slot(
    uri=SSSOM.subject_source_version,
    name="subject_source_version",
    curie=SSSOM.curie("subject_source_version"),
    model_uri=SSSOM.subject_source_version,
    domain=None,
    range=Optional[str],
)

slots.object_source = Slot(
    uri=SSSOM.object_source,
    name="object_source",
    curie=SSSOM.curie("object_source"),
    model_uri=SSSOM.object_source,
    domain=None,
    range=Optional[str],
)

slots.object_source_version = Slot(
    uri=SSSOM.object_source_version,
    name="object_source_version",
    curie=SSSOM.curie("object_source_version"),
    model_uri=SSSOM.object_source_version,
    domain=None,
    range=Optional[str],
)

slots.mapping_provider = Slot(
    uri=SSSOM.mapping_provider,
    name="mapping_provider",
    curie=SSSOM.curie("mapping_provider"),
    model_uri=SSSOM.mapping_provider,
    domain=None,
    range=Optional[str],
)

slots.mapping_cardinality = Slot(
    uri=SSSOM.mapping_cardinality,
    name="mapping_cardinality",
    curie=SSSOM.curie("mapping_cardinality"),
    model_uri=SSSOM.mapping_cardinality,
    domain=None,
    range=Optional[Union[str, "MappingCardinalityEnum"]],
)

slots.mapping_tool = Slot(
    uri=SSSOM.mapping_tool,
    name="mapping_tool",
    curie=SSSOM.curie("mapping_tool"),
    model_uri=SSSOM.mapping_tool,
    domain=None,
    range=Optional[str],
)

slots.mapping_date = Slot(
    uri=SSSOM.mapping_date,
    name="mapping_date",
    curie=SSSOM.curie("mapping_date"),
    model_uri=SSSOM.mapping_date,
    domain=None,
    range=Optional[str],
    mappings=[DC.date],
)

slots.confidence = Slot(
    uri=SSSOM.confidence,
    name="confidence",
    curie=SSSOM.curie("confidence"),
    model_uri=SSSOM.confidence,
    domain=None,
    range=Optional[float],
)

slots.subject_match_field = Slot(
    uri=SSSOM.subject_match_field,
    name="subject_match_field",
    curie=SSSOM.curie("subject_match_field"),
    model_uri=SSSOM.subject_match_field,
    domain=None,
    range=Optional[Union[str, EntityId]],
)

slots.object_match_field = Slot(
    uri=SSSOM.object_match_field,
    name="object_match_field",
    curie=SSSOM.curie("object_match_field"),
    model_uri=SSSOM.object_match_field,
    domain=None,
    range=Optional[Union[str, EntityId]],
)

slots.match_string = Slot(
    uri=SSSOM.match_string,
    name="match_string",
    curie=SSSOM.curie("match_string"),
    model_uri=SSSOM.match_string,
    domain=None,
    range=Optional[str],
)

slots.subject_preprocessing = Slot(
    uri=SSSOM.subject_preprocessing,
    name="subject_preprocessing",
    curie=SSSOM.curie("subject_preprocessing"),
    model_uri=SSSOM.subject_preprocessing,
    domain=None,
    range=Optional[str],
)

slots.object_preprocessing = Slot(
    uri=SSSOM.object_preprocessing,
    name="object_preprocessing",
    curie=SSSOM.curie("object_preprocessing"),
    model_uri=SSSOM.object_preprocessing,
    domain=None,
    range=Optional[str],
)

slots.match_term_type = Slot(
    uri=SSSOM.match_term_type,
    name="match_term_type",
    curie=SSSOM.curie("match_term_type"),
    model_uri=SSSOM.match_term_type,
    domain=None,
    range=Optional[str],
)

slots.semantic_similarity_score = Slot(
    uri=SSSOM.semantic_similarity_score,
    name="semantic_similarity_score",
    curie=SSSOM.curie("semantic_similarity_score"),
    model_uri=SSSOM.semantic_similarity_score,
    domain=None,
    range=Optional[float],
)

slots.information_content_mica_score = Slot(
    uri=SSSOM.information_content_mica_score,
    name="information_content_mica_score",
    curie=SSSOM.curie("information_content_mica_score"),
    model_uri=SSSOM.information_content_mica_score,
    domain=None,
    range=Optional[float],
)

slots.see_also = Slot(
    uri=RDFS.seeAlso,
    name="see_also",
    curie=RDFS.curie("seeAlso"),
    model_uri=SSSOM.see_also,
    domain=None,
    range=Optional[str],
)

slots.other = Slot(
    uri=SSSOM.other,
    name="other",
    curie=SSSOM.curie("other"),
    model_uri=SSSOM.other,
    domain=None,
    range=Optional[str],
)

slots.comment = Slot(
    uri=RDFS.comment,
    name="comment",
    curie=RDFS.curie("comment"),
    model_uri=SSSOM.comment,
    domain=None,
    range=Optional[str],
)

slots.required = Slot(
    uri=SSSOM.required,
    name="required",
    curie=SSSOM.curie("required"),
    model_uri=SSSOM.required,
    domain=None,
    range=Optional[str],
)

slots.metadata_element = Slot(
    uri=SSSOM.metadata_element,
    name="metadata_element",
    curie=SSSOM.curie("metadata_element"),
    model_uri=SSSOM.metadata_element,
    domain=None,
    range=Optional[str],
)

slots.scope = Slot(
    uri=SSSOM.scope,
    name="scope",
    curie=SSSOM.curie("scope"),
    model_uri=SSSOM.scope,
    domain=None,
    range=Optional[str],
)

slots.rdf_example = Slot(
    uri=SSSOM.rdf_example,
    name="rdf_example",
    curie=SSSOM.curie("rdf_example"),
    model_uri=SSSOM.rdf_example,
    domain=None,
    range=Optional[str],
)

slots.tsv_example = Slot(
    uri=SSSOM.tsv_example,
    name="tsv_example",
    curie=SSSOM.curie("tsv_example"),
    model_uri=SSSOM.tsv_example,
    domain=None,
    range=Optional[str],
)

slots.equivalent_property = Slot(
    uri=SSSOM.equivalent_property,
    name="equivalent_property",
    curie=SSSOM.curie("equivalent_property"),
    model_uri=SSSOM.equivalent_property,
    domain=None,
    range=Optional[str],
)
