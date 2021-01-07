# Auto generated from sssom.yaml by pythongen.py version: 0.4.0
# Generation date: 2020-08-19 09:13
# Schema: sssom
#
# id: http://purl.org/sssom/schema/
# description: Datamodel for Simple Standard for Sharing Ontology Mappings (SSSOM)
# license: https://creativecommons.org/publicdomain/zero/1.0/

import dataclasses
import sys
from typing import Optional, List, Union, Dict, ClassVar, Any
from dataclasses import dataclass
from biolinkml.utils.slot import Slot
from biolinkml.utils.metamodelcore import empty_list, empty_dict, bnode
from biolinkml.utils.yamlutils import YAMLRoot, extended_str, extended_float, extended_int
if sys.version_info < (3, 7, 6):
    from biolinkml.utils.dataclass_extensions_375 import dataclasses_init_fn_with_kwargs
else:
    from biolinkml.utils.dataclass_extensions_376 import dataclasses_init_fn_with_kwargs
from biolinkml.utils.formatutils import camelcase, underscore, sfx
from rdflib import Namespace, URIRef
from biolinkml.utils.curienamespace import CurieNamespace
from includes.types import Double, String

metamodel_version = "1.5.3"

# Overwrite dataclasses _init_fn to add **kwargs in __init__
dataclasses._init_fn = dataclasses_init_fn_with_kwargs

# Namespaces
BIOLINKML = CurieNamespace('biolinkml', 'https://w3id.org/biolink/biolinkml/')
DC = CurieNamespace('dc', 'http://purl.org/dc/terms/')
DCTERMS = CurieNamespace('dcterms', 'http://purl.org/dc/terms/')
OWL = CurieNamespace('owl', 'http://www.w3.org/2002/07/owl#')
RDF = CurieNamespace('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
RDFS = CurieNamespace('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
SSSOM_NS='http://example.org/sssom/'
SSSOM = CurieNamespace('sssom', SSSOM_NS)
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

    mappings: List[Union[dict, "Mapping"]] = empty_list()
    mapping_set_id: Optional[Union[str, EntityId]] = None
    mapping_set_version: Optional[str] = None
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

    def __post_init__(self, **kwargs: Dict[str, Any]):
        self.mappings = [Mapping(*e) for e in self.mappings.items()] if isinstance(self.mappings, dict) \
                         else [v if isinstance(v, Mapping) else Mapping(**v)
                               for v in ([self.mappings] if isinstance(self.mappings, str) else self.mappings)]
        if self.mapping_set_id is not None and not isinstance(self.mapping_set_id, EntityId):
            self.mapping_set_id = EntityId(self.mapping_set_id)
        if self.creator_id is not None and not isinstance(self.creator_id, EntityId):
            self.creator_id = EntityId(self.creator_id)
        if self.subject_match_field is not None and not isinstance(self.subject_match_field, EntityId):
            self.subject_match_field = EntityId(self.subject_match_field)
        if self.object_match_field is not None and not isinstance(self.object_match_field, EntityId):
            self.object_match_field = EntityId(self.object_match_field)
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

    def __post_init__(self, **kwargs: Dict[str, Any]):
        if self.subject_id is not None and not isinstance(self.subject_id, EntityId):
            self.subject_id = EntityId(self.subject_id)
        if self.predicate_id is not None and not isinstance(self.predicate_id, EntityId):
            self.predicate_id = EntityId(self.predicate_id)
        if self.object_id is not None and not isinstance(self.object_id, EntityId):
            self.object_id = EntityId(self.object_id)
        if self.creator_id is not None and not isinstance(self.creator_id, EntityId):
            self.creator_id = EntityId(self.creator_id)
        if self.subject_match_field is not None and not isinstance(self.subject_match_field, EntityId):
            self.subject_match_field = EntityId(self.subject_match_field)
        if self.object_match_field is not None and not isinstance(self.object_match_field, EntityId):
            self.object_match_field = EntityId(self.object_match_field)
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

    id: Union[str, EntityId]

    def __post_init__(self, **kwargs: Dict[str, Any]):
        if self.id is None:
            raise ValueError(f"id must be supplied")
        if not isinstance(self.id, EntityId):
            self.id = EntityId(self.id)
        super().__post_init__(**kwargs)



# Slots
class slots:
    pass

slots.mappings = Slot(uri=SSSOM.mappings, name="mappings", curie=SSSOM.curie('mappings'),
                      model_uri=SSSOM.mappings, domain=None, range=List[Union[dict, Mapping]])

slots.id = Slot(uri=SSSOM.id, name="id", curie=SSSOM.curie('id'),
                      model_uri=SSSOM.id, domain=None, range=URIRef)

slots.subject_id = Slot(uri=OWL.annotatedSource, name="subject_id", curie=OWL.curie('annotatedSource'),
                      model_uri=SSSOM.subject_id, domain=None, range=Optional[Union[str, EntityId]])

slots.subject_label = Slot(uri=SSSOM.subject_label, name="subject_label", curie=SSSOM.curie('subject_label'),
                      model_uri=SSSOM.subject_label, domain=None, range=Optional[str])

slots.subject_category = Slot(uri=SSSOM.subject_category, name="subject_category", curie=SSSOM.curie('subject_category'),
                      model_uri=SSSOM.subject_category, domain=None, range=Optional[str])

slots.predicate_id = Slot(uri=OWL.annotatedProperty, name="predicate_id", curie=OWL.curie('annotatedProperty'),
                      model_uri=SSSOM.predicate_id, domain=None, range=Optional[Union[str, EntityId]])

slots.predicate_label = Slot(uri=SSSOM.predicate_label, name="predicate_label", curie=SSSOM.curie('predicate_label'),
                      model_uri=SSSOM.predicate_label, domain=None, range=Optional[str])

slots.object_id = Slot(uri=OWL.annotatedTarget, name="object_id", curie=OWL.curie('annotatedTarget'),
                      model_uri=SSSOM.object_id, domain=None, range=Optional[Union[str, EntityId]])

slots.object_label = Slot(uri=SSSOM.object_label, name="object_label", curie=SSSOM.curie('object_label'),
                      model_uri=SSSOM.object_label, domain=None, range=Optional[str])

slots.object_category = Slot(uri=SSSOM.object_category, name="object_category", curie=SSSOM.curie('object_category'),
                      model_uri=SSSOM.object_category, domain=None, range=Optional[str])

slots.match_type = Slot(uri=SSSOM.match_type, name="match_type", curie=SSSOM.curie('match_type'),
                      model_uri=SSSOM.match_type, domain=None, range=Optional[str])

slots.mapping_set_id = Slot(uri=SSSOM.mapping_set_id, name="mapping_set_id", curie=SSSOM.curie('mapping_set_id'),
                      model_uri=SSSOM.mapping_set_id, domain=None, range=Optional[Union[str, EntityId]])

slots.mapping_set_version = Slot(uri=SSSOM.mapping_set_version, name="mapping_set_version", curie=SSSOM.curie('mapping_set_version'),
                      model_uri=SSSOM.mapping_set_version, domain=None, range=Optional[str], mappings = [OWL.versionInfo])

slots.creator_id = Slot(uri=SSSOM.creator_id, name="creator_id", curie=SSSOM.curie('creator_id'),
                      model_uri=SSSOM.creator_id, domain=None, range=Optional[Union[str, EntityId]], mappings = [DC.creator])

slots.creator_label = Slot(uri=SSSOM.creator_label, name="creator_label", curie=SSSOM.curie('creator_label'),
                      model_uri=SSSOM.creator_label, domain=None, range=Optional[str])

slots.license = Slot(uri=SSSOM.license, name="license", curie=SSSOM.curie('license'),
                      model_uri=SSSOM.license, domain=None, range=Optional[str], mappings = [DC.license])

slots.subject_source = Slot(uri=SSSOM.subject_source, name="subject_source", curie=SSSOM.curie('subject_source'),
                      model_uri=SSSOM.subject_source, domain=None, range=Optional[str])

slots.subject_source_version = Slot(uri=SSSOM.subject_source_version, name="subject_source_version", curie=SSSOM.curie('subject_source_version'),
                      model_uri=SSSOM.subject_source_version, domain=None, range=Optional[str])

slots.object_source = Slot(uri=SSSOM.object_source, name="object_source", curie=SSSOM.curie('object_source'),
                      model_uri=SSSOM.object_source, domain=None, range=Optional[str])

slots.object_source_version = Slot(uri=SSSOM.object_source_version, name="object_source_version", curie=SSSOM.curie('object_source_version'),
                      model_uri=SSSOM.object_source_version, domain=None, range=Optional[str])

slots.mapping_provider = Slot(uri=SSSOM.mapping_provider, name="mapping_provider", curie=SSSOM.curie('mapping_provider'),
                      model_uri=SSSOM.mapping_provider, domain=None, range=Optional[str])

slots.mapping_tool = Slot(uri=SSSOM.mapping_tool, name="mapping_tool", curie=SSSOM.curie('mapping_tool'),
                      model_uri=SSSOM.mapping_tool, domain=None, range=Optional[str])

slots.mapping_date = Slot(uri=SSSOM.mapping_date, name="mapping_date", curie=SSSOM.curie('mapping_date'),
                      model_uri=SSSOM.mapping_date, domain=None, range=Optional[str], mappings = [DC.date])

slots.confidence = Slot(uri=SSSOM.confidence, name="confidence", curie=SSSOM.curie('confidence'),
                      model_uri=SSSOM.confidence, domain=None, range=Optional[float])

slots.subject_match_field = Slot(uri=SSSOM.subject_match_field, name="subject_match_field", curie=SSSOM.curie('subject_match_field'),
                      model_uri=SSSOM.subject_match_field, domain=None, range=Optional[Union[str, EntityId]])

slots.object_match_field = Slot(uri=SSSOM.object_match_field, name="object_match_field", curie=SSSOM.curie('object_match_field'),
                      model_uri=SSSOM.object_match_field, domain=None, range=Optional[Union[str, EntityId]])

slots.match_string = Slot(uri=SSSOM.match_string, name="match_string", curie=SSSOM.curie('match_string'),
                      model_uri=SSSOM.match_string, domain=None, range=Optional[str])

slots.subject_preprocessing = Slot(uri=SSSOM.subject_preprocessing, name="subject_preprocessing", curie=SSSOM.curie('subject_preprocessing'),
                      model_uri=SSSOM.subject_preprocessing, domain=None, range=Optional[str])

slots.object_preprocessing = Slot(uri=SSSOM.object_preprocessing, name="object_preprocessing", curie=SSSOM.curie('object_preprocessing'),
                      model_uri=SSSOM.object_preprocessing, domain=None, range=Optional[str])

slots.match_term_type = Slot(uri=SSSOM.match_term_type, name="match_term_type", curie=SSSOM.curie('match_term_type'),
                      model_uri=SSSOM.match_term_type, domain=None, range=Optional[str])

slots.semantic_similarity_score = Slot(uri=SSSOM.semantic_similarity_score, name="semantic_similarity_score", curie=SSSOM.curie('semantic_similarity_score'),
                      model_uri=SSSOM.semantic_similarity_score, domain=None, range=Optional[float])

slots.information_content_mica_score = Slot(uri=SSSOM.information_content_mica_score, name="information_content_mica_score", curie=SSSOM.curie('information_content_mica_score'),
                      model_uri=SSSOM.information_content_mica_score, domain=None, range=Optional[float])

slots.see_also = Slot(uri=SSSOM.see_also, name="see_also", curie=SSSOM.curie('see_also'),
                      model_uri=SSSOM.see_also, domain=None, range=Optional[str], mappings = [RDFS.seeAlso])

slots.other = Slot(uri=SSSOM.other, name="other", curie=SSSOM.curie('other'),
                      model_uri=SSSOM.other, domain=None, range=Optional[str])

slots.comment = Slot(uri=SSSOM.comment, name="comment", curie=SSSOM.curie('comment'),
                      model_uri=SSSOM.comment, domain=None, range=Optional[str])

slots.required = Slot(uri=SSSOM.required, name="required", curie=SSSOM.curie('required'),
                      model_uri=SSSOM.required, domain=None, range=Optional[str])

slots.metadata_element = Slot(uri=SSSOM.metadata_element, name="metadata_element", curie=SSSOM.curie('metadata_element'),
                      model_uri=SSSOM.metadata_element, domain=None, range=Optional[str])

slots.scope = Slot(uri=SSSOM.scope, name="scope", curie=SSSOM.curie('scope'),
                      model_uri=SSSOM.scope, domain=None, range=Optional[str])

slots.rdf_example = Slot(uri=SSSOM.rdf_example, name="rdf_example", curie=SSSOM.curie('rdf_example'),
                      model_uri=SSSOM.rdf_example, domain=None, range=Optional[str])

slots.tsv_example = Slot(uri=SSSOM.tsv_example, name="tsv_example", curie=SSSOM.curie('tsv_example'),
                      model_uri=SSSOM.tsv_example, domain=None, range=Optional[str])

slots.equivalent_property = Slot(uri=SSSOM.equivalent_property, name="equivalent_property", curie=SSSOM.curie('equivalent_property'),
                      model_uri=SSSOM.equivalent_property, domain=None, range=Optional[str])
