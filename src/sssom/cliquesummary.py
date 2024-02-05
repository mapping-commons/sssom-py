"""
Auto generated from cliquesummary.yaml by pythongen.py version: 0.9.0
Generation date: 2021-08-19 18:02
Schema: sssom-cliquesummary

id: https://w3id.org/sssom/schema/cliquesummary/
description: Data dictionary for clique summaries
license: https://creativecommons.org/publicdomain/zero/1.0/
"""

import dataclasses
import re
import sys
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Union

from jsonasobj2 import JsonObj, as_dict
from linkml_runtime.linkml_model.meta import EnumDefinition, PermissibleValue, PvFormulaOptions
from linkml_runtime.utils.curienamespace import CurieNamespace
from linkml_runtime.utils.dataclass_extensions_376 import dataclasses_init_fn_with_kwargs
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from linkml_runtime.utils.formatutils import camelcase, sfx, underscore
from linkml_runtime.utils.metamodelcore import (
    URI,
    Bool,
    Decimal,
    ElementIdentifier,
    NCName,
    NodeIdentifier,
    URIorCURIE,
    XSDDate,
    XSDDateTime,
    XSDTime,
    bnode,
    empty_dict,
    empty_list,
)
from linkml_runtime.utils.slot import Slot
from linkml_runtime.utils.yamlutils import YAMLRoot, extended_float, extended_int, extended_str
from rdflib import Namespace, URIRef

metamodel_version = "1.7.0"

# Overwrite dataclasses _init_fn to add **kwargs in __init__
dataclasses._init_fn = dataclasses_init_fn_with_kwargs

# Namespaces
BIOLINKML = CurieNamespace("biolinkml", "https://w3id.org/biolink/biolinkml/")
SHEX = CurieNamespace("shex", "http://www.w3.org/ns/shex#")
SSSOM = CurieNamespace("sssom", "https://w3id.org/sssom/")
SSSOM_CS = CurieNamespace("sssom_cs", "https://w3id.org/sssom/cliquesummary")
XSD = CurieNamespace("xsd", "http://www.w3.org/2001/XMLSchema#")
DEFAULT_ = SSSOM.CS


# Types
class Count(int):
    """Count."""

    type_class_uri = XSD.integer
    type_class_curie = "xsd:integer"
    type_name = "count"
    type_model_uri = SSSOM.CS.Count


class String(str):
    """A character string"""

    type_class_uri = XSD.string
    type_class_curie = "xsd:string"
    type_name = "string"
    type_model_uri = SSSOM.CS.String


class Integer(int):
    """An integer"""

    type_class_uri = XSD.integer
    type_class_curie = "xsd:integer"
    type_name = "integer"
    type_model_uri = SSSOM.CS.Integer


class Boolean(Bool):
    """A binary (true or false) value"""

    type_class_uri = XSD.boolean
    type_class_curie = "xsd:boolean"
    type_name = "boolean"
    type_model_uri = SSSOM.CS.Boolean


class Float(float):
    """A real number that conforms to the xsd:float specification"""

    type_class_uri = XSD.float
    type_class_curie = "xsd:float"
    type_name = "float"
    type_model_uri = SSSOM.CS.Float


class Double(float):
    """A real number that conforms to the xsd:double specification"""

    type_class_uri = XSD.double
    type_class_curie = "xsd:double"
    type_name = "double"
    type_model_uri = SSSOM.CS.Double


class Decimal(Decimal):
    """A real number with arbitrary precision that conforms to the xsd:decimal specification"""

    type_class_uri = XSD.decimal
    type_class_curie = "xsd:decimal"
    type_name = "decimal"
    type_model_uri = SSSOM.CS.Decimal


class Time(XSDTime):
    """A time object represents a (local) time of day, independent of any particular day"""

    type_class_uri = XSD.dateTime
    type_class_curie = "xsd:dateTime"
    type_name = "time"
    type_model_uri = SSSOM.CS.Time


class Date(XSDDate):
    """a date (year, month and day) in an idealized calendar"""

    type_class_uri = XSD.date
    type_class_curie = "xsd:date"
    type_name = "date"
    type_model_uri = SSSOM.CS.Date


class Datetime(XSDDateTime):
    """The combination of a date and time"""

    type_class_uri = XSD.dateTime
    type_class_curie = "xsd:dateTime"
    type_name = "datetime"
    type_model_uri = SSSOM.CS.Datetime


class Uriorcurie(URIorCURIE):
    """a URI or a CURIE"""

    type_class_uri = XSD.anyURI
    type_class_curie = "xsd:anyURI"
    type_name = "uriorcurie"
    type_model_uri = SSSOM.CS.Uriorcurie


class Uri(URI):
    """a complete URI"""

    type_class_uri = XSD.anyURI
    type_class_curie = "xsd:anyURI"
    type_name = "uri"
    type_model_uri = SSSOM.CS.Uri


class Ncname(NCName):
    """Prefix part of CURIE"""

    type_class_uri = XSD.string
    type_class_curie = "xsd:string"
    type_name = "ncname"
    type_model_uri = SSSOM.CS.Ncname


class Objectidentifier(ElementIdentifier):
    """A URI or CURIE that represents an object in the model."""

    type_class_uri = SHEX.iri
    type_class_curie = "shex:iri"
    type_name = "objectidentifier"
    type_model_uri = SSSOM.CS.Objectidentifier


class Nodeidentifier(NodeIdentifier):
    """A URI, CURIE or BNODE that represents a node in a model."""

    type_class_uri = SHEX.nonLiteral
    type_class_curie = "shex:nonLiteral"
    type_name = "nodeidentifier"
    type_model_uri = SSSOM.CS.Nodeidentifier


# Class references
class CliqueId(extended_str):
    """CliqueId."""

    pass


@dataclass
class Clique(YAMLRoot):
    """
    A clique
    """

    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = SSSOM.CS.Clique
    class_class_curie: ClassVar[str] = "sssom.cs:Clique"
    class_name: ClassVar[str] = "clique"
    class_model_uri: ClassVar[URIRef] = SSSOM.CS.Clique

    id: Union[str, CliqueId] = None
    members: Optional[Union[str, List[str]]] = empty_list()
    members_labels: Optional[Union[str, List[str]]] = empty_list()
    num_members: Optional[int] = None
    max_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    avg_confidence: Optional[float] = None
    is_conflated: Optional[Union[bool, Bool]] = None
    is_all_conflated: Optional[Union[bool, Bool]] = None
    total_conflated: Optional[int] = None
    proportion_conflated: Optional[float] = None
    conflation_score: Optional[float] = None
    members_count: Optional[int] = None
    min_count_by_source: Optional[int] = None
    max_count_by_source: Optional[int] = None
    avg_count_by_source: Optional[float] = None
    harmonic_mean_count_by_source: Optional[float] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, CliqueId):
            self.id = CliqueId(self.id)

        if not isinstance(self.members, list):
            self.members = [self.members] if self.members is not None else []
        self.members = [v if isinstance(v, str) else str(v) for v in self.members]

        if not isinstance(self.members_labels, list):
            self.members_labels = [self.members_labels] if self.members_labels is not None else []
        self.members_labels = [v if isinstance(v, str) else str(v) for v in self.members_labels]

        if self.num_members is not None and not isinstance(self.num_members, int):
            self.num_members = int(self.num_members)

        if self.max_confidence is not None and not isinstance(self.max_confidence, float):
            self.max_confidence = float(self.max_confidence)

        if self.min_confidence is not None and not isinstance(self.min_confidence, float):
            self.min_confidence = float(self.min_confidence)

        if self.avg_confidence is not None and not isinstance(self.avg_confidence, float):
            self.avg_confidence = float(self.avg_confidence)

        if self.is_conflated is not None and not isinstance(self.is_conflated, Bool):
            self.is_conflated = Bool(self.is_conflated)

        if self.is_all_conflated is not None and not isinstance(self.is_all_conflated, Bool):
            self.is_all_conflated = Bool(self.is_all_conflated)

        if self.total_conflated is not None and not isinstance(self.total_conflated, int):
            self.total_conflated = int(self.total_conflated)

        if self.proportion_conflated is not None and not isinstance(
            self.proportion_conflated, float
        ):
            self.proportion_conflated = float(self.proportion_conflated)

        if self.conflation_score is not None and not isinstance(self.conflation_score, float):
            self.conflation_score = float(self.conflation_score)

        if self.members_count is not None and not isinstance(self.members_count, int):
            self.members_count = int(self.members_count)

        if self.min_count_by_source is not None and not isinstance(self.min_count_by_source, int):
            self.min_count_by_source = int(self.min_count_by_source)

        if self.max_count_by_source is not None and not isinstance(self.max_count_by_source, int):
            self.max_count_by_source = int(self.max_count_by_source)

        if self.avg_count_by_source is not None and not isinstance(self.avg_count_by_source, float):
            self.avg_count_by_source = float(self.avg_count_by_source)

        if self.harmonic_mean_count_by_source is not None and not isinstance(
            self.harmonic_mean_count_by_source, float
        ):
            self.harmonic_mean_count_by_source = float(self.harmonic_mean_count_by_source)

        super().__post_init__(**kwargs)


# Enumerations


# Slots
class slots:
    """Slots."""

    pass


slots.id = Slot(
    uri=SSSOM.CS.id,
    name="id",
    curie=SSSOM.CS.curie("id"),
    model_uri=SSSOM.CS.id,
    domain=None,
    range=URIRef,
)

slots.members = Slot(
    uri=SSSOM.CS.members,
    name="members",
    curie=SSSOM.CS.curie("members"),
    model_uri=SSSOM.CS.members,
    domain=None,
    range=Optional[Union[str, List[str]]],
)

slots.members_labels = Slot(
    uri=SSSOM.CS.members_labels,
    name="members_labels",
    curie=SSSOM.CS.curie("members_labels"),
    model_uri=SSSOM.CS.members_labels,
    domain=None,
    range=Optional[Union[str, List[str]]],
)

slots.num_members = Slot(
    uri=SSSOM.CS.num_members,
    name="num_members",
    curie=SSSOM.CS.curie("num_members"),
    model_uri=SSSOM.CS.num_members,
    domain=None,
    range=Optional[int],
)

slots.sources = Slot(
    uri=SSSOM.CS.sources,
    name="sources",
    curie=SSSOM.CS.curie("sources"),
    model_uri=SSSOM.CS.sources,
    domain=None,
    range=Optional[Union[str, List[str]]],
)

slots.num_sources = Slot(
    uri=SSSOM.CS.num_sources,
    name="num_sources",
    curie=SSSOM.CS.curie("num_sources"),
    model_uri=SSSOM.CS.num_sources,
    domain=None,
    range=Optional[int],
)

slots.max_confidence = Slot(
    uri=SSSOM.CS.max_confidence,
    name="max_confidence",
    curie=SSSOM.CS.curie("max_confidence"),
    model_uri=SSSOM.CS.max_confidence,
    domain=None,
    range=Optional[float],
)

slots.min_confidence = Slot(
    uri=SSSOM.CS.min_confidence,
    name="min_confidence",
    curie=SSSOM.CS.curie("min_confidence"),
    model_uri=SSSOM.CS.min_confidence,
    domain=None,
    range=Optional[float],
)

slots.avg_confidence = Slot(
    uri=SSSOM.CS.avg_confidence,
    name="avg_confidence",
    curie=SSSOM.CS.curie("avg_confidence"),
    model_uri=SSSOM.CS.avg_confidence,
    domain=None,
    range=Optional[float],
)

slots.is_conflated = Slot(
    uri=SSSOM.CS.is_conflated,
    name="is_conflated",
    curie=SSSOM.CS.curie("is_conflated"),
    model_uri=SSSOM.CS.is_conflated,
    domain=None,
    range=Optional[Union[bool, Bool]],
)

slots.is_all_conflated = Slot(
    uri=SSSOM.CS.is_all_conflated,
    name="is_all_conflated",
    curie=SSSOM.CS.curie("is_all_conflated"),
    model_uri=SSSOM.CS.is_all_conflated,
    domain=None,
    range=Optional[Union[bool, Bool]],
)

slots.total_conflated = Slot(
    uri=SSSOM.CS.total_conflated,
    name="total_conflated",
    curie=SSSOM.CS.curie("total_conflated"),
    model_uri=SSSOM.CS.total_conflated,
    domain=None,
    range=Optional[int],
)

slots.proportion_conflated = Slot(
    uri=SSSOM.CS.proportion_conflated,
    name="proportion_conflated",
    curie=SSSOM.CS.curie("proportion_conflated"),
    model_uri=SSSOM.CS.proportion_conflated,
    domain=None,
    range=Optional[float],
)

slots.conflation_score = Slot(
    uri=SSSOM.CS.conflation_score,
    name="conflation_score",
    curie=SSSOM.CS.curie("conflation_score"),
    model_uri=SSSOM.CS.conflation_score,
    domain=None,
    range=Optional[float],
)

slots.members_count = Slot(
    uri=SSSOM.CS.members_count,
    name="members_count",
    curie=SSSOM.CS.curie("members_count"),
    model_uri=SSSOM.CS.members_count,
    domain=None,
    range=Optional[int],
)

slots.min_count_by_source = Slot(
    uri=SSSOM.CS.min_count_by_source,
    name="min_count_by_source",
    curie=SSSOM.CS.curie("min_count_by_source"),
    model_uri=SSSOM.CS.min_count_by_source,
    domain=None,
    range=Optional[int],
)

slots.max_count_by_source = Slot(
    uri=SSSOM.CS.max_count_by_source,
    name="max_count_by_source",
    curie=SSSOM.CS.curie("max_count_by_source"),
    model_uri=SSSOM.CS.max_count_by_source,
    domain=None,
    range=Optional[int],
)

slots.avg_count_by_source = Slot(
    uri=SSSOM.CS.avg_count_by_source,
    name="avg_count_by_source",
    curie=SSSOM.CS.curie("avg_count_by_source"),
    model_uri=SSSOM.CS.avg_count_by_source,
    domain=None,
    range=Optional[float],
)

slots.harmonic_mean_count_by_source = Slot(
    uri=SSSOM.CS.harmonic_mean_count_by_source,
    name="harmonic_mean_count_by_source",
    curie=SSSOM.CS.curie("harmonic_mean_count_by_source"),
    model_uri=SSSOM.CS.harmonic_mean_count_by_source,
    domain=None,
    range=Optional[float],
)
