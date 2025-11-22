"""Internal code for RDF import and export."""

from __future__ import annotations

import logging as _logging
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeAlias, Union, cast

from curies import Converter
from linkml_runtime.linkml_model.meta import SlotDefinition
from linkml_runtime.utils.schemaview import SchemaView
from pandas import DataFrame, Series
from rdflib import BNode, Graph, Literal, Node, URIRef
from rdflib.namespace import RDF, RDFS, XSD
from typing_extensions import override

from .constants import (
    ENTITY_TYPE_RDFS_LITERAL,
    EXTENSION_DEFINITIONS,
    MAPPING_SET_ID,
    MAPPINGS,
    NO_TERM_FOUND,
    OBJECT_ID,
    OBJECT_TYPE,
    PREDICATE_ID,
    PREDICATE_MODIFIER,
    PREDICATE_MODIFIER_NOT,
    RECORD_ID,
    SSSOM_URI_PREFIX,
    SUBJECT_ID,
    SUBJECT_TYPE,
    SSSOMSchemaView,
)
from .util import MappingSetDataFrame, sort_df_rows_columns

__all__ = ["MappingSetRDFConverter"]

logging = _logging.getLogger(__name__)

Triple: TypeAlias = tuple[Node, Node, Node]
DictOrSeries: TypeAlias = Union[Dict[str, Any], Series]
MAPPINGS_IRI = URIRef(MAPPINGS, SSSOM_URI_PREFIX)
EXTENSION_DEFINITION_IRI = URIRef(EXTENSION_DEFINITIONS, SSSOM_URI_PREFIX)

CurieConverterProvider: TypeAlias = Callable[[], Converter]
"""A function that can provide a CURIE converter.

We need this contraption because we have to create objects that will
need to use a CURIE converter at some point, but we want to create such
objects _before_ we get the converter -- because the converter to use
will be specific to a given MSDF or a given RDF graph, which is not yet
known at initialisation time.
"""


class ValueConverter:
    """Base class for all value converters.

    A value converter converts a slot value to or from its RDF
    representation.
    """

    def from_rdf(self, obj: Node) -> Any:
        """Convert a RDF node into a SSSOM slot value.

        :param obj: The node to convert.

        :returns: The converted value.

        :raises ValueError: If the node cannot be converted into the
            correct type of value for the slot.
        """
        raise NotImplementedError

    def to_rdf(self, value: Any) -> Node:
        """Convert a SSSOM slot value into a RDF node.

        For multi-valued slots, this method will be called once for
        every value in the slot.

        :param value: The value to convert.

        :returns: A RDF node representing the value.
        """
        raise NotImplementedError


class BaseStringValueConverter(ValueConverter):
    """Converter for all string-based slots."""

    primary_type: URIRef
    allowed_types: Set[URIRef]

    def __init__(
        self,
        primary_type: URIRef = XSD.string,
        allowed_types: Optional[List[URIRef]] = None,
    ):
        """Create a new instance.

        :param primary_type: The datatype used to represent the value in
            RDF context, according to the SSSOM/RDF specification. A
            value of `rdfs:Resource` means the value is represented as
            a named resource (IRI) rather than as a literal.
        :param allowed_types: Additional RDF types that are acceptable
            to represent the value in RDF context.
        """
        self.primary_type = primary_type
        if allowed_types is not None:
            self.allowed_types = set(allowed_types)
        else:
            self.allowed_types = set()

    @override
    def from_rdf(self, obj: Node) -> str:
        """Convert a RDF node into a string-based value."""
        if isinstance(obj, URIRef) and (
            self.primary_type == RDFS.Resource or RDFS.Resource in self.allowed_types
        ):
            return str(obj)
        elif isinstance(obj, Literal):
            # A "naked" literal is a xsd:string literal
            datatype = obj.datatype or XSD.string
            if datatype == self.primary_type or datatype in self.allowed_types:
                return str(obj.value)

        if self.primary_type == RDFS.Resource:
            msg = "Invalid node type (named resource expected)"
        else:
            msg = f"Invalid node type ({self.primary_type} literal expected)"
        raise ValueError(msg)

    @override
    def to_rdf(self, value: str) -> Node:
        """Convert a string-based value into a RDF node."""
        if self.primary_type == RDFS.Resource:
            return URIRef(value)
        elif self.primary_type == XSD.string:
            # Datatype is not needed for a xsd:string
            return Literal(value)
        else:
            return Literal(value, datatype=self.primary_type)


class StringValueConverter(BaseStringValueConverter):
    """Converter for string-typed slots.

    A string-typed slot is quite naturally represented by a string
    literal. However, for compatibility with the LinkML-based loader,
    we also accept a named resource when converting from RDF.
    """

    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__(allowed_types=[RDFS.Resource])


class NonRelativeURIValueConverter(BaseStringValueConverter):
    """Converter for SSSOM URI-typed slots.

    As par the SSSOM/RDF specification, a URI-typed slot is represented
    as a named RDF resource. However when converting from RDF, we also
    accept (1) `xsd:anyURI` literals (as recommended by the spec) and
    (2) `xsd:string` literals (for compatibility with the LinkML-based
    loader).
    """

    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__(primary_type=RDFS.Resource, allowed_types=[XSD.string, XSD.anyURI])


class EntityReferenceValueConverter(BaseStringValueConverter):
    """Converter for EntityReference-typed slots.

    Entity references are represented as named resources in RDF, but we
    also accept string literals for backwards compatibility.

    Importantly, throughout SSSOM-Py entity references are expected to
    be stored in CURIE form, so this converter must take care of
    compressing the values when converting from RDF and conversely
    expanding them when converting to RDF.
    """

    ccp: CurieConverterProvider

    def __init__(self, ccp: CurieConverterProvider):
        """Create a new instance.

        :param ccp: An object that shall provide the CURIE converter to
            use for CURIE expansion/contraction.
        """
        super().__init__(primary_type=RDFS.Resource, allowed_types=[XSD.string])
        self.ccp = ccp

    @override
    def from_rdf(self, obj: Node) -> str:
        """Convert a RDF node into an entity reference value."""
        value = super().from_rdf(obj)
        return self.ccp().compress(value, passthrough=True)

    @override
    def to_rdf(self, value: str) -> Node:
        """Convert an entity reference value into a RDF node."""
        value = self.ccp().expand(value, passthrough=True)
        return super().to_rdf(value)


class DateValueConverter(ValueConverter):
    """Converter for date-typed slots.

    Date-typed slots are represented in RDF as ISO-formatted `xsd:date`
    literals.
    """

    @override
    def from_rdf(self, obj: Node) -> date:
        """Convert a RDF node into a SSSOM date value."""
        if isinstance(obj, Literal) and obj.datatype == XSD.date:
            # RDFLib guarantees us to return a date
            return cast(date, obj.toPython())

        raise ValueError("Invalid node type (xsd:date literal expected)")

    @override
    def to_rdf(self, value: Union[str, date]) -> Node:
        """Convert a SSSOM date value into a RDF node."""
        if isinstance(value, date):
            return Literal(value, datatype=XSD.date)
        else:
            # Let's just hope the value is already in the shape
            # of an ISO-formatted date...
            return Literal(str(value), datatype=XSD.date)


class DoubleValueConverter(ValueConverter):
    """Converter for double-typed slots.

    Double-typed slots are represented in RDF as `xsd:double` literals.
    """

    @override
    def from_rdf(self, obj: Node) -> float:
        """Convert a RDF node into a SSSOM double value."""
        if isinstance(obj, Literal) and obj.datatype == XSD.double:
            # RDFLib guarantees us to return a float
            return cast(float, obj.toPython())

        raise ValueError("Invalid node type (xsd:double expected)")

    @override
    def to_rdf(self, value: float) -> Node:
        """Convert a SSSOM double value into a RDF node."""
        return Literal(value, datatype=XSD.double)


class EnumValueConverter(ValueConverter):
    """Converter for enum-typed slots.

    An enum value is represented in RDF as a named resource if the value
    is defined (in the LinkML model) as having a `meaning` IRI, or as a
    string literal otherwise.

    When parsing, even if the value does have a `meaning` IRI we accept
    both the named resource form and the string literal form.
    """

    allowed_values: Set[str]
    values_by_uri: Dict[URIRef, str]
    uris_by_value: Dict[str, URIRef]

    def __init__(self, schema: SchemaView, name: str):
        """Create a new instance.

        :param schema: The SSSOM LinkML schema.
        :param name: The name of the enum type.
        """
        self.values_by_iri = {}
        self.uris_by_value = {}

        definition = schema.get_enum(name)
        self.allowed_values = set(definition.permissible_values.keys())
        for k, v in definition.permissible_values.items():
            if v.meaning is not None:
                uri = URIRef(schema.expand_curie(v.meaning))
                self.values_by_iri[uri] = k
                self.uris_by_value[k] = uri

    @override
    def from_rdf(self, obj: Node) -> Any:
        """Convert a RDF node into a SSSOM enum value."""
        if isinstance(obj, URIRef):
            value = self.values_by_iri.get(obj)
            if value is None:
                raise ValueError(f"Invalid enum value {obj}")
            return value
        elif isinstance(obj, Literal):
            if obj.datatype is None or obj.datatype == XSD.string:
                if obj.value not in self.allowed_values:
                    raise ValueError(f"Invalid enum value {obj}")
                return obj.value

        raise ValueError("Invalid node type (IRI or string literal expected)")

    @override
    def to_rdf(self, value: Any) -> Node:
        """Convert a SSSOM enum value into a RDF node."""
        # Make sure we have a _text_ value, regardless what the rest of
        # SSSOM-Py can give us...
        value = str(value)
        uri = self.uris_by_value.get(value)
        if uri is not None:
            return uri
        else:
            # We do _not_ check whether the value is a valid value for
            # the enum. If it is not, then something wrong happened
            # upstream of us (a set was constructed or accepted with
            # invalid values) and it's not up to us to deal with the
            # the fallout.
            return Literal(value)


class ValueConverterFactory:
    """Helper object to create value converters."""

    constructors: Dict[str, Type[ValueConverter]]

    def __init__(self) -> None:
        """Create a new instance."""
        self.constructors = {
            "date": DateValueConverter,
            "double": DoubleValueConverter,
            "string": StringValueConverter,
            "ncname": StringValueConverter,
            "EntityReference": EntityReferenceValueConverter,
            "uriorcurie": EntityReferenceValueConverter,
            "NonRelativeURI": NonRelativeURIValueConverter,
        }

    def create(
        self, range_name: str, schema: SchemaView, ccp: CurieConverterProvider
    ) -> Optional[ValueConverter]:
        """Create a new value converter.

        :param range_name: The range for which a value converter is
            wanted.
        :param schema: The SSSOM LinkML schema.
        :param ccp: The object that will provide the CURIE converter to
            use, for the value converters that need one.

        :returns: A suitable value converter for the range. May be None
            if the range is not a scalar range.
        """
        if schema.get_class(range_name) is not None:
            # This range is for objects, not scalar values
            return None

        if range_name.endswith("_enum"):
            return EnumValueConverter(schema, range_name)

        ctor = self.constructors.get(range_name)
        if ctor == EntityReferenceValueConverter:
            # CURIE provider needed
            return EntityReferenceValueConverter(ccp)
        elif ctor is not None:
            return ctor()
        else:
            # This should only happen if a brand new type of slot has
            # been introduced in the SSSOM schema
            raise NotImplementedError(f"Range {range_name} is not supported")


class ObjectConverter:
    """Base class for conversion of SSSOM objects to and from RDF.

    One instance of this class will handle the (de)serialisation of one
    particular type of object (e.g. MappingSet, Mapping).

    The base class implements the logic that is common to all types of
    objects. It should be subclassed for objects that require specific
    logic.
    """

    name: str
    """The name of the SSSOM class whose conversion is handled by this
    object."""

    object_uri: URIRef
    """The URI representing the SSSOM class in RDF context."""

    slots_by_name: Dict[str, SlotDefinition]
    """All the valid slots for the class, addressed by their names."""

    slots_by_uri: Dict[URIRef, SlotDefinition]
    """All the valid slots for the class, addressed by the URIs that
    represent them in RDF context."""

    value_converters: Dict[str, ValueConverter]
    """All the value converters used to convert slot values to/from RDF,
    addressed by slot range."""

    schema: SSSOMSchemaView
    """Helper object to access information from the SSSOM schema."""

    ccp: CurieConverterProvider
    """The object to call when CURIE expansion/contraction is needed."""

    def __init__(self, class_name: str, ccp: CurieConverterProvider):
        """Create a new instance for a class of objects.

        :param class_name: The name of the SSSOM class of objects to
            convert.
        :param ccp: A callable object that shall provide a CURIE
            converter on demand.
        """
        self.schema = SSSOMSchemaView()
        self.ccp = ccp

        # Prepare the slot tables...
        self.slots_by_name = {}
        self.slots_by_uri = {}
        ranges: List[str] = []
        for slot in self.schema.view.class_induced_slots(class_name):
            self.slots_by_name[slot.name] = slot
            self.slots_by_uri[self._get_slot_uri(slot)] = slot
            ranges.append(slot.range)

        # ... and the scalar converters table
        self.value_converters = {}
        factory = ValueConverterFactory()
        for rng in set(ranges):
            vc = factory.create(rng, self.schema.view, ccp)
            if vc is not None:
                self.value_converters[rng] = vc

        # The name and URI of the class
        self.name = self._fix_class_name(class_name)
        object_class = self.schema.view.get_class(class_name)
        if object_class.class_uri is not None:
            self.object_uri = URIRef(self.schema.view.expand_curie(object_class.class_uri))
        else:
            self.object_uri = URIRef(self.name, SSSOM_URI_PREFIX)

    #
    # Conversion from RDF
    #

    def dict_from_rdf(
        self, graph: Graph, subject: Node, dest: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Parse a SSSOM object from a RDF node.

        Given a RDF node representing a SSSOM object, this method
        constructs a dictionary from all the triples where the node is
        the subject.

        :param graph: The graph to parse the object from.
        :param subject: The root node of the object to extract.
        :param dest: The dictionary into which the parsed object will be
            stored, if specified. Client code may use this argument to
            provide a "pre-filled" dictionary with default values. If
            not provided, a new empty dictionary will be used instead.

        :returns: A dictionary representing the SSSOM object, where keys
            are the slot names. This will be the same dictionary as the
            `dest` argument, if present.

        :raises ValueError: If the contents of the RDF graph does not
            represent a valid SSSOM object.
        """
        if dest is None:
            dest = {}
        self._init_dict_from_rdf(graph, subject, dest)

        for pred, obj in graph.predicate_objects(subject):
            pred = cast(URIRef, pred)
            if pred == RDF.type:
                if obj != self.object_uri:
                    raise ValueError(f"Invalid type {obj} for a {self.name} object")
                continue

            if self._process_triple(graph, cast(Triple, [subject, pred, obj]), dest):
                continue

            slot = self.slots_by_uri.get(pred)
            if slot is not None:
                try:
                    value = self._get_value_converter(slot).from_rdf(obj)
                except ValueError as e:
                    raise ValueError(f"Invalid value for {slot.name}", e)
                if slot.multivalued:
                    self._multivalue_from_rdf(value, slot.name, dest)
                else:
                    dest[slot.name] = value
            elif not self._extension_from_rdf(cast(Triple, [subject, pred, obj]), dest):
                logging.warning(f"Ignoring unexpected triple {subject} {pred} {obj}")

        return dest

    # Helper methods for conversion from RDF

    def _init_dict_from_rdf(self, graph: Graph, subject: Node, dest: Dict[str, Any]) -> None:
        """Initialize a dictionary representing a parsed object.

        This method is used to check that the subject node is of a
        suitable type for the object to be parsed. Subclasses may
        override it to alter the check as needed and also to perform
        any additional specific operations at the beginning of the
        conversion.

        :param graph: The graph to parse the object from.
        :param subject: The root node of the object to extract.
        :param dest: The dictionary in which to store the parsed object.

        :raises ValueError: If the provided root node is not suitable
            for the type of object to extract.
        """
        if not isinstance(subject, BNode):
            raise ValueError(f"Invalid node type for a {self.name} object")

    def _process_triple(self, graph: Graph, triple: Triple, dest: Dict[str, Any]) -> bool:
        """Process an individual triple associated with a SSSOM object.

        This method is intended to allow subclasses to implement custom
        processing of some RDF triples, before letting the triple being
        handled by the common logic in the `dict_from_rdf` method.

        :param graph: The graph to parse the object from.
        :param triple: The triple to process.
        :param dest: The dictionary in which to store the parsed object.

        :returns: True if the triple has been processed successfully, or
            False to let the triple be handed by the common logic.
        """
        return False

    def _extension_from_rdf(self, triple: Triple, dest: Dict[str, Any]) -> bool:
        """Process a triple that may represent an extension slot.

        :param triple: The triple to process.
        :param dest: The dictionary in which to store the parsed object.

        :returns: True if the triple does represent a valid extension
            slot, False otherwise.
        """
        # FIXME: Not implemented yet, ignore all for now
        logging.warning(f"Ignoring possible extension slot {triple[1]}")
        return False

    def _multivalue_from_rdf(self, value: Any, slot_name: str, dest: Dict[str, Any]) -> None:
        """Store a single value into a multi-valued slot.

        Subclasses should override this method if the object for which
        they are intended does not represent multi-valued slots as a
        simple list.

        :param value: The value to store.
        :param slot_name: The name of the slot the value is for.
        :param dest: The dictionary in which to store the parsed object.
        """
        if slot_name not in dest:
            dest[slot_name] = [value]
        else:
            dest[slot_name].append(value)

    #
    # Conversion to RDF
    #

    def dict_to_rdf(self, graph: Graph, obj: DictOrSeries) -> Node:
        """Export a SSSOM object to a RDF graph.

        :param graph: The graph to export the object to.
        :param obj: The dictionary representing the SSSOM object to
            export.

        :returns: The root node representing the exporting object.
        """
        subject = self._init_dict_to_rdf(graph, obj)
        graph.add(cast(Triple, [subject, RDF.type, self.object_uri]))

        for k, v in obj.items():
            key = str(k)
            if self._process_slot(graph, subject, key, v):
                continue

            slot = self.slots_by_name.get(key)
            if slot is not None:
                pred = self._get_slot_uri(slot)
                converter = self._get_value_converter(slot)
                if slot.multivalued:
                    for value in self._get_multi_values(v):
                        if not self._is_empty(value):
                            o = converter.to_rdf(value)
                            graph.add(cast(Triple, [subject, pred, o]))
                else:
                    if not self._is_empty(v):
                        graph.add(cast(Triple, [subject, pred, converter.to_rdf(v)]))
            elif not self._extension_to_rdf(graph, subject, key, v):
                logging.warning(f"Ignoring unexpected {key}={v} slot")

        return subject

    # Helper methods for conversion to RDF

    def _is_empty(self, value: Any) -> bool:
        """Check is value is an empty value.

        This method is mostly a hack to cope with the fact that "empty"
        values are not properly represented as such in the data frame
        part of a MSDF (apart for double-typed slots).

        :param value: The value to check.

        :returns: True if the value is empty.
        """
        return value is None or (hasattr(value, "__len__") and len(value) == 0)

    def _init_dict_to_rdf(self, graph: Graph, obj: DictOrSeries) -> Node:
        """Create the root node representing a SSSOM object.

        Subclasses should override this method to customize the way
        their SSSOM object is represented in RDF. The default behaviour
        is to represent the object as a blank node.

        :param graph: The graph the object is exported to.
        :param obj: The dictionary representing the SSSOM object to
            export.

        :returns: The node representing the exported object.
        """
        return BNode()

    def _process_slot(self, graph: Graph, subject: Node, name: str, value: Any) -> bool:
        """Process an individual slot for RDF export.

        This method is intended to allow subclasses to implement custom
        processing to export some SSSOM slots, before letting the slot
        being handled by the common logic in the `dict_to_rdf` method.

        :param graph: The graph the object is exported to.
        :param subject: The node representing the exported object.
        :param name: The name of the slot to process.
        :param value: The value of the slot:

        :returns: True if the slot has been processed successfully, or
            False to let the slot be handled by the common logic.
        """
        return False

    def _extension_to_rdf(self, graph: Graph, subject: Node, name: str, value: Any) -> bool:
        """Export an extension slot to RDF.

        :param graph: The graph to export to.
        :param subject: The node representing the exported object.
        :param name: The name of the extension slot.
        :param value: The value of the extension slot.

        :returns: True if `name` is the name of a valid extension,
            otherwise False.
        """
        # FIXME: Not implemented yet, ignore all for now
        logging.warning(f"Ignoring extension slot {name}")
        return False

    def _get_multi_values(self, value: Any) -> List[Any]:
        """Get the values of a multi valued slot as a list.

        Subclasses should override this method if their object does not
        represent multi-valued slots as a simple list.

        :param value: The object representing the value of a multivalued
            slot.

        :returns: The list of values.
        """
        if not isinstance(value, list):
            # Should not really happen (the value of a multi-valued slot
            # should always be a list, even if it has only one value),
            # but just in case
            return [value]
        else:
            return value

    #
    # Other helper methods
    #

    def _get_value_converter(self, slot: SlotDefinition) -> ValueConverter:
        """Get the value converter for a given slot.

        :param slot: The slot for which to convert value.

        :returns: A value converter suitable to convert a RDF node into
            a SSSOM slot value, or a SSSOM slot value into a RDF node.
        """
        converter = self.value_converters.get(slot.range)
        if converter is None:
            # This should have been caught already at init time by the
            # ValueConverterFactory
            raise NotImplementedError(f"Unsupported range {slot.range} for {slot.name}")
        return converter

    def _get_slot_uri(self, slot: SlotDefinition) -> URIRef:
        """Get the URI that represents a SSSOM slot.

        A SSSOM slot is represented by a property whose URI is either
        explicitly defined in the LinkML model (in which case it is
        available in the `slot_uri` field of the slot definition), or
        constructed as `sssom:slot_name`.

        :param slot: The slot definition.

        :returns: The URI that represents the slot in RDF context.
        """
        if slot.slot_uri is not None:
            return URIRef(self.schema.view.expand_curie(slot.slot_uri))
        else:
            return URIRef(slot.name, SSSOM_URI_PREFIX)

    def _fix_class_name(self, name: str) -> str:
        """Transform a LinkML class name.

        The SSSOM LinkML schema defines its classes with lower-case,
        space-containing names, which must be transformed into
        Pascal-cased names (e.g. `mapping set` becomes `MappingSet`).
        There does not seem to be a way to obtain the transformed name
        from the SSSOMSchemaView object (or any other LinkML object),
        so we do the transformation here.

        :param name: The original name of a LinkML-defined class.

        :returns: The Pascal-case form of the name.
        """
        fixed: List[str] = []
        upper = True
        for _, c in enumerate(name):
            if c == " ":
                upper = True
            elif upper:
                fixed.append(c.upper())
                upper = False
            else:
                fixed.append(c)
        return "".join(fixed)


class MappingSetRDFConverter(ObjectConverter):
    """Helper class to convert mapping sets to/from RDF.

    Use this class to deserialise a mapping set from a RDF graph into a
    MappingSetDataFrame object::

        from rdflib import Graph

        g = Graph().parse("my_set.sssom.ttl")
        rdf_converter = MappingSetRDFConverter()
        msdf = rdf_converter.msdf_from_rdf(g)

    or to serialise a MappingSetDataFrame into a RDF graph::

        g = rdf_converter().msdf_to_rdf(msdf)
        g.serialize("my_set.sssom.ttl")

    """

    mapping_converter: ObjectConverter
    extension_definition_converter: ObjectConverter
    curie_converter: Converter
    hydrate: bool

    def __init__(self, hydrate: bool = False):
        """Create a new instance.

        :param hydrate: Default value for the `hydrate` parameter of the
            `msdf_to_rdf` method.
        """
        super().__init__("mapping set", self.get_curie_converter)
        self.mapping_converter = MappingConverter(self.get_curie_converter)
        self.extension_definition_converter = ObjectConverter(
            "extension definition", self.get_curie_converter
        )
        self.hydrate = hydrate
        self.curie_converter = Converter()

    def get_curie_converter(self) -> Converter:
        """Provide the current CURIE converter."""
        return self.curie_converter

    #
    # Conversion from RDF
    #

    def msdf_from_rdf(
        self,
        graph: Graph,
        curie_converter: Optional[Converter] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MappingSetDataFrame:
        """Extract a MappingSetDataFrame from a RDF graph.

        This is the main method intended for use by client code for RDF
        parsing.

        :param graph: The graph to parse from.
        :param curie_converter: The CURIE converter to use for all CURIE
            expansion or contraction operations. If not specified, a
            converter based on the namespaces declared within the graph
            will be used instead. This is the converter that will be
            associated with the returned MappingSetDataFrame object.
        :param meta: Default metadata to use, if any.

        :returns: The first MappingSetDataFrame found in the graph.
        """
        if curie_converter is None:
            curie_converter = Converter.from_rdflib(graph)
        self.curie_converter = curie_converter

        sets = [s for s in graph.subjects(RDF.type, self.object_uri)]
        if len(sets) == 0:
            raise Exception("No mapping set in graph")
        elif len(sets) > 1:
            logging.warning("More than one mapping set in graph, ignoring supernumerary sets")

        meta = self.dict_from_rdf(graph, sets[0], dest=meta)
        if MAPPINGS in meta:
            # dict_from_rdf returns a dictionary containing everything,
            # including the mappings. We must take the mappings out and
            # turn them into a data frame instead.
            mappings = meta.pop(MAPPINGS)
            df = sort_df_rows_columns(DataFrame(m for m in mappings))
        else:
            # Empty set?
            df = DataFrame()

        return MappingSetDataFrame.with_converter(df=df, metadata=meta, converter=curie_converter)

    @override
    def _init_dict_from_rdf(self, graph: Graph, subject: Node, dest: Dict[str, Any]) -> None:
        # A mapping set can (and in fact *should*) be represented by a
        # named resource, which is then interpreted as the value of the
        # MAPPING_SET_ID slot.
        if isinstance(subject, URIRef):
            dest[MAPPING_SET_ID] = str(subject)
        elif not isinstance(subject, BNode):
            raise ValueError("Invalid node type for a MappingSet object")

        # Extension definitions should be processed early on, so that we
        # have them at hand when we'll process the individual triples.
        extension_definitions: List[Dict[str, Any]] = []
        for ed_node in graph.objects(subject, EXTENSION_DEFINITION_IRI):
            ed = self.extension_definition_converter.dict_from_rdf(graph, ed_node)
            extension_definitions.append(ed)
        if len(extension_definitions) > 0:
            dest[EXTENSION_DEFINITIONS] = extension_definitions

    @override
    def _process_triple(self, graph: Graph, triple: Triple, dest: Dict[str, Any]) -> bool:
        done = False
        if triple[1] == MAPPINGS_IRI:
            mapping = self.mapping_converter.dict_from_rdf(graph, triple[2])
            if MAPPINGS not in dest:
                dest[MAPPINGS] = [mapping]
            else:
                dest[MAPPINGS].append(mapping)
            done = True
        elif triple[1] == EXTENSION_DEFINITION_IRI:
            # Already dealt with in the initialisation step
            done = True

        return done

    #
    # Conversion to RDF
    #

    def msdf_to_rdf(
        self,
        msdf: MappingSetDataFrame,
        graph: Optional[Graph] = None,
        curie_converter: Optional[Converter] = None,
        hydrate: Optional[bool] = None,
    ) -> Graph:
        """Export a MappingSetDataFrame into a RDF graph.

        This is the main method intended for use by client code for RDF
        export.

        :param msdf: The MappingSetDataFrame to export.
        :param graph: The graph to export to. If not given a new graph
            will be created.
        :param curie_converter: The CURIE converter to use for all CURIE
            expansion or contraction operations. If not given, the
            converter bound to the MappingSetDataFrame will be used
            instead.
        :param hydrate: Whether to generate "direct triples" for each
            mapping in the MappingSetDataFrame.

        :returns: The RDF graph with the exported MappingSetDataFrame.
            This will be the same object as the `graph` argument, if
            given.
        """
        if graph is None:
            graph = Graph()
        self.curie_converter = curie_converter or msdf.converter
        for k, v in self.curie_converter.bimap.items():
            graph.namespace_manager.bind(k, v)
        if hydrate is None:
            hydrate = self.hydrate
        ms_node = self.dict_to_rdf(graph, msdf.metadata)
        for _, row in msdf.df.iterrows():
            self._mapping_to_rdf(graph, ms_node, row, hydrate)

        return graph

    @override
    def _init_dict_to_rdf(self, graph: Graph, obj: DictOrSeries) -> Node:
        if MAPPING_SET_ID in obj:
            return URIRef(obj[MAPPING_SET_ID])
        else:
            return BNode()

    @override
    def _process_slot(self, graph: Graph, subject: Node, name: str, value: Any) -> bool:
        done = False
        if name == MAPPING_SET_ID:
            # Already dealt with in the initialisation step
            done = True
        elif name == EXTENSION_DEFINITIONS:
            for ed in value:
                ed_node = self.extension_definition_converter.dict_to_rdf(graph, ed)
                graph.add(cast(Triple, [subject, EXTENSION_DEFINITION_IRI, ed_node]))
            done = True
        elif name == MAPPINGS:
            # We don't really expect to have any mappings here, as the
            # dict_to_rdf method is supposed to be called with only the
            # metadata part of a MappingSetDataFrame. Still, we cover
            # possibility, just in case.
            for mapping in value:
                self._mapping_to_rdf(graph, subject, mapping, self.hydrate)
            done = True

        return done

    def _mapping_to_rdf(
        self, graph: Graph, subject: Node, mapping: DictOrSeries, hydrate: bool
    ) -> None:
        mapping_node = self.mapping_converter.dict_to_rdf(graph, mapping)
        graph.add(cast(Triple, [subject, MAPPINGS_IRI, mapping_node]))

        if hydrate:
            subject_id = mapping.get(SUBJECT_ID)
            predicate_id = mapping.get(PREDICATE_ID)
            object_id = mapping.get(OBJECT_ID)

            if (
                mapping.get(SUBJECT_TYPE) != ENTITY_TYPE_RDFS_LITERAL
                and mapping.get(OBJECT_TYPE) != ENTITY_TYPE_RDFS_LITERAL
                and mapping.get(PREDICATE_MODIFIER) != PREDICATE_MODIFIER_NOT
                and subject_id is not None
                and subject_id != NO_TERM_FOUND
                and object_id is not None
                and object_id != NO_TERM_FOUND
                and predicate_id is not None
            ):
                subject_ref = URIRef(self.curie_converter.expand(subject_id))
                pred_ref = URIRef(self.curie_converter.expand(predicate_id))
                object_ref = URIRef(self.curie_converter.expand(object_id))
                graph.add(cast(Triple, [subject_ref, pred_ref, object_ref]))


class MappingConverter(ObjectConverter):
    """Specialised class to (de)serialise Mapping objects."""

    def __init__(self, ccp: CurieConverterProvider) -> None:
        """Create a new instance."""
        super().__init__("mapping", ccp)

    @override
    def _init_dict_from_rdf(self, graph: Graph, subject: Node, dest: Dict[str, Any]) -> None:
        # If the root node is a named resource, then it is interpreted
        # as the RECORD_ID for the mapping.
        if isinstance(subject, URIRef):
            dest[RECORD_ID] = self.ccp().compress(str(subject), passthrough=True)
        elif not isinstance(subject, BNode):
            raise ValueError(f"Invalid node type for a {self.name} object")

    @override
    def _multivalue_from_rdf(self, value: Any, slot_name: str, dest: Dict[str, Any]) -> None:
        # Within the data frame part of a MappingSetDataFrame, a
        # multivalued slot is expected to be represented as a single
        # string containing |-separated values.
        if slot_name not in dest:
            dest[slot_name] = value
        else:
            dest[slot_name] = dest[slot_name] + "|" + value

    @override
    def _init_dict_to_rdf(self, graph: Graph, obj: DictOrSeries) -> Node:
        if RECORD_ID in obj:
            return URIRef(self.ccp().expand(obj[RECORD_ID]))
        else:
            return BNode()

    @override
    def _get_multi_values(self, value: Any) -> List[Any]:
        return cast(List[Any], value.split("|"))
