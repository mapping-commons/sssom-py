"""Internal code for RDF import and export."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Set, Union, cast

from curies import Converter
from linkml_runtime.linkml_model.meta import SlotDefinition
from linkml_runtime.utils.schemaview import SchemaView
from pandas import DataFrame
from rdflib import BNode, Graph, Literal, Node, URIRef
from rdflib.namespace import RDF, XSD
from sssom_schema import EntityReference
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
    MetadataType,
    SSSOMSchemaView,
)
from .util import MappingSetDataFrame, sort_df_rows_columns

TRIPLE = tuple[Node, Node, Node]
MAPPINGS_IRI = URIRef(MAPPINGS, SSSOM_URI_PREFIX)
EXTENSION_DEFINITION_IRI = URIRef(EXTENSION_DEFINITIONS, SSSOM_URI_PREFIX)


class ValueConverter(object):
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


class StringValueConverter(ValueConverter):
    """Converter for string-typed slots.

    A string-typed slot is quite naturally represented by a string
    literal. Howver, for compatibility with the LinkML-based loader,
    we also accept a named resource when converting from RDF.
    """

    @override
    def from_rdf(self, obj: Node) -> str:
        """Convert a RDF node into a SSSOM string value."""
        if isinstance(obj, URIRef):
            return str(obj)
        elif isinstance(obj, Literal):
            if obj.datatype is None or obj.datatype == XSD.string:
                return str(obj.value)

        raise ValueError("Invalid node type (string literal expected)")

    @override
    def to_rdf(self, value: str) -> Node:
        """Convert a SSSOM string value into a RDF node."""
        return Literal(str(value))


class NonRelativeURIValueConverter(ValueConverter):
    """Converter for SSSOM URI-typed slots.

    As par the SSSOM/RDF specification, a URI-typed slot is represented
    as a named RDF resource. However when converting from RDF, we also
    accept (1) `xsd:anyURI` literals (as recommended by the spec) and
    (2) `xsd:string` literals and "naked" literals (for compatibility
    with the LinkML-based loader).
    """

    @override
    def from_rdf(self, obj: Node) -> str:
        """Convert a RDF node into a SSSOM URI value."""
        if isinstance(obj, URIRef):
            return str(obj)
        elif isinstance(obj, Literal):
            if obj.datatype is None or (obj.datatype == XSD.string or obj.datatype == XSD.anyURI):
                return str(obj.value)

        raise ValueError("Invalid node type (xsd:anyURI literal expected)")

    @override
    def to_rdf(self, value: str) -> Node:
        """Convert a SSSOM URI value into a RDF node."""
        return URIRef(str(value))


class EntityReferenceValueConverter(ValueConverter):
    """Converter for EntityReference-typed slots.

    Entity references are represented as named resources in RDF, but we
    also accept string literals for backwards compatibility.

    Importantly, throughout SSSOM-Py entity references are expected to
    be stored in CURIE form, so this converter must take care of
    compressing the values when converting from RDF and conversely
    expanding them when converting to RDF.
    """

    prefix_manager: Converter

    def __init__(self, prefix_manager: Converter):
        """Create a new instance.

        :param prefix_manager: The CURIEs converter to use for
            expanding and compressing entity references.
        """
        self.prefix_manager = prefix_manager

    @override
    def from_rdf(self, obj: Node) -> str:
        """Convert a RDF node into a SSSOM entity reference value."""
        if isinstance(obj, URIRef):
            # Pass-through because we should probably not assume
            # that the CURIE converter will know how to compress
            # every single IRI found in the graph
            return self.prefix_manager.compress(str(obj), passthrough=True)
        elif isinstance(obj, Literal):
            if obj.datatype is None or obj.datatype == XSD.string:
                return self.prefix_manager.compress(obj.value, passthrough=True)

        raise ValueError("Invalid node type (IRI expected)")

    @override
    def to_rdf(self, value: Union[str, EntityReference]) -> Node:
        """Convert a SSSOM entity reference value into a RDF node."""
        # Pass-through because even though all entity references in a
        # MappingSetDataFrame should really be in CURIE form, it happens
        # frequently that they are not -- including in some of our own
        # test cases. :(
        return URIRef(self.prefix_manager.expand(str(value), passthrough=True))


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

        :param enum: The class that implements the enum type to convert
            to and from.
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


class ObjectConverter(object):
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

    curie_converter: Converter
    """The CURIE converter to use for CURIE expansion/contraction
    throughout the (de)serialisation process."""

    schema: SSSOMSchemaView
    """Helper object to access information from the SSSOM schema."""

    def __init__(self, class_name: str, curie_converter: Converter):
        """Create a new instance for a class of objects.

        :param class_name: The name of the SSSOM class of objects to
            convert.
        :param curie_converter: The CURIE converter to use to
            expand/contract CURIEs when converting to/from RDF.
        """
        self.schema = SSSOMSchemaView()
        self.curie_converter = curie_converter

        str_value_converter = StringValueConverter()
        er_value_converter = EntityReferenceValueConverter(curie_converter)
        self.value_converters = {
            "string": str_value_converter,
            "ncname": str_value_converter,
            "EntityReference": er_value_converter,
            "uriorcurie": er_value_converter,
            "NonRelativeURI": NonRelativeURIValueConverter(),
            "date": DateValueConverter(),
            "double": DoubleValueConverter(),
            "entity_type_enum": EnumValueConverter(self.schema.view, "entity_type_enum"),
            "sssom_version_enum": EnumValueConverter(self.schema.view, "sssom_version_enum"),
            "mapping_cardinality_enum": EnumValueConverter(
                self.schema.view, "mapping_cardinality_enum"
            ),
            "predicate_modifier_enum": EnumValueConverter(
                self.schema.view, "predicate_modifier_enum"
            ),
        }

        self.slots_by_name = {}
        self.slots_by_uri = {}
        for slot in self.schema.view.class_induced_slots(class_name):
            self.slots_by_name[slot.name] = slot
            self.slots_by_uri[self._get_slot_uri(slot)] = slot

        self.name = self._fix_class_name(class_name)
        object_class = self.schema.view.get_class(class_name)
        if object_class.class_uri is not None:
            self.object_uri = URIRef(self.schema.view.expand_curie(object_class.class_uri))
        else:
            self.object_uri = URIRef(self.name, SSSOM_URI_PREFIX)

    # Methods for conversion from RDF

    def dict_from_rdf(self, g: Graph, subject: Node) -> Dict[str, Any]:
        """Parse a SSSOM object from a RDF node.

        Given a RDF node representing a SSSOM object, this method
        constructs a dictionary from all the triples where the node is
        the subject.

        :param g: The graph to parse the object from.
        :param subject: The root node of the object to extract.

        :returns: A dictionary representing the SSSOM object, where keys
            are the slot names.

        :raises ValueError: If the contents of the RDF graph does not
            represent a valid SSSOM object.
        """
        dest = self._init_dict_from_rdf(g, subject)

        for pred, obj in g.predicate_objects(subject):
            pred = cast(URIRef, pred)
            if pred == RDF.type:
                if obj != self.object_uri:
                    raise ValueError(f"Invalid type {obj} for a {self.name} object")
                continue

            if self._process_triple(g, cast(TRIPLE, [subject, pred, obj]), dest):
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
            elif not self._extension_from_rdf(cast(TRIPLE, [subject, pred, obj]), dest):
                logging.warning(f"Ignoring unexpected triple {subject} {pred} {obj}")

        return dest

    def _init_dict_from_rdf(self, g: Graph, subject: Node) -> Dict[str, Any]:
        """Initialize a dictionary representing a parsed object.

        Subclasses should override this method to adapt it for the type
        of objects they are intended for. The default behaviour as
        implemented here is to expect any object to be represented by a
        blank node.

        :param g: The graph to parse the object from.
        :param subject: The root node of the object to extract.

        :returns: A dictionary that can be filled to represent the SSSOM
            object.

        :raises ValueError: If the provided root node is not suitable
            for the type of object to extract.
        """
        if not isinstance(subject, BNode):
            raise ValueError(f"Invalid node type for a {self.name} object")
        return {}

    def _process_triple(self, g: Graph, triple: TRIPLE, dest: Dict[str, Any]) -> bool:
        """Process an individual triple associated with a SSSOM object.

        This method is intended to allow subclasses to implement custom
        processing of some RDF triples, before letting the triple being
        handled by the common logic in the `dict_from_rdf` method.

        :param g: The graph to parse the object from.
        :param triple: The triple to process.
        :param dest: The dictionary representing the extracted object.

        :returns: True if the triple has been processed successfully, or
            False to let the triple be handed by the common logic.
        """
        return False

    def _extension_from_rdf(self, triple: TRIPLE, dest: Dict[str, Any]) -> bool:
        """Process a triple that may represent an extension slot.

        :param triple: The triple to process.
        :param dest: The dictionary representing the extracted object.

        :returns: True if the triple does represent a valid extension
            slot, False otherwise.
        """
        # FIXME: Not implemented yet, ignore all for now
        return False

    def _multivalue_from_rdf(self, value: Any, slot_name: str, dest: Dict[str, Any]) -> None:
        """Store a single value into a multi-valued slot.

        Subclasses should override this method if the object for which
        they are intended does not represent multi-valued slots as a
        simple list.

        :param value: The value to store.
        :param slot_name: The name of the slot the value is for.
        :param dest: The dictionary representing the extracted object.
        """
        if slot_name not in dest:
            dest[slot_name] = [value]
        else:
            dest[slot_name].append(value)

    # Methods for conversion to RDF

    def dict_to_rdf(self, g: Graph, obj: Dict[str, Any]) -> Node:
        """Export a SSSOM object to a RDF graph.

        :param g: The graph to export the object to.
        :param obj: The dictionary representing the SSSOM object to
            export.

        :returns: The root node representing the exporting object.
        """
        subject = self._init_dict_to_rdf(g, obj)
        g.add(cast(TRIPLE, [subject, RDF.type, self.object_uri]))

        for k, v in obj.items():
            if self._process_slot(g, subject, k, v):
                continue

            slot = self.slots_by_name.get(k)
            if slot is not None:
                pred = self._get_slot_uri(slot)
                converter = self._get_value_converter(slot)
                if slot.multivalued:
                    for value in self._get_multi_values(v):
                        if not self._is_null_like(value):
                            o = converter.to_rdf(value)
                            g.add(cast(TRIPLE, [subject, pred, o]))
                else:
                    if not self._is_null_like(v):
                        g.add(cast(TRIPLE, [subject, pred, converter.to_rdf(v)]))
            elif not self._extension_to_rdf(g, subject, k, v):
                logging.warning(f"Ignoring unexpected {k}={v} slot")

        return subject

    def _is_null_like(self, value: Any) -> bool:
        return value is None or (hasattr(value, "__len__") and len(value) == 0)

    def _init_dict_to_rdf(self, g: Graph, obj: Dict[str, Any]) -> Node:
        """Create the root node representing a SSSOM object.

        Subclasses should override this method to customize the way
        their SSSOM object is represented in RDF. The default bhaviour
        is to represent the object as a blank node.
        """
        return BNode()

    def _process_slot(self, g: Graph, subject: Node, name: str, value: Any) -> bool:
        """Process an individual slot for RDF export.

        This method is intended to allow subclasses to implement custom
        processing to export some SSSOM slots, before letting the slot
        being handled by the common logic in the `dict_to_rdf` method.

        :param g: The graph the object is exported to.
        :param subject: The node representing the exported object.
        :param name: The name of the slot to process.
        :param value: The value of the slot:

        :returns: True if the slot has been processed successfully, or
            False to let the slot be handled by the common logic.
        """
        return False

    def _extension_to_rdf(self, g: Graph, subject: Node, name: str, value: Any) -> bool:
        """Export an extension slot to RDF.

        :param g: The graph to export to.
        :param subject: The node representing the exported object.
        :param name: The name of the extension slot.
        :param value: The value of the extension slot.

        :returns: True if `name` is the name of a valid extension,
            otherwise False.
        """
        # FIXME: Not implemented yet, ignore all for now
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

    # Helper methods

    def _get_value_converter(self, slot: SlotDefinition) -> ValueConverter:
        """Get the value converter for a given slot.

        :param slot: The slot for which to convert value.

        :returns: A value converter suitable to convert a RDF node into
            a SSSOM slot value, or a SSSOM slot value into a RDF node.
        """
        converter = self.value_converters.get(slot.range)
        if converter is None:
            # This should only happen if a brand new type of slot has
            # been introduced in the SSSOM schema
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
    """Specialised class to (de)serialise MappingSet objects."""

    mapping_converter: ObjectConverter
    extension_definition_converter: ObjectConverter
    hydrate: bool
    default_metadata: Optional[MetadataType]

    def __init__(self, curie_converter: Converter):
        """Create a new instance.

        :param curie_converter: The CURIE converter to use throughout
            all (de)serialisation operations. When deserialising, it
            would be typically expected that the CURIE converter had
            been previously obtained from the very graph from which a
            set is to be deserialised, but any converter can be used.
            Likewise, when serialising, the converter would typically
            come from the MappingSetDataFrame that is to be serialised.
        """
        super().__init__("mapping set", curie_converter)
        self.mapping_converter = MappingConverter(curie_converter)
        self.extension_definition_converter = ObjectConverter(
            "extension definition", curie_converter
        )
        self.hydrate = False
        self.default_metadata = None

    @override
    def _init_dict_from_rdf(self, g: Graph, subject: Node) -> Dict[str, Any]:
        dest: Dict[str, Any] = {}

        if self.default_metadata is not None:
            dest.update(self.default_metadata)

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
        for ed_node in g.objects(subject, EXTENSION_DEFINITION_IRI):
            ed = self.extension_definition_converter.dict_from_rdf(g, ed_node)
            extension_definitions.append(ed)
        if len(extension_definitions) > 0:
            dest[EXTENSION_DEFINITIONS] = extension_definitions

        return dest

    @override
    def _process_triple(self, g: Graph, triple: TRIPLE, dest: Dict[str, Any]) -> bool:
        done = False
        if triple[1] == MAPPINGS_IRI:
            mapping = self.mapping_converter.dict_from_rdf(g, triple[2])
            if MAPPINGS not in dest:
                dest[MAPPINGS] = [mapping]
            else:
                dest[MAPPINGS].append(mapping)
            done = True
        elif triple[1] == EXTENSION_DEFINITION_IRI:
            # Already dealt with in the initialisation step
            done = True

        return done

    @override
    def _init_dict_to_rdf(self, g: Graph, obj: Dict[str, Any]) -> Node:
        if MAPPING_SET_ID in obj:
            return URIRef(obj[MAPPING_SET_ID])
        else:
            return BNode()

    @override
    def _process_slot(self, g: Graph, subject: Node, name: str, value: Any) -> bool:
        done = False
        if name == MAPPING_SET_ID:
            # Already dealt with in the initialisation step
            done = True
        elif name == EXTENSION_DEFINITIONS:
            for ed in value:
                ed_node = self.extension_definition_converter.dict_to_rdf(g, ed)
                g.add(cast(TRIPLE, [subject, EXTENSION_DEFINITION_IRI, ed_node]))
            done = True
        elif name == MAPPINGS:
            # We don't really expect to have any mappings here, as the
            # dict_to_rdf method is supposed to be called with only the
            # metadata part of a MappingSetDataFrame. Still, we cover
            # possibility, just in case.
            for mapping in value:
                self._mapping_to_rdf(g, subject, mapping)
            done = True

        return done

    def _mapping_to_rdf(self, g: Graph, subject: Node, mapping: Dict[str, Any]) -> None:
        mapping_node = self.mapping_converter.dict_to_rdf(g, mapping)
        g.add(cast(TRIPLE, [subject, MAPPINGS_IRI, mapping_node]))

        if self.hydrate:
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
                g.add(cast(TRIPLE, [subject_ref, pred_ref, object_ref]))

    def msdf_from_rdf(self, g: Graph) -> MappingSetDataFrame:
        """Extract a MappingSetDataFrame from a RDF graph.

        :param g: The graph to extract from.

        :returns: The first MappingSetDataFrame found in the graph.
        """
        sets = [s for s in g.subjects(RDF.type, self.object_uri)]
        if len(sets) == 0:
            raise Exception("No mapping set in graph")
        elif len(sets) > 1:
            logging.warning("More than one mapping set in graph, ignoring supernumerary sets")

        meta = self.dict_from_rdf(g, sets[0])
        if MAPPINGS in meta:
            # dict_from_rdf returns a dictionary containing everything,
            # including the mappings. We must take the mappings out and
            # turn them into a data frame instead.
            mappings = meta.pop(MAPPINGS)
            df = sort_df_rows_columns(DataFrame(m for m in mappings))
        else:
            # Empty set?
            df = DataFrame()

        return MappingSetDataFrame.with_converter(
            df=df, metadata=meta, converter=self.curie_converter
        )

    def msdf_to_rdf(self, g: Graph, msdf: MappingSetDataFrame) -> None:
        """Export a MappingSetDataFrame into a RDF graph.

        :param g: The graph to export to.
        :param msdf: The MappingSetDataFrame to export.
        """
        ms_node = self.dict_to_rdf(g, msdf.metadata)
        for _, row in msdf.df.iterrows():
            self._mapping_to_rdf(g, ms_node, row)

    @classmethod
    def from_rdf(
        cls,
        g: Graph,
        curie_converter: Optional[Converter] = None,
        default_meta: Optional[MetadataType] = None,
    ) -> MappingSetDataFrame:
        """Extract a MappingSetDataFrame from a RDF graph.

        This is the intended client-facing interface to deserialise
        a MappingSetDataFrame from a RDF graph.

        :param g: Graph: The graph to extract from.
        :param curie_converter: The CURIE converter to use for
            contracting IRIs into their CURIE form. If not provided,
            a default converter will be automatically created using
            namespace declarations found in the graph itself.
        :param default_meta: Default mapping set metadata to use to
            complement the metadata found in the graph.

        :returns: The extracted MappingSetDataFrame.
        """
        if curie_converter is None:
            curie_converter = Converter.from_rdflib(g)
        conv = MappingSetRDFConverter(curie_converter)
        if default_meta is not None:
            conv.default_metadata = default_meta
        return conv.msdf_from_rdf(g)

    @classmethod
    def to_rdf(cls, msdf: MappingSetDataFrame, hydrate: bool = False) -> Graph:
        """Export a MappingSetDataFrame into a RDF graph.

        This is the intended client-facing interface to serialise a
        MappingSetDataFrame into a RDF graph.

        :param msdf: The MappingSetDataFrame to export.

        :returns: A newly created RDF graph.
        """
        g = Graph()
        for k, v in msdf.converter.bimap.items():
            g.namespace_manager.bind(k, v)
        conv = MappingSetRDFConverter(msdf.converter)
        conv.hydrate = hydrate
        conv.msdf_to_rdf(g, msdf)
        return g


class MappingConverter(ObjectConverter):
    """Specialised class to (de)serialise Mapping objects."""

    def __init__(self, curie_converter: Converter):
        """Create a new instance.

        :param curie_converter: The CURIE converter to use throughout
            all (de)serialisation operations.
        """
        super().__init__("mapping", curie_converter)

    @override
    def _init_dict_from_rdf(self, g: Graph, subject: Node) -> Dict[str, Any]:
        dest: Dict[str, Any] = {}

        # If the root node is a named resource, then it is interpreted
        # as the RECORD_ID for the mapping.
        if isinstance(subject, URIRef):
            dest[RECORD_ID] = str(subject)
        elif not isinstance(subject, BNode):
            raise ValueError(f"Invalid node type for a {self.name} object")
        return dest

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
    def _init_dict_to_rdf(self, g: Graph, obj: Dict[str, Any]) -> Node:
        if RECORD_ID in obj:
            return URIRef(self.curie_converter.expand(obj[RECORD_ID]))
        else:
            return BNode()

    @override
    def _get_multi_values(self, value: Any) -> List[Any]:
        return cast(List[Any], value.split("|"))
