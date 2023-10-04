"""SSSOM parsers."""

import io
import itertools as itt
import json
import logging as _logging
import re
import typing
from collections import ChainMap, Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO, Tuple, Union, cast
from xml.dom import Node, minidom
from xml.dom.minidom import Document

import curies
import numpy as np
import pandas as pd
import requests
import yaml
from curies import Converter
from linkml_runtime.loaders.json_loader import JSONLoader
from pandas.errors import EmptyDataError
from rdflib import Graph, URIRef
from sssom_schema import Mapping, MappingSet

from sssom.constants import (
    CONFIDENCE,
    CURIE_MAP,
    DEFAULT_LICENSE,
    DEFAULT_MAPPING_PROPERTIES,
    LICENSE,
    MAPPING_JUSTIFICATION,
    MAPPING_JUSTIFICATION_UNSPECIFIED,
    MAPPING_SET_ID,
    OBJECT_ID,
    OBJECT_LABEL,
    OBJECT_SOURCE,
    OBJECT_SOURCE_ID,
    OWL_EQUIV_CLASS_URI,
    PREDICATE_ID,
    RDF_TYPE,
    RDF_TYPE_URI,
    RDFS_SUBCLASS_OF,
    SKOS_BROAD_MATCH,
    SKOS_BROAD_MATCH_URI,
    SKOS_EXACT_MATCH,
    SKOS_EXACT_MATCH_URI,
    SKOS_NARROW_MATCH,
    SKOS_NARROW_MATCH_URI,
    SUBJECT_ID,
    SUBJECT_LABEL,
    SUBJECT_SOURCE,
    SUBJECT_SOURCE_ID,
    SSSOMSchemaView,
)

from .context import HINT, _get_built_in_prefix_map, ensure_converter
from .sssom_document import MappingSetDocument
from .typehints import Metadata, MetadataType, generate_mapping_set_id, get_default_metadata
from .util import (
    PREFIX_MAP_KEY,
    SSSOM_DEFAULT_RDF_SERIALISATION,
    URI_SSSOM_MAPPINGS,
    MappingSetDataFrame,
    get_file_extension,
    is_multivalued_slot,
    raise_for_bad_path,
    safe_compress,
    to_mapping_set_dataframe,
)

logging = _logging.getLogger(__name__)

# * *******************************************************
# Parsers (from file)


def _open_input(input: Union[str, Path, TextIO]) -> io.StringIO:
    """Transform a URL, a filepath (from pathlib), or a string (with file contents) to a StringIO object.

    :param input: A string representing a URL, a filepath, or file contents,
                              or a Path object representing a filepath.
    :return: A StringIO object containing the input data.
    """
    # If the import already is a StrinIO, return it
    if isinstance(input, io.StringIO):
        return input
    elif isinstance(input, Path):
        input = str(input)

    if isinstance(input, str):
        if input.startswith("http://") or input.startswith("https://"):
            # It's a URL
            data = requests.get(input, timeout=30).content
            return io.StringIO(data.decode("utf-8"))
        elif "\n" in input or "\r" in input:
            # It's string data
            return io.StringIO(input)
        else:
            # It's a local file path
            with open(input, "r") as file:
                file_content = file.read()
            return io.StringIO(file_content)

    raise IOError(f"Could not determine the type of input {input}")


def _separate_metadata_and_table_from_stream(s: io.StringIO):
    s.seek(0)

    # Create a new StringIO object for filtered data
    table_component = io.StringIO()
    metadata_component = io.StringIO()

    header_section = True

    # Filter out lines starting with '#'
    for line in s:
        if not line.startswith("#"):
            table_component.write(line)
            if header_section:
                header_section = False
        elif header_section:
            metadata_component.write(line)
        else:
            logging.info(
                f"Line {line} is starting with hash symbol, but header section is already passed. "
                f"This line is skipped"
            )

    # Reset the cursor to the start of the new StringIO object
    table_component.seek(0)
    metadata_component.seek(0)
    return table_component, metadata_component


def _read_pandas_and_metadata(input: io.StringIO, sep: str = None):
    """Read a tabular data file by wrapping func:`pd.read_csv` to handles comment lines correctly.

    :param input: The file to read. If no separator is given, this file should be named.
    :param sep: File separator for pandas
    :return: A pandas dataframe
    """
    table_stream, metadata_stream = _separate_metadata_and_table_from_stream(input)

    try:
        df = pd.read_csv(table_stream, sep=sep, dtype=str, engine="python")
        df.fillna("", inplace=True)
    except EmptyDataError as e:
        logging.warning(f"Seems like the dataframe is empty: {e}")
        df = pd.DataFrame(
            columns=[
                SUBJECT_ID,
                SUBJECT_LABEL,
                PREDICATE_ID,
                OBJECT_ID,
                MAPPING_JUSTIFICATION,
            ]
        )

    if isinstance(df, pd.DataFrame):
        sssom_metadata = _read_metadata_from_table(metadata_stream)
        return df, sssom_metadata

    return None, None


def _get_seperator_symbol_from_file_path(file):
    r"""
    Take as an input a filepath and return the seperate symbol used, for example, by pandas.

    :param file: the file path
    :return: the seperator symbols as a string, e.g. '\t'
    """
    if isinstance(file, Path) or isinstance(file, str):
        extension = get_file_extension(file)
        if extension == "tsv":
            return "\t"
        elif extension == "csv":
            return ","
        logging.warning(f"Could not guess file extension for {file}")
    return None


def parse_sssom_table(
    file_path: Union[str, Path, TextIO],
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
    **kwargs,
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a :class:`MappingSetDataFrame`."""
    if isinstance(file_path, Path) or isinstance(file_path, str):
        raise_for_bad_path(file_path)
    stream: io.StringIO = _open_input(file_path)
    sep_new = _get_seperator_symbol_from_file_path(file_path)
    df, sssom_metadata = _read_pandas_and_metadata(stream, sep_new)
    if meta is None:
        meta = {}

    # The priority order for combining prefix maps are:
    #  1. Built-in prefix map
    #  2. Internal prefix map inside the document
    #  3. Prefix map passed through this function inside the ``meta``
    #  4. Prefix map passed through this function to ``prefix_map`` (handled with ensure_converter)
    converter = curies.chain(
        [
            _get_built_in_prefix_map(),
            Converter.from_prefix_map(sssom_metadata.pop(CURIE_MAP, {})),
            Converter.from_prefix_map(meta.pop(CURIE_MAP, {})),
            ensure_converter(prefix_map, use_defaults=False),
        ]
    )

    # The priority order for combining metadata is:
    #  1. Metadata appearing in the SSSOM document
    #  2. Metadata passed through ``meta`` to this function
    #  3. Default metadata
    combine_meta = dict(
        ChainMap(
            sssom_metadata,
            meta,
            get_default_metadata(),
        )
    )

    msdf = from_sssom_dataframe(df, prefix_map=converter, meta=combine_meta)
    return msdf


def parse_sssom_rdf(
    file_path: str,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
    serialisation=SSSOM_DEFAULT_RDF_SERIALISATION,
    **kwargs
    # mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a :class:`MappingSetDataFrame`."""
    raise_for_bad_path(file_path)
    metadata = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)

    g = Graph()
    g.parse(file_path, format=serialisation)
    msdf = from_sssom_rdf(g, prefix_map=metadata.prefix_map, meta=metadata.metadata)
    # df: pd.DataFrame = msdf.df
    # if mapping_predicates and not df.empty():
    #     msdf.df = df[df["predicate_id"].isin(mapping_predicates)]
    return msdf


def parse_sssom_json(
    file_path: str,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
    **kwargs
    # mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a  :class`MappingSetDataFrame`."""
    raise_for_bad_path(file_path)
    metadata = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)

    with open(file_path) as json_file:
        jsondoc = json.load(json_file)
    msdf = from_sssom_json(jsondoc=jsondoc, prefix_map=metadata.prefix_map, meta=metadata.metadata)
    # df: pd.DataFrame = msdf.df
    # if mapping_predicates and not df.empty():
    #     msdf.df = df[df["predicate_id"].isin(mapping_predicates)]
    return msdf


# Import methods from external file formats


def parse_obographs_json(
    file_path: str,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
    mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse an obographs file as a JSON object and translates it into a MappingSetDataFrame.

    :param file_path: The path to the obographs file
    :param prefix_map: an optional prefix map
    :param meta: an optional dictionary of metadata elements
    :param mapping_predicates: an optional list of mapping predicates that should be extracted
    :return: A SSSOM MappingSetDataFrame
    """
    raise_for_bad_path(file_path)

    _xmetadata = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)

    with open(file_path) as json_file:
        jsondoc = json.load(json_file)

    return from_obographs(
        jsondoc,
        prefix_map=_xmetadata.prefix_map,
        meta=_xmetadata.metadata,
        mapping_predicates=mapping_predicates,
    )


def _get_prefix_map_and_metadata(
    prefix_map: HINT = None, meta: Optional[MetadataType] = None
) -> Metadata:
    if prefix_map and meta and PREFIX_MAP_KEY in meta:
        logging.info(
            "Prefix map provided as parameter, but SSSOM file provides its own prefix map. "
            "Prefix map provided externally is disregarded in favour of the prefix map in the SSSOM file."
        )
        prefix_map = meta[PREFIX_MAP_KEY]
    converter = ensure_converter(prefix_map)
    if meta is None:
        meta = Metadata.default().metadata
    return Metadata(converter=converter, metadata=meta)


def _address_multivalued_slot(k: str, v: Any) -> Union[str, List[str]]:
    if is_multivalued_slot(k) and v is not None and isinstance(v, str):
        # IF k is multivalued, then v = List[values]
        return [s.strip() for s in v.split("|")]
    else:
        return v


def _init_mapping_set(meta: Optional[MetadataType]) -> MappingSet:
    license = DEFAULT_LICENSE
    mapping_set_id = generate_mapping_set_id()
    if meta is not None:
        if MAPPING_SET_ID in meta.keys():
            mapping_set_id = meta[MAPPING_SET_ID]
        if LICENSE in meta.keys():
            license = meta[LICENSE]
    return MappingSet(mapping_set_id=mapping_set_id, license=license)


def _get_mdict_ms_and_bad_attrs(row: pd.Series, bad_attrs: Counter) -> Tuple[dict, Counter]:
    mdict = {}
    sssom_schema_object = (
        SSSOMSchemaView.instance if hasattr(SSSOMSchemaView, "instance") else SSSOMSchemaView()
    )
    for k, v in row.items():
        if v and v == v:
            ok = False
            if k:
                k = str(k)
            v = _address_multivalued_slot(k, v)

            if k in sssom_schema_object.mapping_slots:
                mdict[k] = v
                ok = True

            # ! This causes propogation of
            # ! mappings level metadata to the mapping set
            # ! which is not desirable atm.
            # if k in sssom_schema_object.mapping_set_slots:
            #     ms[k] = v
            #     ok = True
            if not ok:
                bad_attrs[k] += 1
    return (mdict, bad_attrs)


def parse_alignment_xml(
    file_path: str,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
    mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV -> MappingSetDocument -> MappingSetDataFrame."""
    raise_for_bad_path(file_path)

    metadata = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)
    logging.info("Loading from alignment API")
    xmldoc = minidom.parse(file_path)
    msdf = from_alignment_minidom(
        xmldoc,
        prefix_map=metadata.prefix_map,
        meta=metadata.metadata,
        mapping_predicates=mapping_predicates,
    )
    return msdf


# Readers (from object)


def from_sssom_dataframe(
    df: pd.DataFrame,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Convert a dataframe to a MappingSetDataFrame.

    :param df: A mappings dataframe
    :param prefix_map: A prefix map
    :param meta: A metadata dictionary
    :return: MappingSetDataFrame
    """
    converter = ensure_converter(prefix_map)

    # Need to revisit this solution.
    # This is to address: A value is trying to be set on a copy of a slice from a DataFrame
    if CONFIDENCE in df.columns:
        df2 = df.copy()
        df2[CONFIDENCE].replace(r"^\s*$", np.NaN, regex=True, inplace=True)
        df = df2

    mlist: List[Mapping] = []
    ms = _init_mapping_set(meta)
    bad_attrs: typing.Counter[str] = Counter()
    for _, row in df.iterrows():
        mdict, bad_attrs = _get_mdict_ms_and_bad_attrs(row, bad_attrs)
        _add_valid_mapping_to_list(mdict, mlist, flip_superclass_assertions=True)
    for k, v in bad_attrs.most_common():
        logging.debug(f"No attr for {k} [{v} instances]")
    # the autogenerated code's type annotations are _really_ messy. This is in fact okay,
    # so with a heavy heart we employ type:ignore
    ms.mappings = mlist  # type:ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    doc = MappingSetDocument(mapping_set=ms, converter=converter)
    return to_mapping_set_dataframe(doc)


def from_sssom_rdf(
    g: Graph,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Convert an SSSOM RDF graph into a SSSOM data table.

    :param g: the Graph (rdflib)
    :param prefix_map: A dictionary containing the prefix map, defaults to None
    :param meta: Potentially additional metadata, defaults to None
    :return: MappingSetDataFrame object
    """
    converter = ensure_converter(prefix_map)

    ms = _init_mapping_set(meta)
    mlist: List[Mapping] = []
    for sx, px, ox in g.triples((None, URIRef(URI_SSSOM_MAPPINGS), None)):
        mdict: Dict[str, Any] = {}
        # TODO replace with g.predicate_objects()
        for _, predicate, o in g.triples((ox, None, None)):
            if not isinstance(predicate, URIRef):
                continue
            try:
                predicate_curie = safe_compress(predicate, converter)
            except ValueError as e:
                logging.debug(e)
                continue
            if predicate_curie.startswith("sssom:"):
                key = predicate_curie.replace("sssom:", "")
            elif predicate_curie == "owl:annotatedProperty":
                key = "predicate_id"
            elif predicate_curie == "owl:annotatedTarget":
                key = "object_id"
            elif predicate_curie == "owl:annotatedSource":
                key = "subject_id"
            else:
                continue

            if isinstance(o, URIRef):
                v: Any
                try:
                    v = safe_compress(o, converter)
                except ValueError as e:
                    logging.debug(e)
                    continue
            else:
                v = o.toPython()

            mdict[key] = _address_multivalued_slot(key, v)

        if not mdict:
            logging.warning(
                f"While trying to prepare a mapping for {sx},{px}, {ox}, something went wrong. "
                f"This usually happens when a critical prefix_map entry is missing."
            )
            continue
        _add_valid_mapping_to_list(mdict, mlist, flip_superclass_assertions=True)

    ms.mappings = mlist  # type: ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, converter=converter)
    return to_mapping_set_dataframe(mdoc)


def from_sssom_json(
    jsondoc: Union[str, dict, TextIO],
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Load a mapping set dataframe from a JSON object.

    :param jsondoc: JSON document
    :param prefix_map: Prefix map
    :param meta: metadata
    :return: MappingSetDataFrame object
    """
    converter = ensure_converter(prefix_map)
    mapping_set = cast(MappingSet, JSONLoader().load(source=jsondoc, target_class=MappingSet))

    _set_metadata_in_mapping_set(mapping_set, metadata=meta)
    mapping_set_document = MappingSetDocument(mapping_set=mapping_set, converter=converter)
    return to_mapping_set_dataframe(mapping_set_document)


def from_alignment_minidom(
    dom: Document,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
    mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Read a minidom Document object.

    :param dom: XML (minidom) object
    :param prefix_map: A prefix map
    :param meta: Optional meta data
    :param mapping_predicates: Optional list of mapping predicates to extract
    :return: MappingSetDocument
    :raises ValueError: for alignment format: xml element said, but not set to yes. Only XML is supported!
    """
    converter = ensure_converter(prefix_map)
    ms = _init_mapping_set(meta)
    mlist: List[Mapping] = []
    # bad_attrs = {}

    if not mapping_predicates:
        mapping_predicates = DEFAULT_MAPPING_PROPERTIES

    alignments = dom.getElementsByTagName("Alignment")
    for n in alignments:
        for e in n.childNodes:
            if e.nodeType == Node.ELEMENT_NODE:
                node_name = e.nodeName
                if node_name == "map":
                    cell = e.getElementsByTagName("Cell")
                    for c_node in cell:
                        mdict = _cell_element_values(
                            c_node, converter, mapping_predicates=mapping_predicates
                        )
                        _add_valid_mapping_to_list(mdict, mlist, flip_superclass_assertions=True)

                elif node_name == "xml":
                    if e.firstChild.nodeValue != "yes":
                        raise ValueError(
                            "Alignment format: xml element said, but not set to yes. Only XML is supported!"
                        )
                elif node_name == "onto1":
                    ms[SUBJECT_SOURCE_ID] = e.firstChild.nodeValue
                elif node_name == "onto2":
                    ms[OBJECT_SOURCE_ID] = e.firstChild.nodeValue
                elif node_name == "uri1":
                    ms[SUBJECT_SOURCE] = e.firstChild.nodeValue
                elif node_name == "uri2":
                    ms[OBJECT_SOURCE] = e.firstChild.nodeValue

    ms.mappings = mlist  # type: ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mapping_set_document = MappingSetDocument(mapping_set=ms, converter=converter)
    return to_mapping_set_dataframe(mapping_set_document)


def _get_obographs_predicate_id(obographs_predicate: str):
    if obographs_predicate == "is_a":
        return RDFS_SUBCLASS_OF
    return obographs_predicate


def from_obographs(
    jsondoc: Dict,
    prefix_map: HINT = None,
    meta: Optional[MetadataType] = None,
    mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Convert a obographs json object to an SSSOM data frame.

    :param jsondoc: The JSON object representing the ontology in obographs format
    :param prefix_map: The prefix map to be used
    :param meta: Any additional metadata that needs to be added to the resulting SSSOM data frame, defaults to None
    :param mapping_predicates: Optional list of mapping predicates to extract
    :raises Exception: When there is no CURIE
    :return: An SSSOM data frame (MappingSetDataFrame)
    """
    converter = ensure_converter(prefix_map)
    ms = _init_mapping_set(meta)
    mlist: List[Mapping] = []
    # bad_attrs = {}

    if not mapping_predicates:
        mapping_predicates = DEFAULT_MAPPING_PROPERTIES

    labels = {}

    # Build a dictionary of labels to populate _label columns
    if "graphs" in jsondoc:
        for g in jsondoc["graphs"]:
            if "nodes" in g:
                for n in g["nodes"]:
                    nid = n["id"]
                    if "lbl" in n:
                        label = n["lbl"]
                    else:
                        label = ""
                    labels[nid] = label

    if "graphs" in jsondoc:
        for g in jsondoc["graphs"]:
            if "nodes" in g:
                for n in g["nodes"]:
                    nid = n["id"]
                    if "lbl" in n:
                        label = n["lbl"]
                    else:
                        label = ""
                    if "meta" in n:
                        if (
                            "xrefs" in n["meta"]
                            and "http://www.geneontology.org/formats/oboInOwl#hasDbXref"
                            in mapping_predicates
                        ):
                            for xref in n["meta"]["xrefs"]:
                                xref_id = xref["val"]
                                mdict: Dict[str, Any] = {}
                                try:
                                    mdict[SUBJECT_ID] = safe_compress(nid, converter)
                                    mdict[OBJECT_ID] = safe_compress(xref_id, converter)
                                    mdict[SUBJECT_LABEL] = label
                                    mdict[PREDICATE_ID] = "oboInOwl:hasDbXref"
                                    mdict[MAPPING_JUSTIFICATION] = MAPPING_JUSTIFICATION_UNSPECIFIED
                                    _add_valid_mapping_to_list(mdict, mlist)
                                except ValueError as e:
                                    logging.debug(e)
                        if "basicPropertyValues" in n["meta"]:
                            for value in n["meta"]["basicPropertyValues"]:
                                pred = value["pred"]
                                if pred in mapping_predicates:
                                    xref_id = value["val"]
                                    mdict = {}
                                    try:
                                        mdict[SUBJECT_ID] = safe_compress(nid, converter)
                                        mdict[OBJECT_ID] = safe_compress(xref_id, converter)
                                        mdict[SUBJECT_LABEL] = label
                                        mdict[PREDICATE_ID] = safe_compress(pred, converter)
                                        mdict[
                                            MAPPING_JUSTIFICATION
                                        ] = MAPPING_JUSTIFICATION_UNSPECIFIED
                                        _add_valid_mapping_to_list(mdict, mlist)
                                    except ValueError as e:
                                        # FIXME this will cause ragged mappings
                                        logging.warning(e)
            if "edges" in g:
                for edge in g["edges"]:
                    mdict = {}
                    subject_id = edge["sub"]
                    predicate_id = _get_obographs_predicate_id(edge["pred"])
                    object_id = edge["obj"]
                    if predicate_id in mapping_predicates:
                        mdict[SUBJECT_ID] = safe_compress(subject_id, converter)
                        mdict[OBJECT_ID] = safe_compress(object_id, converter)
                        mdict[SUBJECT_LABEL] = (
                            labels[subject_id] if subject_id in labels.keys() else ""
                        )
                        mdict[OBJECT_LABEL] = (
                            labels[object_id] if object_id in labels.keys() else ""
                        )
                        mdict[PREDICATE_ID] = safe_compress(predicate_id, converter)
                        mdict[MAPPING_JUSTIFICATION] = MAPPING_JUSTIFICATION_UNSPECIFIED
                        _add_valid_mapping_to_list(mdict, mlist)
            if "equivalentNodesSets" in g and OWL_EQUIV_CLASS_URI in mapping_predicates:
                for equivalents in g["equivalentNodesSets"]:
                    if "nodeIds" in equivalents:
                        for ec1 in equivalents["nodeIds"]:
                            for ec2 in equivalents["nodeIds"]:
                                if ec1 != ec2:
                                    mdict = {}
                                    mdict[SUBJECT_ID] = safe_compress(ec1, converter)
                                    mdict[OBJECT_ID] = safe_compress(ec2, converter)
                                    mdict[PREDICATE_ID] = safe_compress(
                                        OWL_EQUIV_CLASS_URI, converter
                                    )
                                    mdict[MAPPING_JUSTIFICATION] = MAPPING_JUSTIFICATION_UNSPECIFIED
                                    mdict[SUBJECT_LABEL] = (
                                        labels[ec1] if ec1 in labels.keys() else ""
                                    )
                                    mdict[OBJECT_LABEL] = (
                                        labels[ec2] if ec2 in labels.keys() else ""
                                    )
                                    _add_valid_mapping_to_list(mdict, mlist)
    else:
        raise Exception("No graphs element in obographs file, wrong format?")

    ms.mappings = mlist  # type: ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, converter=converter)
    return to_mapping_set_dataframe(mdoc)


# All from_* take as an input a python object (data frame, json, etc.) and return a MappingSetDataFrame
# All read_* take as an input a file handle and return a MappingSetDataFrame (usually wrapping a from_* method)


PARSING_FUNCTIONS: typing.Mapping[str, Callable] = {
    "tsv": parse_sssom_table,
    "obographs-json": parse_obographs_json,
    "alignment-api-xml": parse_alignment_xml,
    "json": parse_sssom_json,
    "rdf": parse_sssom_rdf,
}


def get_parsing_function(input_format: Optional[str], filename: str) -> Callable:
    """Return appropriate parser function based on input format of file.

    :param input_format: File format
    :param filename: Filename
    :raises Exception: Unknown file format
    :return: Appropriate 'read' function
    """
    if input_format is None:
        input_format = get_file_extension(filename)
    func = PARSING_FUNCTIONS.get(input_format)
    if func is None:
        raise Exception(f"Unknown input format: {input_format}")
    return func


def _flip_superclass_assertion(mapping: Mapping) -> Mapping:
    if mapping.predicate_id == "sssom:superClassOf":
        mapping.predicate_id = "rdfs:subClassOf"
        return _swap_object_subject(mapping)
    return mapping


def _swap_object_subject(mapping: Mapping) -> Mapping:
    members = [
        attr.replace("subject_", "")
        for attr in dir(mapping)
        if not callable(getattr(mapping, attr))
        and not attr.startswith("__")
        and attr.startswith("subject_")
    ]
    for var in members:
        subject_val = getattr(mapping, "subject_" + var)
        object_val = getattr(mapping, "object_" + var)
        setattr(mapping, "subject_" + var, object_val)
        setattr(mapping, "object_" + var, subject_val)
    return mapping


def _read_metadata_from_table(stream: io.StringIO) -> Dict[str, Any]:
    yamlstr = ""
    for line in stream:
        if line.startswith("#"):
            yamlstr += re.sub("^#", "", line)
        else:
            break

    if yamlstr:
        meta = yaml.safe_load(yamlstr)
        logging.info(f"Meta={meta}")
        return meta
    return {}


def _set_metadata_in_mapping_set(
    mapping_set: MappingSet, metadata: Optional[MetadataType] = None
) -> None:
    if metadata is None:
        logging.info("Tried setting metadata but none provided.")
    else:
        for k, v in metadata.items():
            if k != PREFIX_MAP_KEY:
                mapping_set[k] = v


def _cell_element_values(cell_node, converter: Converter, mapping_predicates) -> Optional[Dict]:
    mdict: Dict[str, Any] = {}
    for child in cell_node.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            try:
                if child.nodeName == "entity1":
                    mdict[SUBJECT_ID] = safe_compress(child.getAttribute("rdf:resource"), converter)
                elif child.nodeName == "entity2":
                    mdict[OBJECT_ID] = safe_compress(child.getAttribute("rdf:resource"), converter)
                elif child.nodeName == "measure":
                    mdict[CONFIDENCE] = child.firstChild.nodeValue
                elif child.nodeName == "relation":
                    relation = child.firstChild.nodeValue
                    if (relation == "=") and (SKOS_EXACT_MATCH_URI in mapping_predicates):
                        mdict[PREDICATE_ID] = SKOS_EXACT_MATCH
                    elif (relation == "<") and (SKOS_BROAD_MATCH_URI in mapping_predicates):
                        mdict[PREDICATE_ID] = SKOS_BROAD_MATCH
                    elif (relation == ">") and (SKOS_NARROW_MATCH_URI in mapping_predicates):
                        mdict[PREDICATE_ID] = SKOS_NARROW_MATCH
                    # elif (relation == "%") and (SOMETHING in mapping_predicates)
                    #     # Incompatible.
                    #     pass
                    # elif (relation == "HasInstance") and (SOMETHING in mapping_predicates):
                    #     pass
                    elif (relation == "InstanceOf") and (RDF_TYPE_URI in mapping_predicates):
                        mdict[PREDICATE_ID] = RDF_TYPE
                    else:
                        logging.warning(f"{relation} not a recognised relation type.")
                else:
                    logging.warning(f"Unsupported alignment api element: {child.nodeName}")
            except ValueError as e:
                logging.warning(e)

    mdict[MAPPING_JUSTIFICATION] = MAPPING_JUSTIFICATION_UNSPECIFIED
    return mdict


# The following methods dont really belong in the parser package..


def to_mapping_set_document(msdf: MappingSetDataFrame) -> MappingSetDocument:
    """Convert a MappingSetDataFrame to a MappingSetDocument."""
    mlist: List[Mapping] = []
    ms = _init_mapping_set(msdf.metadata)
    bad_attrs: Counter = Counter()
    if msdf.df is not None:
        for _, row in msdf.df.iterrows():
            mdict, bad_attrs = _get_mdict_ms_and_bad_attrs(row, bad_attrs)
            _add_valid_mapping_to_list(mdict, mlist)
    for k, v in bad_attrs.items():
        logging.warning(f"No attr for {k} [{v} instances]")
    ms.mappings = mlist  # type: ignore
    if msdf.metadata is not None:
        for k, v in msdf.metadata.items():
            if k != PREFIX_MAP_KEY:
                ms[k] = _address_multivalued_slot(k, v)
    return MappingSetDocument(mapping_set=ms, converter=msdf.converter)


def split_dataframe(
    msdf: MappingSetDataFrame,
) -> Dict[str, MappingSetDataFrame]:
    """Group the mapping set dataframe into several subdataframes by prefix.

    :param msdf: MappingSetDataFrame object
    :raises RuntimeError: DataFrame object within MappingSetDataFrame is None
    :return: Mapping object
    """
    if msdf.df is None:
        raise RuntimeError
    subject_prefixes = set(msdf.df[SUBJECT_ID].str.split(":", n=1, expand=True)[0])
    object_prefixes = set(msdf.df[OBJECT_ID].str.split(":", n=1, expand=True)[0])
    relations = set(msdf.df[PREDICATE_ID])
    return split_dataframe_by_prefix(
        msdf=msdf,
        subject_prefixes=subject_prefixes,
        object_prefixes=object_prefixes,
        relations=relations,
    )


def split_dataframe_by_prefix(
    msdf: MappingSetDataFrame,
    subject_prefixes: Iterable[str],
    object_prefixes: Iterable[str],
    relations: Iterable[str],
) -> Dict[str, MappingSetDataFrame]:
    """Split a mapping set dataframe by prefix.

    :param msdf: An SSSOM MappingSetDataFrame
    :param subject_prefixes: a list of prefixes pertaining to the subject
    :param object_prefixes: a list of prefixes pertaining to the object
    :param relations: a list of relations of interest
    :return: a dict of SSSOM data frame names to MappingSetDataFrame
    """
    df = msdf.df
    if df is None:
        raise ValueError
    meta = msdf.metadata
    split_to_msdf: Dict[str, MappingSetDataFrame] = {}
    for subject_prefix, object_prefix, relation in itt.product(
        subject_prefixes, object_prefixes, relations
    ):
        relation_prefix, relation_id = relation.split(":")
        split = f"{subject_prefix.lower()}_{relation_id.lower()}_{object_prefix.lower()}"
        if subject_prefix not in msdf.converter.bimap:
            logging.warning(f"{split} - missing subject prefix - {subject_prefix}")
            continue
        if object_prefix not in msdf.converter.bimap:
            logging.warning(f"{split} - missing object prefix - {object_prefix}")
            continue
        df_subset = df[
            (df[SUBJECT_ID].str.startswith(subject_prefix + ":"))
            & (df[PREDICATE_ID] == relation)
            & (df[OBJECT_ID].str.startswith(object_prefix + ":"))
        ]
        if 0 == len(df_subset):
            logging.debug(f"No matches ({len(df_subset)} matches found)")
            continue
        subconverter = msdf.converter.get_subconverter(
            [subject_prefix, object_prefix, relation_prefix]
        )
        split_to_msdf[split] = from_sssom_dataframe(
            df_subset, prefix_map=dict(subconverter.bimap), meta=meta
        )
    return split_to_msdf


def _ensure_valid_mapping_from_dict(mdict):
    """
    Return a valid mapping object if it can be constructed, else None.

    :param mdict: A dictionary containing the mapping metadata.
    :return: A valid Mapping object, or None.
    """
    mdict.setdefault(MAPPING_JUSTIFICATION, MAPPING_JUSTIFICATION_UNSPECIFIED)

    try:
        m = Mapping(**mdict)
        return m
    except ValueError as e:
        logging.warning(
            f"One mapping in the mapping set is not well-formed, "
            f"and therfore not included in the mapping set ({mdict}). Error: {e}"
        )
    return None


def _add_valid_mapping_to_list(mdict, mlist, flip_superclass_assertions=False):
    """
    Validate the mapping and append to the list if valid.

    Parameters:
    - mdict (dict): A dictionary containing the mapping metadata.
    - mlist (list): The list to which the valid mapping should be appended.
    """
    mapping = _ensure_valid_mapping_from_dict(mdict)
    if mapping:
        if flip_superclass_assertions:
            mlist.append(_flip_superclass_assertion(mdict))
        else:
            mlist.append(mapping)
