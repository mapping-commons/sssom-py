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
from linkml_runtime.loaders.rdflib_loader import RDFLibLoader
from pandas.errors import EmptyDataError
from rdflib import Graph
from sssom_schema import Mapping, MappingSet

from sssom.constants import (
    CONFIDENCE,
    CURIE_MAP,
    DEFAULT_MAPPING_PROPERTIES,
    LICENSE,
    MAPPING_JUSTIFICATION,
    MAPPING_JUSTIFICATION_UNSPECIFIED,
    MAPPING_SET_ID,
    OBJECT_ID,
    OBJECT_LABEL,
    OBJECT_SOURCE,
    OBJECT_SOURCE_ID,
    OBO_HAS_DB_XREF_URI,
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
    MetadataType,
    _get_sssom_schema_object,
    get_default_metadata,
)

from .context import ConverterHint, _get_built_in_prefix_map, ensure_converter
from .sssom_document import MappingSetDocument
from .util import (
    SSSOM_DEFAULT_RDF_SERIALISATION,
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
            # We strip any trailing tabs. Such tabs may have been left
            # by a spreadsheet editor who treated the header lines as
            # if they were normal data lines; they would prevent the
            # YAML parser from correctly parsing the metadata block.
            metadata_component.write(line.rstrip("\t\n") + "\n")
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


def _is_check_valid_extension_slot(slot_name, meta):
    extension_definitions = meta.get("extension_definitions", [])
    return any(
        "property" in entry and entry.get("slot_name") == slot_name
        for entry in extension_definitions
    )


def _is_irregular_metadata(metadata_list: List[Dict]):
    fail_metadata = False
    for m in metadata_list:
        for key in m:
            if key not in _get_sssom_schema_object().mapping_set_slots:
                if not _is_check_valid_extension_slot(key, m):
                    logging.warning(
                        f"Metadata key '{key}' is not a standard SSSOM mapping set metadata field. See "
                        f"https://mapping-commons.github.io/sssom/spec-model/#non-standard-slots on how to "
                        f"specify additional, non-standard fields in a SSSOM file."
                    )
                    fail_metadata = True
    return fail_metadata


def _check_redefined_builtin_prefixes(sssom_metadata, meta, prefix_map):

    # There are three ways in which prefixes can be communicated, so we will check all of them
    # This is a bit overly draconian, as in the end, only the highest priority one gets picked
    # But since this only constitues a (logging) warning, I think its worth reporting
    builtin_converter = _get_built_in_prefix_map()
    sssom_metadata_converter = _get_converter_pop_replace_curie_map(sssom_metadata)
    meta_converter = _get_converter_pop_replace_curie_map(meta)
    prefix_map_converter = ensure_converter(prefix_map, use_defaults=False)
    is_valid_prefixes = True

    for converter in [sssom_metadata_converter, meta_converter, prefix_map_converter]:
        for builtin_prefix, builtin_uri in builtin_converter.bimap.items():
            if builtin_prefix in converter.bimap:
                if builtin_uri != converter.bimap[builtin_prefix]:
                    logging.warning(
                        f"A built-in prefix ({builtin_prefix}) was provided, "
                        f"but the provided URI expansion ({converter.bimap[builtin_prefix]}) does not correspond "
                        f"to the required URI expansion: {builtin_uri}. The prefix will be ignored."
                    )
                    is_valid_prefixes = False
            # NOTE during refactor replace the following line by https://github.com/biopragmatics/curies/pull/136
            reverse_bimap = {value: key for key, value in builtin_converter.bimap.items()}
            if builtin_uri in reverse_bimap:
                if builtin_prefix != reverse_bimap[builtin_uri]:
                    logging.warning(
                        f"A built-in URI namespace ({builtin_uri}) was used in (one of) the provided prefix map(s), "
                        f"but the provided prefix ({reverse_bimap[builtin_uri]}) does not correspond to the "
                        f"standard prefix: {builtin_prefix}. The prefix will be ignored."
                    )
                    is_valid_prefixes = False
    return is_valid_prefixes


def _fail_in_strict_parsing_mode(is_valid_built_in_prefixes, is_valid_metadata):
    report = ""
    if not is_valid_built_in_prefixes:
        report += "STRONG WARNING: The prefix map provided contains built-in prefixes that were redefined.+\n"
    if not is_valid_metadata:
        report += (
            "STRONG WARNING: The metadata provided contains non-standard and undefined metadata.+\n"
        )

    if report:
        raise ValueError(report)


def _get_converter_pop_replace_curie_map(sssom_metadata):
    """
    Pop CURIE_MAP from sssom_metadata, process it, and restore it if it existed.

    Args:
        sssom_metadata (dict): The metadata dictionary.

    Returns:
        Converter: A Converter object created from the CURIE_MAP.
    """
    curie_map = sssom_metadata.pop(CURIE_MAP, {})

    # Process the popped value
    sssom_metadata_converter = Converter.from_prefix_map(curie_map)

    # Reinsert CURIE_MAP if it was present
    if curie_map:
        sssom_metadata[CURIE_MAP] = curie_map

    return sssom_metadata_converter


def parse_sssom_table(
    file_path: Union[str, Path, TextIO],
    prefix_map: ConverterHint = None,
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

    is_valid_built_in_prefixes = _check_redefined_builtin_prefixes(sssom_metadata, meta, prefix_map)
    is_valid_metadata = _is_irregular_metadata([sssom_metadata, meta])

    if kwargs.get("strict"):
        _fail_in_strict_parsing_mode(is_valid_built_in_prefixes, is_valid_metadata)

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
    prefix_map: ConverterHint = None,
    meta: Optional[MetadataType] = None,
    serialisation=SSSOM_DEFAULT_RDF_SERIALISATION,
    **kwargs,
    # mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a :class:`MappingSetDataFrame`."""
    raise_for_bad_path(file_path)

    g = Graph()
    g.parse(file_path, format=serialisation)

    # Initialize meta if it's None
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
            Converter.from_rdflib(g),
            Converter.from_prefix_map(meta.pop(CURIE_MAP, {})),
            ensure_converter(prefix_map, use_defaults=False),
        ]
    )
    msdf = from_sssom_rdf(g, prefix_map=converter, meta=meta)
    # df: pd.DataFrame = msdf.df
    # if mapping_predicates and not df.empty():
    #     msdf.df = df[df["predicate_id"].isin(mapping_predicates)]
    return msdf


def parse_sssom_json(
    file_path: str, prefix_map: ConverterHint = None, meta: Optional[MetadataType] = None, **kwargs
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a :class:`MappingSetDataFrame`."""
    raise_for_bad_path(file_path)

    with open(file_path) as json_file:
        jsondoc = json.load(json_file)

    # Initialize meta if it's None
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
            Converter.from_jsonld(file_path),
            Converter.from_prefix_map(meta.pop(CURIE_MAP, {})),
            ensure_converter(prefix_map, use_defaults=False),
        ]
    )

    msdf = from_sssom_json(jsondoc=jsondoc, prefix_map=converter, meta=meta)
    return msdf


# Import methods from external file formats


def parse_obographs_json(
    file_path: str,
    prefix_map: ConverterHint = None,
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

    converter, meta = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)

    with open(file_path) as json_file:
        jsondoc = json.load(json_file)

    return from_obographs(
        jsondoc,
        prefix_map=converter,
        meta=meta,
        mapping_predicates=mapping_predicates,
    )


def _get_prefix_map_and_metadata(
    prefix_map: ConverterHint = None, meta: Optional[MetadataType] = None
) -> Tuple[Converter, MetadataType]:
    if meta is None:
        meta = get_default_metadata()
    converter = curies.chain(
        [
            _get_built_in_prefix_map(),
            Converter.from_prefix_map(meta.pop(CURIE_MAP, {})),
            ensure_converter(prefix_map, use_defaults=False),
        ]
    )
    return converter, meta


def _address_multivalued_slot(k: str, v: Any) -> Union[str, List[str]]:
    if isinstance(v, str) and is_multivalued_slot(k):
        # IF k is multivalued, then v = List[values]
        return [s.strip() for s in v.split("|")]
    else:
        return v


def _init_mapping_set(meta: Optional[MetadataType]) -> MappingSet:
    _metadata = dict(ChainMap(meta or {}, get_default_metadata()))
    mapping_set = MappingSet(mapping_set_id=_metadata[MAPPING_SET_ID], license=_metadata[LICENSE])
    _set_metadata_in_mapping_set(mapping_set=mapping_set, metadata=meta)
    return mapping_set


def _get_mapping_dict(
    row: pd.Series, bad_attrs: Counter, mapping_slots: typing.Set[str]
) -> Dict[str, Any]:
    """Generate a mapping dictionary from a given row of data.

    It also updates the 'bad_attrs' counter for keys that are not present
    in the sssom_schema_object's mapping_slots.
    """
    # Populate the mapping dictionary with key-value pairs from the row,
    # only if the value exists, is not NaN, and the key is in the schema's mapping slots.
    # The value could be a string or a list and is handled accordingly via _address_multivalued_slot().

    mdict = {
        k: _address_multivalued_slot(k, v)
        for k, v in row.items()
        if v and pd.notna(v) and k in mapping_slots
    }

    # Update bad_attrs for keys not in mapping_slots
    bad_keys = set(row.keys()) - mapping_slots
    for bad_key in bad_keys:
        bad_attrs[bad_key] += 1
    return mdict


def parse_alignment_xml(
    file_path: str,
    prefix_map: ConverterHint = None,
    meta: Optional[MetadataType] = None,
    mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV -> MappingSetDocument -> MappingSetDataFrame."""
    raise_for_bad_path(file_path)

    converter, meta = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)
    logging.info("Loading from alignment API")
    xmldoc = minidom.parse(file_path)
    msdf = from_alignment_minidom(
        xmldoc,
        prefix_map=converter,
        meta=meta,
        mapping_predicates=mapping_predicates,
    )
    return msdf


# Readers (from object)


def from_sssom_dataframe(
    df: pd.DataFrame,
    prefix_map: ConverterHint = None,
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
        df.replace({CONFIDENCE: r"^\s*$"}, np.nan, regex=True, inplace=True)

    mapping_set = _get_mapping_set_from_df(df=df, meta=meta)
    doc = MappingSetDocument(mapping_set=mapping_set, converter=converter)
    return to_mapping_set_dataframe(doc)


def from_sssom_rdf(
    g: Graph,
    prefix_map: ConverterHint = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Convert an SSSOM RDF graph into a SSSOM data table.

    :param g: the Graph (rdflib)
    :param prefix_map: A dictionary containing the prefix map, defaults to None
    :param meta: Potentially additional metadata, defaults to None
    :return: MappingSetDataFrame object
    """
    converter = ensure_converter(prefix_map)
    mapping_set = cast(
        MappingSet,
        RDFLibLoader().load(
            source=g,
            target_class=MappingSet,
            schemaview=_get_sssom_schema_object().view,
            prefix_map=converter.bimap,
            ignore_unmapped_predicates=True,
        ),
    )

    # The priority order for combining metadata is:
    #  1. Metadata appearing in the SSSOM document
    #  2. Metadata passed through ``meta`` to this function
    #  3. Default metadata

    # As the Metadata appearing in the SSSOM document is already parsed by LinkML
    # we only need to overwrite the metadata from 2 and 3 if it is not present
    combine_meta = dict(
        ChainMap(
            meta or {},
            get_default_metadata(),
        )
    )

    _set_metadata_in_mapping_set(mapping_set, metadata=combine_meta, overwrite=False)
    mdoc = MappingSetDocument(mapping_set=mapping_set, converter=converter)
    return to_mapping_set_dataframe(mdoc)


def from_sssom_json(
    jsondoc: Union[str, dict, TextIO],
    prefix_map: ConverterHint = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Load a mapping set dataframe from a JSON object.

    :param jsondoc: JSON document
    :param prefix_map: Prefix map
    :param meta: metadata used to augment the metadata existing in the mapping set
    :return: MappingSetDataFrame object
    """
    converter = ensure_converter(prefix_map)

    mapping_set = cast(MappingSet, JSONLoader().load(source=jsondoc, target_class=MappingSet))

    # The priority order for combining metadata is:
    #  1. Metadata appearing in the SSSOM document
    #  2. Metadata passed through ``meta`` to this function
    #  3. Default metadata

    # As the Metadata appearing in the SSSOM document is already parsed by LinkML
    # we only need to overwrite the metadata from 2 and 3 if it is not present
    combine_meta = dict(
        ChainMap(
            meta or {},
            get_default_metadata(),
        )
    )

    _set_metadata_in_mapping_set(mapping_set, metadata=combine_meta, overwrite=False)
    mapping_set_document = MappingSetDocument(mapping_set=mapping_set, converter=converter)
    return to_mapping_set_dataframe(mapping_set_document)


def from_alignment_minidom(
    dom: Document,
    prefix_map: ConverterHint = None,
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
                        mdict: Dict[str, Any] = _cell_element_values(
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
    mapping_set_document = MappingSetDocument(mapping_set=ms, converter=converter)
    return to_mapping_set_dataframe(mapping_set_document)


def _get_obographs_predicate_id(obographs_predicate: str):
    if obographs_predicate == "is_a":
        return RDFS_SUBCLASS_OF
    return obographs_predicate


def from_obographs(
    jsondoc: Dict,
    prefix_map: ConverterHint = None,
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

    if not mapping_predicates:
        mapping_predicates = DEFAULT_MAPPING_PROPERTIES

    graphs = jsondoc.get("graphs")
    if not graphs:
        raise Exception("No graphs element in obographs file, wrong format?")

    #: A dictionary of node URIs to node labels
    labels: Mapping[str, str] = {
        node["id"]: node.get("lbl")
        for graph in graphs
        for node in graph.get("nodes", [])
        if node.get("lbl")
    }

    for graph in graphs:
        for node in graph.get("nodes", []):
            meta = node.get("meta")
            if not meta:
                continue

            node_uri = node["id"]
            if OBO_HAS_DB_XREF_URI in mapping_predicates:
                for xref in meta.get("xrefs", []):
                    mdict = _make_mdict(
                        node_uri, OBO_HAS_DB_XREF_URI, xref["val"], converter, labels
                    )
                    _add_valid_mapping_to_list(mdict, mlist)

            for value in meta.get("basicPropertyValues", []):
                predicate_uri = value["pred"]
                if predicate_uri not in mapping_predicates:
                    continue
                mdict = _make_mdict(node_uri, predicate_uri, value["val"], converter, labels)
                _add_valid_mapping_to_list(mdict, mlist)

        for edge in graph.get("edges", []):
            predicate_uri = _get_obographs_predicate_id(edge["pred"])
            if predicate_uri not in mapping_predicates:
                continue
            mdict = _make_mdict(edge["sub"], predicate_uri, edge["obj"], converter, labels)
            _add_valid_mapping_to_list(mdict, mlist)

        if OWL_EQUIV_CLASS_URI in mapping_predicates:
            for equivalents in graph.get("equivalentNodesSets", []):
                node_uris = equivalents.get("nodeIds")
                if not node_uris:
                    continue
                for subject_uri, object_uri in itt.product(node_uris, repeat=2):
                    if subject_uri == object_uri:
                        continue
                    mdict = _make_mdict(
                        subject_uri, OWL_EQUIV_CLASS_URI, object_uri, converter, labels
                    )
                    _add_valid_mapping_to_list(mdict, mlist)

    ms.mappings = mlist  # type: ignore
    mdoc = MappingSetDocument(mapping_set=ms, converter=converter)
    return to_mapping_set_dataframe(mdoc)


def _make_mdict(
    subject_id: str,
    predicate_id: str,
    object_id: str,
    converter: Converter,
    labels: typing.Mapping[str, str],
):
    mdict = {
        MAPPING_JUSTIFICATION: MAPPING_JUSTIFICATION_UNSPECIFIED,
    }
    try:
        subject_curie = safe_compress(subject_id, converter)
    except ValueError as e:
        logging.debug("could not parse subject %s - %s", subject_id, e)
    else:
        mdict[SUBJECT_ID] = subject_curie

    try:
        predicate_curie = safe_compress(predicate_id, converter)
    except ValueError as e:
        logging.debug("could not parse predicate %s - %s", predicate_id, e)
    else:
        mdict[PREDICATE_ID] = predicate_curie

    try:
        object_curie = safe_compress(object_id, converter)
    except ValueError as e:
        logging.debug("could not parse object %s - %s", object_id, e)
    else:
        mdict[OBJECT_ID] = object_curie

    if subject_id in labels:
        mdict[SUBJECT_LABEL] = labels[subject_id]
    if object_id in labels:
        mdict[OBJECT_LABEL] = labels[object_id]
    return mdict


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
    if mapping.predicate_id != "sssom:superClassOf":
        return mapping
    mapping.predicate_id = "rdfs:subClassOf"
    return _swap_object_subject(mapping)


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
    mapping_set: MappingSet, metadata: Optional[MetadataType] = None, overwrite: bool = True
) -> None:
    if metadata is None:
        logging.info("Tried setting metadata but none provided.")
    else:
        for k, v in metadata.items():
            if k != CURIE_MAP:
                if (
                    hasattr(mapping_set, k)
                    and getattr(mapping_set, k) is not None
                    and not overwrite
                ):
                    continue
                mapping_set[k] = _address_multivalued_slot(k, v)


def _cell_element_values(cell_node, converter: Converter, mapping_predicates) -> Dict[str, Any]:
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
    ms = _get_mapping_set_from_df(df=msdf.df, meta=msdf.metadata)
    return MappingSetDocument(mapping_set=ms, converter=msdf.converter)


def _get_mapping_set_from_df(df: pd.DataFrame, meta: Optional[MetadataType] = None) -> MappingSet:
    mapping_set = _init_mapping_set(meta)
    bad_attrs: Counter = Counter()

    mapping_slots = set(_get_sssom_schema_object().mapping_slots)

    df.apply(
        lambda row: _add_valid_mapping_to_list(
            _get_mapping_dict(row, bad_attrs, mapping_slots), mapping_set.mappings
        ),
        axis=1,
    )

    for k, v in bad_attrs.items():
        logging.warning(f"No attr for {k} [{v} instances]")
    return mapping_set


def split_dataframe(
    msdf: MappingSetDataFrame,
) -> Dict[str, MappingSetDataFrame]:
    """Group the mapping set dataframe into several subdataframes by prefix.

    :param msdf: MappingSetDataFrame object
    :raises RuntimeError: DataFrame object within MappingSetDataFrame is None
    :return: Mapping object
    """
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


def _ensure_valid_mapping_from_dict(mdict: Dict[str, Any]):
    """
    Return a valid mapping object if it can be constructed, else None.

    :param mdict: A dictionary containing the mapping metadata.
    :return: A valid Mapping object, or None.
    """
    mdict.setdefault(MAPPING_JUSTIFICATION, MAPPING_JUSTIFICATION_UNSPECIFIED)

    try:
        m = Mapping(**mdict)
        if m.subject_type == "rdfs literal":
            if m.subject_label is None:
                raise ValueError("Missing subject_label")
        elif m.subject_id is None:
            raise ValueError("Missing subject_id")
        if m.object_type == "rdfs literal":
            if m.object_label is None:
                raise ValueError("Missing object_label")
        elif m.object_id is None:
            raise ValueError("Missing object_id")
    except ValueError as e:
        logging.warning(
            f"One mapping in the mapping set is not well-formed, "
            f"and therfore not included in the mapping set ({mdict}). Error: {e}"
        )
        return None
    else:
        return m


def _add_valid_mapping_to_list(
    mdict: Dict[str, Any], mlist: List[Mapping], *, flip_superclass_assertions=False
):
    """
    Validate the mapping and append to the list if valid.

    Parameters:
    - mdict (dict): A dictionary containing the mapping metadata.
    - mlist (list): The list to which the valid mapping should be appended.
    - flip_superclass_assertions (bool): an optional paramter that flips sssom:superClassOf to rdfs:subClassOf
    """
    mapping = _ensure_valid_mapping_from_dict(mdict)
    if not mapping:
        return None
    if flip_superclass_assertions:
        mapping = _flip_superclass_assertion(mapping)
    mlist.append(mapping)
