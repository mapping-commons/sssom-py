"""SSSOM parsers."""

import json
import logging
import re
import typing
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union, cast
from urllib.request import urlopen
from xml.dom import Node, minidom
from xml.dom.minidom import Document

import numpy as np
import pandas as pd
import validators
import yaml
from deprecation import deprecated
from linkml_runtime.loaders.json_loader import JSONLoader
from rdflib import Graph, URIRef

from sssom.constants import (
    DEFAULT_MAPPING_PROPERTIES,
    MAPPING_SET_SLOTS,
    MAPPING_SLOTS,
    OWL_EQUIV_CLASS,
    RDFS_SUBCLASS_OF,
)

from .context import (
    DEFAULT_LICENSE,
    DEFAULT_MAPPING_SET_ID,
    add_built_in_prefixes_to_prefix_map,
    get_default_metadata,
)
from .sssom_datamodel import Mapping, MappingSet
from .sssom_document import MappingSetDocument
from .typehints import Metadata, MetadataType, PrefixMap
from .util import (
    PREFIX_MAP_KEY,
    SSSOM_DEFAULT_RDF_SERIALISATION,
    URI_SSSOM_MAPPINGS,
    MappingSetDataFrame,
    NoCURIEException,
    curie_from_uri,
    get_file_extension,
    is_multivalued_slot,
    raise_for_bad_path,
    read_pandas,
    to_mapping_set_dataframe,
)

# Constants
MATCH_TYPE_UNSPECIFIED = "Unspecified"


# * DEPRECATED methods *****************************************


@deprecated(
    deprecated_in="0.3.10",
    removed_in="0.3.11",
    details="Use 'parse_sssom_table' instead.",
)
def read_sssom_table(
    file_path: Union[str, Path],
    prefix_map: Optional[PrefixMap] = None,
    meta: Optional[MetadataType] = None,
    **kwargs,
) -> MappingSetDataFrame:
    """DEPRECATE."""
    return parse_sssom_table(
        file_path=file_path, prefix_map=prefix_map, meta=meta, kwargs=kwargs
    )


@deprecated(
    deprecated_in="0.3.10",
    removed_in="0.3.11",
    details="Use 'parse_sssom_rdf' instead.",
)
def read_sssom_rdf(
    file_path: str,
    prefix_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
    serialisation=SSSOM_DEFAULT_RDF_SERIALISATION,
    **kwargs,
) -> MappingSetDataFrame:
    """DEPRECATE."""
    return parse_sssom_rdf(
        file_path=file_path,
        prefix_map=prefix_map,
        meta=meta,
        serialisation=serialisation,
        kwargs=kwargs,
    )


@deprecated(
    deprecated_in="0.3.10",
    removed_in="0.3.11",
    details="Use 'parse_sssom_json' instead.",
)
def read_sssom_json(
    file_path: str,
    prefix_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
    **kwargs
    # mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """DEPRECATE."""
    return parse_sssom_json(
        file_path=file_path, prefix_map=prefix_map, meta=meta, kwarg=kwargs
    )


# * *******************************************************
# Parsers (from file)


def parse_sssom_table(
    file_path: Union[str, Path],
    prefix_map: Optional[PrefixMap] = None,
    meta: Optional[MetadataType] = None,
    **kwargs
    # mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a :class:`MappingSetDataFrame`."""
    raise_for_bad_path(file_path)
    df = read_pandas(file_path)
    # if mapping_predicates:
    #     # Filter rows based on presence of predicate_id list provided.
    #     df = df[df["predicate_id"].isin(mapping_predicates)]

    # If SSSOM external metadata is provided, merge it with the internal metadata
    sssom_metadata = _read_metadata_from_table(file_path)

    if sssom_metadata:
        if meta:
            for k, v in meta.items():
                if k in sssom_metadata:
                    if sssom_metadata[k] != v:
                        logging.warning(
                            f"SSSOM internal metadata {k} ({sssom_metadata[k]}) "
                            f"conflicts with provided ({meta[k]})."
                        )
                else:
                    logging.info(
                        f"Externally provided metadata {k}:{v} is added to metadata set."
                    )
                    sssom_metadata[k] = v
        meta = sssom_metadata

    prefix_map, meta = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)
    msdf = from_sssom_dataframe(df, prefix_map=prefix_map, meta=meta)
    return msdf


def parse_sssom_rdf(
    file_path: str,
    prefix_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
    serialisation=SSSOM_DEFAULT_RDF_SERIALISATION,
    **kwargs
    # mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a :class:`MappingSetDataFrame`."""
    raise_for_bad_path(file_path)
    metadata = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)

    g = Graph()
    g.load(file_path, format=serialisation)
    msdf = from_sssom_rdf(g, prefix_map=metadata.prefix_map, meta=metadata.metadata)
    # df: pd.DataFrame = msdf.df
    # if mapping_predicates and not df.empty():
    #     msdf.df = df[df["predicate_id"].isin(mapping_predicates)]
    return msdf


def parse_sssom_json(
    file_path: str,
    prefix_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
    **kwargs
    # mapping_predicates: Optional[List[str]] = None,
) -> MappingSetDataFrame:
    """Parse a TSV to a :class:`MappingSetDocument` to a  :class`MappingSetDataFrame`."""
    raise_for_bad_path(file_path)
    metadata = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)

    with open(file_path) as json_file:
        jsondoc = json.load(json_file)
    msdf = from_sssom_json(
        jsondoc=jsondoc, prefix_map=metadata.prefix_map, meta=metadata.metadata
    )
    # df: pd.DataFrame = msdf.df
    # if mapping_predicates and not df.empty():
    #     msdf.df = df[df["predicate_id"].isin(mapping_predicates)]
    return msdf


# Import methods from external file formats


def parse_obographs_json(
    file_path: str,
    prefix_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
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
    prefix_map: Optional[PrefixMap] = None, meta: Optional[MetadataType] = None
) -> Metadata:
    default_metadata = get_default_metadata()

    if prefix_map is None:
        logging.warning(
            "No prefix map provided (not recommended), trying to use defaults.."
        )
        prefix_map = default_metadata.prefix_map

    if meta is None:
        meta = default_metadata.metadata
    else:
        if prefix_map and PREFIX_MAP_KEY in meta:
            logging.info(
                "Prefix map provided as parameter, but SSSOM file provides its own prefix map. "
                "Prefix map provided externally is disregarded in favour of the prefix map in the SSSOM file."
            )
            prefix_map = cast(PrefixMap, meta[PREFIX_MAP_KEY])

    return Metadata(prefix_map=prefix_map, metadata=meta)


def _address_multivalued_slot(k: str, v: str) -> Union[str, List[str]]:
    if is_multivalued_slot(k) and v is not None and isinstance(v, str) and "|" in v:
        # IF k is multivalued, then v = List[values]
        return [s.strip() for s in v.split("|")]
    else:
        return v


def _init_mapping_set(meta: Optional[MetadataType]) -> MappingSet:
    license = DEFAULT_LICENSE
    mapping_set_id = DEFAULT_MAPPING_SET_ID
    if meta is not None:
        if "mapping_set_id" in meta.keys():
            mapping_set_id = meta["mapping_set_id"]
        if "license" in meta.keys():
            license = meta["license"]
    return MappingSet(mapping_set_id=mapping_set_id, license=license)


def _get_mdict_ms_and_bad_attrs(
    row: pd.Series, ms: MappingSet, bad_attrs: Counter
) -> Tuple[dict, MappingSet, Counter]:

    mdict = {}

    for k, v in row.items():
        if v and v == v:
            ok = False
            if k:
                k = str(k)
            v = _address_multivalued_slot(k, v)
            # if hasattr(Mapping, k):
            if k in MAPPING_SLOTS:
                mdict[k] = v
                ok = True
            # if hasattr(MappingSet, k):
            if k in MAPPING_SET_SLOTS:
                ms[k] = v
                ok = True
            if not ok:
                bad_attrs[k] += 1
    return (mdict, ms, bad_attrs)


def parse_alignment_xml(
    file_path: str,
    prefix_map: Dict[str, str],
    meta: Dict[str, str],
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
    prefix_map: Optional[PrefixMap] = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Convert a dataframe to a MappingSetDataFrame.

    :param df: A mappings dataframe
    :param prefix_map: A prefix map
    :param meta: A metadata dictionary
    :return: MappingSetDataFrame
    """
    prefix_map = _ensure_prefix_map(prefix_map)

    if "confidence" in df.columns:
        df["confidence"].replace(r"^\s*$", np.NaN, regex=True, inplace=True)

    mlist: List[Mapping] = []
    ms = _init_mapping_set(meta)
    bad_attrs: typing.Counter[str] = Counter()
    for _, row in df.iterrows():
        mdict, ms, bad_attrs = _get_mdict_ms_and_bad_attrs(row, ms, bad_attrs)
        mlist.append(_prepare_mapping(Mapping(**mdict)))
    for k, v in bad_attrs.most_common():
        logging.warning(f"No attr for {k} [{v} instances]")
    # the autogenerated code's type annotations are _really_ messy. This is in fact okay,
    # so with a heavy heart we employ type:ignore
    ms.mappings = mlist  # type:ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    doc = MappingSetDocument(mapping_set=ms, prefix_map=prefix_map)
    return to_mapping_set_dataframe(doc)


def from_sssom_rdf(
    g: Graph,
    prefix_map: Optional[PrefixMap] = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Convert an SSSOM RDF graph into a SSSOM data table.

    :param g: the Graph (rdflib)
    :param prefix_map: A dictionary containing the prefix map, defaults to None
    :param meta: Potentially additional metadata, defaults to None
    :return: MappingSetDataFrame object
    """
    prefix_map = _ensure_prefix_map(prefix_map)

    ms = _init_mapping_set(meta)
    mlist: List[Mapping] = []

    for sx, px, ox in g.triples((None, URIRef(URI_SSSOM_MAPPINGS), None)):
        mdict: Dict[str, Any] = {}
        # TODO replace with g.predicate_objects()
        for _s, p, o in g.triples((ox, None, None)):
            if isinstance(p, URIRef):
                try:
                    p_id = curie_from_uri(p, prefix_map)
                    k = None

                    if p_id.startswith("sssom:"):
                        k = p_id.replace("sssom:", "")
                    elif p_id == "owl:annotatedProperty":
                        k = "predicate_id"
                    elif p_id == "owl:annotatedTarget":
                        k = "object_id"
                    elif p_id == "owl:annotatedSource":
                        k = "subject_id"

                    if isinstance(o, URIRef):
                        v: Any
                        v = curie_from_uri(o, prefix_map)
                    else:
                        v = o.toPython()
                    if k:
                        v = _address_multivalued_slot(k, v)
                        mdict[k] = v

                except NoCURIEException as e:
                    logging.warning(e)
        if mdict:
            m = _prepare_mapping(Mapping(**mdict))
            if _is_valid_mapping(m):
                mlist.append(m)
            else:
                logging.warning(
                    f"While trying to prepare a mapping for {mdict}, something went wrong. "
                    f"One of subject_id, object_id or predicate_id was missing."
                )
        else:
            logging.warning(
                f"While trying to prepare a mapping for {sx},{px}, {ox}, something went wrong. "
                f"This usually happens when a critical prefix_map entry is missing."
            )

    ms.mappings = mlist  # type: ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, prefix_map=prefix_map)
    return to_mapping_set_dataframe(mdoc)


def from_sssom_json(
    jsondoc: Union[str, dict, TextIO],
    prefix_map: Dict[str, str],
    meta: Dict[str, str] = None,
) -> MappingSetDataFrame:
    """Load a mapping set dataframe from a JSON object.

    :param jsondoc: JSON document
    :param prefix_map: Prefix map
    :param meta: metadata
    :return: MappingSetDataFrame object
    """
    prefix_map = _ensure_prefix_map(prefix_map)
    mapping_set = cast(
        MappingSet, JSONLoader().load(source=jsondoc, target_class=MappingSet)
    )

    _set_metadata_in_mapping_set(mapping_set, metadata=meta)
    mapping_set_document = MappingSetDocument(
        mapping_set=mapping_set, prefix_map=prefix_map
    )
    return to_mapping_set_dataframe(mapping_set_document)


def from_alignment_minidom(
    dom: Document,
    prefix_map: PrefixMap,
    meta: MetadataType,
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
    # FIXME: should be prefix_map =  _check_prefix_map(prefix_map)
    _ensure_prefix_map(prefix_map)
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
                            c_node, prefix_map, mapping_predicates=mapping_predicates
                        )
                        if mdict:
                            m = _prepare_mapping(mdict)
                            mlist.append(m)
                        else:
                            logging.warning(
                                f"While trying to prepare a mapping for {c_node}, something went wrong. "
                                f"This usually happens when a critical prefix_map entry is missing."
                            )

                elif node_name == "xml":
                    if e.firstChild.nodeValue != "yes":
                        raise ValueError(
                            "Alignment format: xml element said, but not set to yes. Only XML is supported!"
                        )
                elif node_name == "onto1":
                    ms["subject_source_id"] = e.firstChild.nodeValue
                elif node_name == "onto2":
                    ms["object_source_id"] = e.firstChild.nodeValue
                elif node_name == "uri1":
                    ms["subject_source"] = e.firstChild.nodeValue
                elif node_name == "uri2":
                    ms["object_source"] = e.firstChild.nodeValue

    ms.mappings = mlist  # type: ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mapping_set_document = MappingSetDocument(mapping_set=ms, prefix_map=prefix_map)
    return to_mapping_set_dataframe(mapping_set_document)


def _get_obographs_predicate_id(obographs_predicate: str):
    if obographs_predicate == "is_a":
        return RDFS_SUBCLASS_OF
    return obographs_predicate


def from_obographs(
    jsondoc: Dict,
    prefix_map: PrefixMap,
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
    _ensure_prefix_map(prefix_map)
    ms = _init_mapping_set(meta)
    mlist: List[Mapping] = []
    # bad_attrs = {}

    if not mapping_predicates:
        mapping_predicates = DEFAULT_MAPPING_PROPERTIES

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
                                    mdict["subject_id"] = curie_from_uri(
                                        nid, prefix_map
                                    )
                                    mdict["object_id"] = curie_from_uri(
                                        xref_id, prefix_map
                                    )
                                    mdict["subject_label"] = label
                                    mdict["predicate_id"] = "oboInOwl:hasDbXref"
                                    mdict["match_type"] = MATCH_TYPE_UNSPECIFIED
                                    mlist.append(Mapping(**mdict))
                                except NoCURIEException as e:
                                    # FIXME this will cause all sorts of ragged Mappings
                                    logging.warning(e)
                        if "basicPropertyValues" in n["meta"]:
                            for value in n["meta"]["basicPropertyValues"]:
                                pred = value["pred"]
                                if pred in mapping_predicates:
                                    xref_id = value["val"]
                                    mdict = {}
                                    try:
                                        mdict["subject_id"] = curie_from_uri(
                                            nid, prefix_map
                                        )
                                        mdict["object_id"] = curie_from_uri(
                                            xref_id, prefix_map
                                        )
                                        mdict["subject_label"] = label
                                        mdict["predicate_id"] = curie_from_uri(
                                            pred, prefix_map
                                        )
                                        mdict["match_type"] = MATCH_TYPE_UNSPECIFIED
                                        mlist.append(Mapping(**mdict))
                                    except NoCURIEException as e:
                                        # FIXME this will cause ragged mappings
                                        logging.warning(e)
            elif "edges" in g:
                for edge in g["edges"]:
                    mdict = {}
                    subject_id = edge["sub"]
                    predicate_id = _get_obographs_predicate_id(edge["pred"])
                    object_id = edge["obj"]
                    if predicate_id in mapping_predicates:
                        mdict["subject_id"] = curie_from_uri(subject_id, prefix_map)
                        mdict["object_id"] = curie_from_uri(object_id, prefix_map)
                        mdict["predicate_id"] = curie_from_uri(predicate_id, prefix_map)
                        mdict["match_type"] = MATCH_TYPE_UNSPECIFIED
                        mlist.append(Mapping(**mdict))
            elif "equivalentNodesSets" in g and OWL_EQUIV_CLASS in mapping_predicates:
                for equivalents in g["equivalentNodesSets"]:
                    if "nodeIds" in equivalents:
                        for ec1 in equivalents["nodeIds"]:
                            for ec2 in equivalents["nodeIds"]:
                                if ec1 != ec2:
                                    mdict = {}
                                    mdict["subject_id"] = curie_from_uri(
                                        ec1, prefix_map
                                    )
                                    mdict["object_id"] = curie_from_uri(ec2, prefix_map)
                                    mdict["predicate_id"] = curie_from_uri(
                                        OWL_EQUIV_CLASS, prefix_map
                                    )
                                    mdict["match_type"] = MATCH_TYPE_UNSPECIFIED
                                    mlist.append(Mapping(**mdict))
    else:
        raise Exception("No graphs element in obographs file, wrong format?")

    ms.mappings = mlist  # type: ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, prefix_map=prefix_map)
    return to_mapping_set_dataframe(mdoc)


# All from_* take as an input a python object (data frame, json, etc) and return a MappingSetDataFrame
# All read_* take as an input a a file handle and return a MappingSetDataFrame (usually wrapping a from_* method)


def get_parsing_function(input_format: Optional[str], filename: str) -> Callable:
    """Return appropriate parser function based on input format of file.

    :param input_format: File format
    :param filename: Filename
    :raises Exception: Unknown file format
    :return: Appropriate 'read' function
    """
    if input_format is None:
        input_format = get_file_extension(filename)
    if input_format == "tsv":
        return parse_sssom_table
    elif input_format == "rdf":
        return parse_sssom_rdf
    elif input_format == "json":
        return parse_sssom_json
    elif input_format == "alignment-api-xml":
        return parse_alignment_xml
    elif input_format == "obographs-json":
        return parse_obographs_json
    else:
        raise Exception(f"Unknown input format: {input_format}")


def _ensure_prefix_map(prefix_map: Optional[PrefixMap] = None) -> PrefixMap:
    if not prefix_map:
        raise Exception("No valid prefix_map provided")
    else:
        return add_built_in_prefixes_to_prefix_map(prefix_map)


def _prepare_mapping(mapping: Mapping) -> Mapping:
    p = mapping.predicate_id
    if p == "sssom:superClassOf":
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


def _read_metadata_from_table(path: Union[str, Path]) -> Dict[str, Any]:
    if isinstance(path, Path) or not validators.url(path):
        with open(path) as file:
            yamlstr = ""
            for line in file:
                if line.startswith("#"):
                    yamlstr += re.sub("^#", "", line)
                else:
                    break
    else:
        response = urlopen(path)
        yamlstr = ""
        for lin in response:
            line = lin.decode("utf-8")
            if line.startswith("#"):
                yamlstr += re.sub("^#", "", line)
            else:
                break

    if yamlstr:
        meta = yaml.safe_load(yamlstr)
        logging.info(f"Meta={meta}")
        return meta
    return {}


def _is_valid_mapping(m: Mapping) -> bool:
    return bool(m.predicate_id and m.object_id and m.subject_id)


def _set_metadata_in_mapping_set(
    mapping_set: MappingSet, metadata: Optional[MetadataType] = None
) -> None:
    if metadata is None:
        logging.info("Tried setting metadata but none provided.")
    else:
        for k, v in metadata.items():
            if k != PREFIX_MAP_KEY:
                mapping_set[k] = v


def _cell_element_values(
    cell_node, prefix_map: PrefixMap, mapping_predicates
) -> Optional[Mapping]:
    mdict: Dict[str, Any] = {}
    for child in cell_node.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            try:
                if child.nodeName == "entity1":
                    mdict["subject_id"] = curie_from_uri(
                        child.getAttribute("rdf:resource"), prefix_map
                    )
                elif child.nodeName == "entity2":
                    mdict["object_id"] = curie_from_uri(
                        child.getAttribute("rdf:resource"), prefix_map
                    )
                elif child.nodeName == "measure":
                    mdict["confidence"] = child.firstChild.nodeValue
                elif child.nodeName == "relation":
                    relation = child.firstChild.nodeValue
                    if (relation == "=") and (OWL_EQUIV_CLASS in mapping_predicates):
                        mdict["predicate_id"] = "owl:equivalentClass"
                    else:
                        logging.warning(f"{relation} not a recognised relation type.")
                else:
                    logging.warning(
                        f"Unsupported alignment api element: {child.nodeName}"
                    )
            except NoCURIEException as e:
                logging.warning(e)

    mdict["match_type"] = MATCH_TYPE_UNSPECIFIED

    m = Mapping(**mdict)
    if _is_valid_mapping(m):
        return m
    else:
        return None


# The following methods dont really belong in the parser package..


def to_mapping_set_document(msdf: MappingSetDataFrame) -> MappingSetDocument:
    """Convert a MappingSetDataFrame to a MappingSetDocument."""
    if not msdf.prefix_map:
        raise Exception("No valid prefix_map provided")

    mlist: List[Mapping] = []
    ms = _init_mapping_set(msdf.metadata)
    bad_attrs: Counter = Counter()
    if msdf.df is not None:
        for _, row in msdf.df.iterrows():
            mdict, ms, bad_attrs = _get_mdict_ms_and_bad_attrs(row, ms, bad_attrs)

            m = _prepare_mapping(Mapping(**mdict))
            mlist.append(m)
    for k, v in bad_attrs.items():
        logging.warning(f"No attr for {k} [{v} instances]")
    ms.mappings = mlist  # type: ignore
    if msdf.metadata is not None:
        for k, v in msdf.metadata.items():
            if k != PREFIX_MAP_KEY:
                ms[k] = v
    return MappingSetDocument(mapping_set=ms, prefix_map=msdf.prefix_map)


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
    subject_prefixes = set(msdf.df["subject_id"].str.split(":", 1, expand=True)[0])
    object_prefixes = set(msdf.df["object_id"].str.split(":", 1, expand=True)[0])
    relations = set(msdf.df["predicate_id"])
    return split_dataframe_by_prefix(
        msdf=msdf,
        subject_prefixes=subject_prefixes,
        object_prefixes=object_prefixes,
        relations=relations,
    )


def split_dataframe_by_prefix(
    msdf: MappingSetDataFrame, subject_prefixes, object_prefixes, relations
) -> Dict[str, MappingSetDataFrame]:
    """Split a mapping set dataframe by prefix.

    :param msdf: An SSSOM MappingSetDataFrame
    :param subject_prefixes: a list of prefixes pertaining to the subject
    :param object_prefixes: a list of prefixes pertaining to the object
    :param relations: a list of relations of interest
    :return: a dict of SSSOM data frame names to MappingSetDataFrame
    """
    df = msdf.df
    prefix_map = msdf.prefix_map
    meta = msdf.metadata
    splitted = {}
    for pre_subj in subject_prefixes:
        for pre_obj in object_prefixes:
            for rel in relations:
                relpre = rel.split(":")[0]
                relppost = rel.split(":")[1]
                split_name = f"{pre_subj.lower()}_{relppost.lower()}_{pre_obj.lower()}"
                if df is not None:
                    dfs = df[
                        (df["subject_id"].str.startswith(pre_subj + ":"))
                        & (df["predicate_id"] == rel)
                        & (df["object_id"].str.startswith(pre_obj + ":"))
                    ]
                if pre_subj in prefix_map and pre_obj in prefix_map and len(dfs) > 0:
                    cm = {
                        pre_subj: prefix_map[pre_subj],
                        pre_obj: prefix_map[pre_obj],
                        relpre: prefix_map[relpre],
                    }
                    msdf = from_sssom_dataframe(dfs, prefix_map=cm, meta=meta)
                    splitted[split_name] = msdf
                else:
                    logging.warning(
                        f"Not adding {split_name} because there is a missing prefix ({pre_subj}, {pre_obj}), "
                        f"or no matches ({len(dfs)} matches found)"
                    )
    return splitted
