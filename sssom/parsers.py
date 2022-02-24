"""SSSOM parsers."""

import json
import logging
import re
import typing
from collections import Counter
from dateutil import parser as date_parser
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

# TODO: PR comment: where matchtypeenum? can't find sssomschema, Mapping, or MappingSet. only MappingSetDataFrame
# from .sssom_datamodel import Mapping, MappingSet, MatchTypeEnum
from sssom_schema import Mapping, MappingSet


from sssom.constants import (
    CONFIDENCE,
    CURIE_MAP,
    DEFAULT_MAPPING_PROPERTIES,
    LICENSE,
    MAPPING_JUSTIFICATION,
    MAPPING_JUSTIFICATION_UNSPECIFIED,
    MAPPING_SET_ID,
    MAPPING_SET_SLOTS,
    MAPPING_SLOTS,
    OBJECT_ID,
    OBJECT_SOURCE,
    OBJECT_SOURCE_ID,
    OWL_EQUIV_CLASS,
    PREDICATE_ID,
    RDFS_SUBCLASS_OF,
    SUBJECT_ID,
    SUBJECT_LABEL,
    SUBJECT_SOURCE,
    SUBJECT_SOURCE_ID,
)

from .context import (
    DEFAULT_LICENSE,
    DEFAULT_MAPPING_SET_ID,
    add_built_in_prefixes_to_prefix_map,
    get_default_metadata,
)
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

        if "curie_map" in sssom_metadata:
            if prefix_map:
                for k, v in prefix_map.items():
                    if k in sssom_metadata[CURIE_MAP]:
                        if sssom_metadata[CURIE_MAP][k] != v:
                            logging.warning(
                                f"SSSOM prefix map {k} ({sssom_metadata[CURIE_MAP][k]}) "
                                f"conflicts with provided ({prefix_map[k]})."
                            )
                    else:
                        logging.info(
                            f"Externally provided metadata {k}:{v} is added to metadata set."
                        )
                        sssom_metadata[CURIE_MAP][k] = v
            prefix_map = sssom_metadata[CURIE_MAP]

    meta_all = _get_prefix_map_and_metadata(prefix_map=prefix_map, meta=meta)
    msdf = from_sssom_dataframe(
        df, prefix_map=meta_all.prefix_map, meta=meta_all.metadata
    )
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


def parse_snomed_icd10cm_map_tsv(
    file_path: str,
    prefix_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
) -> MappingSetDataFrame:
    """Parse special SNOMED ICD10CM mapping file and translates it into a MappingSetDataFrame.

    :param file_path: The path to the obographs file
    :param prefix_map: an optional prefix map
    :param meta: an optional dictionary of metadata elements
    :return: A SSSOM MappingSetDataFrame
    """
    raise_for_bad_path(file_path)
    df = read_pandas(file_path)
    df2 = from_snomed_icd10cm_map_tsv(df, prefix_map=prefix_map, meta=meta)
    return df2


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


def _address_multivalued_slot(k: str, v: Any) -> Union[str, List[str]]:
    if is_multivalued_slot(k) and v is not None and isinstance(v, str):
        # IF k is multivalued, then v = List[values]
        return [s.strip() for s in v.split("|")]
    else:
        return v


def _init_mapping_set(meta: Optional[MetadataType]) -> MappingSet:
    license = DEFAULT_LICENSE
    mapping_set_id = DEFAULT_MAPPING_SET_ID
    if meta is not None:
        if MAPPING_SET_ID in meta.keys():
            mapping_set_id = meta[MAPPING_SET_ID]
        if LICENSE in meta.keys():
            license = meta[LICENSE]
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
                    ms[SUBJECT_SOURCE_ID] = e.firstChild.nodeValue
                elif node_name == "onto2":
                    ms[OBJECT_SOURCE_ID] = e.firstChild.nodeValue
                elif node_name == "uri1":
                    ms[SUBJECT_SOURCE] = e.firstChild.nodeValue
                elif node_name == "uri2":
                    ms[OBJECT_SOURCE] = e.firstChild.nodeValue

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
                                    mdict[SUBJECT_ID] = curie_from_uri(nid, prefix_map)
                                    mdict[OBJECT_ID] = curie_from_uri(
                                        xref_id, prefix_map
                                    )
                                    mdict[SUBJECT_LABEL] = label
                                    mdict[PREDICATE_ID] = "oboInOwl:hasDbXref"
                                    mdict[
                                        MAPPING_JUSTIFICATION
                                    ] = MAPPING_JUSTIFICATION_UNSPECIFIED
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
                                        mdict[SUBJECT_ID] = curie_from_uri(
                                            nid, prefix_map
                                        )
                                        mdict[OBJECT_ID] = curie_from_uri(
                                            xref_id, prefix_map
                                        )
                                        mdict[SUBJECT_LABEL] = label
                                        mdict[PREDICATE_ID] = curie_from_uri(
                                            pred, prefix_map
                                        )
                                        mdict[
                                            MAPPING_JUSTIFICATION
                                        ] = MAPPING_JUSTIFICATION_UNSPECIFIED
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
                        mdict[SUBJECT_ID] = curie_from_uri(subject_id, prefix_map)
                        mdict[OBJECT_ID] = curie_from_uri(object_id, prefix_map)
                        mdict[PREDICATE_ID] = curie_from_uri(predicate_id, prefix_map)
                        mdict[MAPPING_JUSTIFICATION] = MAPPING_JUSTIFICATION_UNSPECIFIED
                        mlist.append(Mapping(**mdict))
            elif "equivalentNodesSets" in g and OWL_EQUIV_CLASS in mapping_predicates:
                for equivalents in g["equivalentNodesSets"]:
                    if "nodeIds" in equivalents:
                        for ec1 in equivalents["nodeIds"]:
                            for ec2 in equivalents["nodeIds"]:
                                if ec1 != ec2:
                                    mdict = {}
                                    mdict[SUBJECT_ID] = curie_from_uri(ec1, prefix_map)
                                    mdict[OBJECT_ID] = curie_from_uri(ec2, prefix_map)
                                    mdict[PREDICATE_ID] = curie_from_uri(
                                        OWL_EQUIV_CLASS, prefix_map
                                    )
                                    mdict[
                                        MAPPING_JUSTIFICATION
                                    ] = MAPPING_JUSTIFICATION_UNSPECIFIED
                                    mlist.append(Mapping(**mdict))
    else:
        raise Exception("No graphs element in obographs file, wrong format?")

    ms.mappings = mlist  # type: ignore
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, prefix_map=prefix_map)
    return to_mapping_set_dataframe(mdoc)


def from_snomed_icd10cm_map_tsv(
    df: pd.DataFrame,
    prefix_map: Optional[PrefixMap] = None,
    meta: Optional[MetadataType] = None,
) -> MappingSetDataFrame:
    """Convert a snomed_icd10cm_map dataframe to a MappingSetDataFrame.

    :param df: A mappings dataframe
    :param prefix_map: A prefix map
    :param meta: A metadata dictionary
    :return: MappingSetDataFrame

    # Field descriptions
    # - Taken from: doc_Icd10cmMapReleaseNotes_Current-en-US_US1000124_20210901.pdf
    FIELD,DATA_TYPE,PURPOSE,Joe's comments
    - id,UUID,A 128 bit unsigned integer, uniquely identifying the map record,
    - effectiveTime,Time,Specifies the inclusive date at which this change becomes effective.,
    - active,Boolean,Specifies whether the member’s state was active (=1) or inactive (=0) from the nominal release date
     specified by the effectiveTime field.,
    - moduleId,SctId,Identifies the member version’s module. Set to a child of 900000000000443000|Module| within the
    metadata hierarchy.,The only value in the entire set is '5991000124107', which has label 'SNOMED CT to ICD-10-CM
    rule-based mapping module' (
    https://www.findacode.com/snomed/5991000124107--snomed-ct-to-icd-10-cm-rule-based-mapping-module.html).
    - refSetId,SctId,Set to one of the children of the |Complex map type| concept in the metadata hierarchy.,The only
    value in the entire set is '5991000124107', which has label 'ICD-10-CM complex map reference set' (
    https://www.findacode.com/snomed/6011000124106--icd-10-cm-complex-map-reference-set.html).
    - referencedComponentId,SctId,The SNOMED CT source concept ID that is the subject of the map record.,
    - mapGroup,Integer,An integer identifying a grouping of complex map records which will designate one map target at
    the time of map rule evaluation. Source concepts that require two map targets for classification will have two sets
    of map groups.,
    - mapPriority,Integer,Within a map group, the mapPriority specifies the order in which complex map records should be
    evaluated to determine the correct map target.,
    - mapRule,String,A machine-readable rule, (evaluating to either ‘true’ or ‘false’ at run-time) that indicates
    whether this map record should be selected within its map group.,
    - mapAdvice,String,Human-readable advice that may be employed by the software vendor to give an end-user advice on
    selection of the appropriate target code. This includes a) a summary statement of the map rule logic, b) a statement
    of any limitations of the map record and c) additional classification guidance for the coding professional.,
    - mapTarget,String,The target ICD-10 classification code of the map record.,
    - correlationId,SctId,A child of |Map correlation value| in the metadata hierarchy, identifying the correlation
    between the SNOMED CT concept and the target code.,
    - mapCategoryId,SctId,Identifies the SNOMED CT concept in the metadata hierarchy which is the MapCategory for the
    associated map record. This is a subtype of 447634004 |ICD-10 Map Category value|.,
    """
    # https://www.findacode.com/snomed/447561005--snomed-ct-source-code-to-target-map-correlation-not-specified.html
    match_type_snomed_unspecified_id = 447561005
    prefix_map = _ensure_prefix_map(prefix_map)
    ms = _init_mapping_set(meta)

    mlist: List[Mapping] = []
    for _, row in df.iterrows():
        mdict = {
            'subject_id': f'SNOMED:{row["referencedComponentId"]}',
            'subject_label': row['referencedComponentName'],

            # 'predicate_id': 'skos:exactMatch',
            # - mapCategoryId: can use for mapping predicate? Or is correlationId more suitable?
            #   or is there a SKOS predicate I can map to in case where predicate is unknown? I think most of these
            #   mappings are attempts at exact matches, but I can't be sure (at least not without using these fields
            #   to determine: mapGroup, mapPriority, mapRule, mapAdvice).
            # mapCategoryId,mapCategoryName: Only these in set: 447637006 "MAP SOURCE CONCEPT IS PROPERLY CLASSIFIED",
            #   447638001 "MAP SOURCE CONCEPT CANNOT BE CLASSIFIED WITH AVAILABLE DATA",
            #   447639009 "MAP OF SOURCE CONCEPT IS CONTEXT DEPENDENT"
            # 'predicate_modifier': '???',
            #   Description: Modifier for negating the prediate. See https://github.com/mapping-commons/sssom/issues/40
            #   Range: PredicateModifierEnum: (joe: only lists 'Not' as an option)
            #   Example: Not Negates the predicate, see documentation of predicate_modifier_enum
            # - predicate_id <- mapAdvice?
            # - predicate_modifier <- mapAdvice?
            #   mapAdvice: Pipe-delimited qualifiers. Ex:
            #   "ALWAYS Q71.30 | CONSIDER LATERALITY SPECIFICATION"
            #   "IF LISSENCEPHALY TYPE 3 FAMILIAL FETAL AKINESIA SEQUENCE SYNDROME CHOOSE Q04.3 | MAP OF SOURCE CONCEPT
            #   IS CONTEXT DEPENDENT"
            #   "MAP SOURCE CONCEPT CANNOT BE CLASSIFIED WITH AVAILABLE DATA"
            'predicate_id': f'SNOMED:{row["mapCategoryId"]}',
            'predicate_label': row['mapCategoryName'],

            'object_id': f'ICD10CM:{row["mapTarget"]}',
            'object_label': row['mapTargetName'],

            # match_type <- mapRule?
            #   ex: TRUE: when "ALWAYS <code>" is in pipe-delimited list in mapAdvice, this always shows TRUE. Does this
            #       mean I could use skos:exactMatch in these cases?
            # match_type <- correlationId?: This may look redundant, but I want to be explicit. In officially downloaded
            #   SNOMED mappings, all of them had correlationId of 447561005, which also happens to be 'unspecified'.
            #   If correlationId is indeed more appropriate for predicate_id, then I don't think there is a representative
            #   field for 'match_type'.
            'match_type': MatchTypeEnum('Unspecified') if row['correlationId'] == match_type_snomed_unspecified_id \
                else  MatchTypeEnum('Unspecified'),

            'mapping_date': date_parser.parse(str(row['effectiveTime'])).date(),
            'other': '|'.join([f'{k}={str(row[k])}' for k in [
                'id',
                'active',
                'moduleId',
                'refsetId',
                'mapGroup',
                'mapPriority',
                'mapRule',
                'mapAdvice',
            ]]),

            # More fields (https://mapping-commons.github.io/sssom/Mapping/):
            # - subject_category: absent
            # - author_id: can this be "SNOMED"?
            # - author_label: can this be "SNOMED"?
            # - reviewer_id: can this be "SNOMED"?
            # - reviewer_label: can this be "SNOMED"?
            # - creator_id: can this be "SNOMED"?
            # - creator_label: can this be "SNOMED"?
            # - license: Is this something that can be determined?
            # - subject_source: URL of some official page for SNOMED version used?
            # - subject_source_version: Is this knowable?
            # - objectCategory <= mapRule?
            #   mapRule: ex: TRUE: when "ALWAYS <code>" is in pipe-delimited list in mapAdvice, this always shows TRUE.
            #     Does this mean I could use skos:exactMatch in these cases?
            #     object_category:
            #   objectCategory:
            #     Description: The conceptual category to which the subject belongs to. This can be a string denoting
            #     the category or a term from a controlled vocabulary.
            #     Example: UBERON:0001062 (The CURIE of the Uberon term for "anatomical entity".)
            # - object_source: URL of some official page for ICD10CM version used?
            # - object_source_version: would this be "10CM" as in "ICD10CM"? Or something else? Or nothing?
            # - mapping_provider: can this be "SNOMED"?
            # - mapping_cardinality: Could I determine 1:1 or 1:many or many:1 based on:
            #   mapGroup, mapPriority, mapRule, mapAdvice?
            # - match_term_type: What is this?
            # - see_also: Should this be a URL to the SNOMED term?
            # - comment: Description: Free text field containing either curator notes or text generated by tool providing
            #   additional informative information.
        }
        mlist.append(_prepare_mapping(Mapping(**mdict)))

    ms.mappings = mlist
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    doc = MappingSetDocument(mapping_set=ms, prefix_map=prefix_map)
    return to_mapping_set_dataframe(doc)


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
    elif input_format == "snomed-icd10cm-map-tsv":
        return parse_snomed_icd10cm_map_tsv

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
                    mdict[SUBJECT_ID] = curie_from_uri(
                        child.getAttribute("rdf:resource"), prefix_map
                    )
                elif child.nodeName == "entity2":
                    mdict[OBJECT_ID] = curie_from_uri(
                        child.getAttribute("rdf:resource"), prefix_map
                    )
                elif child.nodeName == "measure":
                    mdict[CONFIDENCE] = child.firstChild.nodeValue
                elif child.nodeName == "relation":
                    relation = child.firstChild.nodeValue
                    if (relation == "=") and (OWL_EQUIV_CLASS in mapping_predicates):
                        mdict[PREDICATE_ID] = "owl:equivalentClass"
                    else:
                        logging.warning(f"{relation} not a recognised relation type.")
                else:
                    logging.warning(
                        f"Unsupported alignment api element: {child.nodeName}"
                    )
            except NoCURIEException as e:
                logging.warning(e)

    mdict[MAPPING_JUSTIFICATION] = MAPPING_JUSTIFICATION_UNSPECIFIED

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
                ms[k] = _address_multivalued_slot(k, v)
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
    subject_prefixes = set(msdf.df[SUBJECT_ID].str.split(":", 1, expand=True)[0])
    object_prefixes = set(msdf.df[OBJECT_ID].str.split(":", 1, expand=True)[0])
    relations = set(msdf.df[PREDICATE_ID])
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
                        (df[SUBJECT_ID].str.startswith(pre_subj + ":"))
                        & (df[PREDICATE_ID] == rel)
                        & (df[OBJECT_ID].str.startswith(pre_obj + ":"))
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
