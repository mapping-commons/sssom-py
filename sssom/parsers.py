import json
import logging
import re
import typing
from typing import Any, Dict, Optional, Set, TextIO, Union
from urllib.request import urlopen
from xml.dom import Node, minidom
from xml.dom.minidom import Document

import numpy as np
import pandas as pd
import validators
import yaml
from linkml_runtime.loaders.json_loader import JSONLoader
from rdflib import Graph, URIRef

from .context import add_built_in_prefixes_to_prefix_map, get_default_metadata
from .sssom_datamodel import Mapping, MappingSet
from .sssom_document import MappingSetDocument
from .util import (
    SSSOM_DEFAULT_RDF_SERIALISATION,
    URI_SSSOM_MAPPINGS,
    MappingSetDataFrame,
    NoCURIEException,
    curie_from_uri,
    get_file_extension,
    raise_for_bad_path,
    read_pandas,
    to_mapping_set_dataframe,
)

# Readers (from file)


def read_sssom_table(
    file_path: str, curie_map: Dict[str, Any] = None, meta: Dict[str, Any] = None
) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    raise_for_bad_path(file_path)
    df = read_pandas(file_path)

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

    curie_map, meta = _get_curie_map_and_metadata(curie_map=curie_map, meta=meta)

    msdf = from_sssom_dataframe(df, curie_map=curie_map, meta=meta)
    return msdf


def read_sssom_rdf(
    file_path: str,
    curie_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
    serialisation=SSSOM_DEFAULT_RDF_SERIALISATION,
) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    raise_for_bad_path(file_path)
    curie_map, meta = _get_curie_map_and_metadata(curie_map=curie_map, meta=meta)

    g = Graph()
    g.load(file_path, format=serialisation)
    # json_obj = json.loads(g.serialize(format="json-ld"))
    # print(json_obj)
    # msdf = from_sssom_json(json_obj, curie_map=curie_map, meta=meta)
    msdf = from_sssom_rdf(g, curie_map=curie_map, meta=meta)
    return msdf


def read_sssom_json(
    file_path: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None
) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    raise_for_bad_path(file_path)
    curie_map, meta = _get_curie_map_and_metadata(curie_map=curie_map, meta=meta)

    with open(file_path) as json_file:
        jsondoc = json.load(json_file)
    msdf = from_sssom_json(jsondoc=jsondoc, curie_map=curie_map, meta=meta)
    return msdf


# Import methods from external file formats


def read_obographs_json(
    file_path: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None
) -> MappingSetDataFrame:
    """
    parses an obographs file as a JSON object and translates it into a MappingSetDataFrame
    :param file_path: The path to the obographs file
    :param curie_map: an optional curie map
    :param meta: an optional dictionary of metadata elements
    :return: A SSSOM MappingSetDataFrame
    """
    raise_for_bad_path(file_path)

    curie_map, meta = _get_curie_map_and_metadata(curie_map=curie_map, meta=meta)

    with open(file_path) as json_file:
        jsondoc = json.load(json_file)

    return from_obographs(jsondoc, curie_map, meta)


def _get_curie_map_and_metadata(curie_map: Dict[str, Any], meta: Dict[str, Any]):
    default_meta, default_curie_map = get_default_metadata()

    if not curie_map:
        logging.warning(
            "No curie map provided (not recommended), trying to use defaults.."
        )
        curie_map = default_curie_map

    if not meta:
        meta = default_meta
    else:
        if curie_map and "curie_map" in meta:
            logging.info(
                "Curie map prvoided as parameter, but SSSOM file provides its own CURIE map. "
                "CURIE map provided externally is disregarded in favour of the curie map in the SSSOM file."
            )
            curie_map = meta["curie_map"]

    return curie_map, meta


def read_alignment_xml(
    file_path: str, curie_map: Dict[str, str], meta: Dict[str, str]
) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    raise_for_bad_path(file_path)

    curie_map, meta = _get_curie_map_and_metadata(curie_map=curie_map, meta=meta)
    logging.info("Loading from alignment API")
    xmldoc = minidom.parse(file_path)
    msdf = from_alignment_minidom(xmldoc, curie_map, meta)
    return msdf


# Readers (from object)


def from_sssom_dataframe(
    df: pd.DataFrame, curie_map: Dict[str, str], meta: Dict[str, str]
) -> MappingSetDataFrame:
    """
    Converts a dataframe to a MappingSetDataFrame
    :param df:
    :param curie_map:
    :param meta:
    :return: MappingSetDataFrame
    """

    _check_curie_map(curie_map)

    if "confidence" in df.columns:
        df["confidence"].replace(r"^\s*$", np.NaN, regex=True, inplace=True)

    mlist = []
    ms = MappingSet()
    bad_attrs = {}
    for _, row in df.iterrows():
        mdict = {}
        for k, v in row.items():
            ok = False
            if k:
                k = str(k)
            # if k.endswith('_id'): # TODO: introspect
            #    v = Entity(id=v)
            if hasattr(Mapping, k):
                mdict[k] = v
                ok = True
            if hasattr(MappingSet, k):
                ms[k] = v
                ok = True
            if not ok:
                if k not in bad_attrs:
                    bad_attrs[k] = 1
                else:
                    bad_attrs[k] += 1
        m = _prepare_mapping(Mapping(**mdict))

        mlist.append(m)
    for k, v in bad_attrs.items():
        logging.warning(f"No attr for {k} [{v} instances]")
    ms.mappings = mlist
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    doc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(doc)


def from_sssom_rdf(
    g: Graph,
    curie_map: Dict[str, str] = None,
    meta: Dict[str, str] = None,
    mapping_predicates: Set[str] = None,
) -> MappingSetDataFrame:
    """
    Converts an SSSOM RDF graph into a SSSOM data table
    Args:
        g: the Grah (rdflib)
        curie_map: A dictionary conatining the prefix map
        meta: Potentially additional metadata
        mapping_predicates: A set of predicates that should be extracted from the RDF graph

    Returns:

    """
    curie_map = _check_curie_map(curie_map)

    if mapping_predicates is None:
        # FIXME unused
        mapping_predicates = _get_default_mapping_predicates()

    ms = MappingSet()
    mlist = []

    for sx, px, ox in g.triples((None, URIRef(URI_SSSOM_MAPPINGS), None)):
        mdict = {}
        # TODO replace with g.predicate_objects()
        for _s, p, o in g.triples((ox, None, None)):
            if isinstance(p, URIRef):
                try:
                    p_id = curie_from_uri(p, curie_map)
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
                        v = curie_from_uri(o, curie_map)
                    else:
                        v = o.toPython()

                    if k:
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
                f"This usually happens when a critical curie_map entry is missing."
            )

    ms.mappings = mlist
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


def from_sssom_json(
    jsondoc: Union[str, dict, TextIO], curie_map: Dict[str, str], meta: Dict[str, str] = None
) -> MappingSetDataFrame:
    _check_curie_map(curie_map)

    # noinspection PyTypeChecker
    ms = JSONLoader().load(source=jsondoc, target_class=MappingSet)

    _set_metadata_in_mapping_set(ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


def from_alignment_minidom(
    dom: Document, curie_map: Dict[str, str], meta: Dict[str, str]
) -> MappingSetDataFrame:
    """
    Reads a minidom Document object
    :param dom: XML (minidom) object
    :param curie_map:
    :param meta: Optional meta data
    :return: MappingSetDocument
    """
    _check_curie_map(curie_map)  # FIXME: should be curie_map =  _check_curie_map(curie_map)

    ms = MappingSet()
    mlist = []
    # bad_attrs = {}

    alignments = dom.getElementsByTagName("Alignment")
    for n in alignments:
        for e in n.childNodes:
            if e.nodeType == Node.ELEMENT_NODE:
                node_name = e.nodeName
                if node_name == "map":
                    cell = e.getElementsByTagName("Cell")
                    for c_node in cell:
                        mdict = _cell_element_values(c_node, curie_map)
                        if mdict:
                            m = _prepare_mapping(mdict)
                            mlist.append(m)
                        else:
                            logging.warning(
                                f"While trying to prepare a mapping for {c_node}, something went wrong. "
                                f"This usually happens when a critical curie_map entry is missing."
                            )

                elif node_name == "xml":
                    if e.firstChild.nodeValue != "yes":
                        raise Exception(
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

    ms.mappings = mlist
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


def from_obographs(
    jsondoc: Dict, curie_map: Dict[str, str], meta: Dict[str, str] = None
) -> MappingSetDataFrame:
    """
    Converts a obographs json object to an SSSOM data frame

    Args:
        jsondoc: The JSON object representing the ontology in obographs format
        curie_map: The curie map to be used
        meta: Any additional metadata that needs to be added to the resulting SSSOM data frame

    Returns:
        An SSSOM data frame (MappingSetDataFrame)

    """
    _check_curie_map(curie_map)

    ms = MappingSet()
    mlist = []
    # bad_attrs = {}

    allowed_properties = [
        "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
        "http://www.w3.org/2004/02/skos/core#exactMatch",
        "http://www.w3.org/2004/02/skos/core#broadMatch",
        "http://www.w3.org/2004/02/skos/core#closeMatch",
        "http://www.w3.org/2004/02/skos/core#narrowMatch",
        "http://www.w3.org/2004/02/skos/core#relatedMatch",
    ]

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
                        if "xrefs" in n["meta"]:
                            for xref in n["meta"]["xrefs"]:
                                xref_id = xref["val"]
                                mdict = {}
                                try:
                                    mdict["subject_id"] = curie_from_uri(nid, curie_map)
                                    mdict["object_id"] = curie_from_uri(
                                        xref_id, curie_map
                                    )
                                    mdict["subject_label"] = label
                                    mdict["predicate_id"] = "oboInOwl:hasDbXref"
                                    mdict["match_type"] = "Unspecified"
                                    mlist.append(Mapping(**mdict))
                                except NoCURIEException as e:
                                    logging.warning(e)
                        if "basicPropertyValues" in n["meta"]:
                            for value in n["meta"]["basicPropertyValues"]:
                                pred = value["pred"]
                                if pred in allowed_properties:
                                    xref_id = value["val"]
                                    mdict = {}
                                    try:
                                        mdict["subject_id"] = curie_from_uri(
                                            nid, curie_map
                                        )
                                        mdict["object_id"] = curie_from_uri(
                                            xref_id, curie_map
                                        )
                                        mdict["subject_label"] = label
                                        mdict["predicate_id"] = curie_from_uri(
                                            pred, curie_map
                                        )
                                        mdict["match_type"] = "Unspecified"
                                        mlist.append(Mapping(**mdict))
                                    except NoCURIEException as e:
                                        logging.warning(e)
    else:
        raise Exception("No graphs element in obographs file, wrong format?")

    ms.mappings = mlist
    _set_metadata_in_mapping_set(mapping_set=ms, metadata=meta)
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


# All from_* take as an input a python object (data frame, json, etc) and return a MappingSetDataFrame
# All read_* take as an input a a file handle and return a MappingSetDataFrame (usually wrapping a from_* method)


def get_parsing_function(input_format, filename):
    if input_format is None:
        input_format = get_file_extension(filename)
    if input_format == "tsv":
        return read_sssom_table
    elif input_format == "rdf":
        return read_sssom_rdf
    elif input_format == "json":
        return read_sssom_json
    elif input_format == "alignment-api-xml":
        return read_alignment_xml
    elif input_format == "obographs-json":
        return read_obographs_json
    else:
        raise Exception(f"Unknown input format: {input_format}")


def _check_curie_map(curie_map):
    if not curie_map:
        raise Exception("No valid curie_map provided")
    else:
        return add_built_in_prefixes_to_prefix_map(curie_map)


def _get_default_mapping_predicates():
    return {
        "oio:hasDbXref",
        "skos:exactMatch",
        "skos:narrowMatch",
        "skos:broadMatch",
        "skos:exactMatch",
        "skos:closeMatch",
        "owl:sameAs",
        "owl:equivalentClass",
        "owl:equivalentProperty",
    }


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


def _read_metadata_from_table(filename: str) -> Any:
    if validators.url(filename):
        response = urlopen(filename)
        yamlstr = ""
        for lin in response:
            line = lin.decode("utf-8")
            if line.startswith("#"):
                yamlstr += re.sub("^#", "", line)
            else:
                break
    else:
        with open(filename, "r") as s:
            yamlstr = ""
            for line in s:
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


def _set_metadata_in_mapping_set(mapping_set: MappingSet, metadata: Dict[str, str]) -> None:
    if not metadata:
        logging.info("Tried setting metadata but none provided.")
    else:
        for k, v in metadata.items():
            if k != "curie_map":
                mapping_set[k] = v


def _cell_element_values(cell_node, curie_map: Dict[str, str]) -> Optional[Mapping]:
    mdict = {}
    for child in cell_node.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            try:
                if child.nodeName == "entity1":
                    mdict["subject_id"] = curie_from_uri(
                        child.getAttribute("rdf:resource"), curie_map
                    )
                elif child.nodeName == "entity2":
                    mdict["object_id"] = curie_from_uri(
                        child.getAttribute("rdf:resource"), curie_map
                    )
                elif child.nodeName == "measure":
                    mdict["confidence"] = child.firstChild.nodeValue
                elif child.nodeName == "relation":
                    relation = child.firstChild.nodeValue
                    if relation == "=":
                        mdict["predicate_id"] = "owl:equivalentClass"
                    else:
                        logging.warning(f"{relation} not a recognised relation type.")
                else:
                    logging.warning(
                        f"Unsupported alignment api element: {child.nodeName}"
                    )
            except NoCURIEException as e:
                logging.warning(e)

    m = Mapping(**mdict)
    if _is_valid_mapping(m):
        return m
    else:
        return None


# The following methods dont really belong in the parser package..


def to_mapping_set_document(msdf: MappingSetDataFrame) -> MappingSetDocument:
    """Convert a MappingSetDataFrame to a MappingSetDocument."""
    if not msdf.prefixmap:
        raise Exception("No valid curie_map provided")

    mlist = []
    ms = MappingSet()
    bad_attrs = {}
    if msdf.df is not None:
        for _, row in msdf.df.iterrows():
            mdict = {}
            for k, v in row.items():
                ok = False
                if k:
                    k = str(k)
                if hasattr(Mapping, k):
                    mdict[k] = v
                    ok = True
                if hasattr(MappingSet, k):
                    ms[k] = v
                    ok = True
                if not ok:
                    if k not in bad_attrs:
                        bad_attrs[k] = 1
                    else:
                        bad_attrs[k] += 1
            m = _prepare_mapping(Mapping(**mdict))
            mlist.append(m)
    for k, v in bad_attrs.items():
        logging.warning(f"No attr for {k} [{v} instances]")
    ms.mappings = mlist
    if msdf.metadata is not None:
        for k, v in msdf.metadata.items():
            if k != "curie_map":
                ms[k] = v
    return MappingSetDocument(mapping_set=ms, curie_map=msdf.prefixmap)


def split_dataframe(
    msdf: MappingSetDataFrame,
) -> typing.Mapping[str, MappingSetDataFrame]:
    df = msdf.df
    if df is not None:
        subject_prefixes = set(df["subject_id"].str.split(":", 1, expand=True)[0])
        object_prefixes = set(df["object_id"].str.split(":", 1, expand=True)[0])
        relations = set(df["predicate_id"])
    return split_dataframe_by_prefix(
        msdf=msdf,
        subject_prefixes=subject_prefixes,
        object_prefixes=object_prefixes,
        relations=relations,
    )


def split_dataframe_by_prefix(
    msdf: MappingSetDataFrame, subject_prefixes, object_prefixes, relations
) -> typing.Mapping[str, MappingSetDataFrame]:
    """

    :param msdf: An SSSOM MappingSetDataFrame
    :param subject_prefixes: a list of prefixes pertaining to the subject
    :param object_prefixes: a list of prefixes pertaining to the object
    :param relations: a list of relations of interest
    :return: a dict of SSSOM data frame names to MappingSetDataFrame
    """
    df = msdf.df
    curie_map = msdf.prefixmap
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
                if pre_subj in curie_map and pre_obj in curie_map and len(dfs) > 0:
                    cm = {
                        pre_subj: curie_map[pre_subj],
                        pre_obj: curie_map[pre_obj],
                        relpre: curie_map[relpre],
                    }
                    msdf = from_sssom_dataframe(dfs, curie_map=cm, meta=meta)
                    splitted[split_name] = msdf
                else:
                    logging.warning(
                        f"Not adding {split_name} because there is a missing prefix ({pre_subj}, {pre_obj}), "
                        f"or no matches ({len(dfs)} matches found)"
                    )
    return splitted
