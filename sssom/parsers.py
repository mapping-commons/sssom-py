import logging
import os
import re
from typing import Dict, Set
from xml.dom import minidom, Node
from xml.dom.minidom import Document
import json

import pandas as pd
import yaml
from rdflib import Graph, URIRef

from .sssom_document import MappingSet, Mapping, MappingSetDocument

from .util import RDF_FORMATS
from .util import MappingSetDataFrame, get_file_extension, to_mapping_set_dataframe

from sssom.util import read_pandas
import validators

cwd = os.path.abspath(os.path.dirname(__file__))


# Readers (from file)


def from_tsv(file_path: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    if validators.url(file_path) or os.path.exists(file_path):
        df = read_pandas(file_path)
        if meta is None:
            meta = _read_metadata_from_table(file_path)
        if 'curie_map' in meta:
            logging.info("Context provided, but SSSOM file provides its own CURIE map. "
                         "CURIE map from context is disregarded.")
            curie_map = meta['curie_map']
        msdf = from_dataframe(df, curie_map=curie_map, meta=meta)
        # msdf = to_mapping_set_dataframe(doc) # Creates a MappingSetDataFrame object
        return msdf
    else:
        raise Exception(f"{file_path} is not a valid file path or url.")


def from_rdf(file_path: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    if validators.url(file_path) or os.path.exists(file_path):
        g = Graph()
        file_format = guess_file_format(file_path)
        g.parse(file_path, format=file_format)
        msdf = from_rdf_graph(g, curie_map, meta)
        # msdf = to_mapping_set_dataframe(doc) # Creates a MappingSetDataFrame object
        return msdf
    else:
        raise Exception(f"{file_path} is not a valid file path or url.")


def from_owl(file_path: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    if validators.url(file_path) or os.path.exists(file_path):
        g = Graph()
        file_format = guess_file_format(file_path)
        g.parse(file_path, format=file_format)
        msdf = from_owl_graph(g, curie_map, meta)
        # msdf = to_mapping_set_dataframe(doc) # Creates a MappingSetDataFrame object
        return msdf
    else:
        raise Exception(f"{file_path} is not a valid file path or url.")


def from_jsonld(file_path: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    if validators.url(file_path) or os.path.exists(file_path):
        g = Graph()
        g.parse(file_path, format="json-ld")
        msdf = from_rdf_graph(g, curie_map, meta)
        return msdf
    else:
        raise Exception(f"{file_path} is not a valid file path or url.")


def from_obographs_json(file_path: str, curie_map: Dict[str, str] = None,
                        meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses an obographs file as a JSON object and translates it into a MappingSetDataFrame
    :param file_path: The path to the obographs file
    :param curie_map: an optional curie map
    :param meta: an optional dictionary of metadata elements
    :return: A SSSOM MappingSetDataFrame
    """
    if validators.url(file_path) or os.path.exists(file_path):
        with open(file_path) as json_file:
            jsondoc = json.load(json_file)

        return from_obographs(jsondoc, curie_map, meta)
    else:
        raise Exception(f"{file_path} is not a valid file path or url.")


def guess_file_format(filename):
    extension = get_file_extension(filename)
    if extension in ["owl", "rdf"]:
        return "xml"
    elif extension in RDF_FORMATS:
        return extension
    else:
        raise Exception(f"File extension {extension} does not correspond to a legal file format")


def from_alignment_xml(file_path: str, curie_map: Dict[str, str] = None,
                       meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    if validators.url(file_path) or os.path.exists(file_path):
        logging.info("Loading from alignment API")
        xmldoc = minidom.parse(file_path)
        msdf = from_alignment_minidom(xmldoc, curie_map, meta)
        return msdf
    else:
        raise Exception(f"{file_path} is not a valid file path or url.")


def from_alignment_minidom(dom: Document, curie_map: Dict[str, str] = None,
                           meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    Reads a minidom Document object
    :param dom: XML (minidom) object
    :param curie_map:
    :param meta: Optional meta data
    :return: MappingSetDocument
    """
    if not curie_map:
        raise Exception(f'No valid curie_map provided')

    ms = MappingSet()
    mlist = []
    bad_attrs = {}

    alignments = dom.getElementsByTagName('Alignment')
    for n in alignments:
        for e in n.childNodes:
            if e.nodeType == Node.ELEMENT_NODE:
                node_name = e.nodeName
                if node_name == "map":
                    cell = e.getElementsByTagName('Cell')
                    for c_node in cell:
                        m = _prepare_mapping(_cell_element_values(c_node, curie_map))
                        mlist.append(m)
                elif node_name == "xml":
                    if e.firstChild.nodeValue != "yes":
                        raise Exception(
                            f"Alignment format: xml element said, but not set to yes. Only XML is supported!")
                elif node_name == "onto1":
                    ms["subject_source_id"] = e.firstChild.nodeValue
                elif node_name == "onto2":
                    ms["object_source_id"] = e.firstChild.nodeValue
                elif node_name == "uri1":
                    ms["subject_source"] = e.firstChild.nodeValue
                elif node_name == "uri2":
                    ms["object_source"] = e.firstChild.nodeValue

    ms.mappings = mlist
    if meta:
        for k, v in meta.items():
            if k != 'curie_map':
                ms[k] = v
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


# Readers (from object)

def from_dataframe(df: pd.DataFrame, curie_map: Dict[str, str], meta: Dict[str, str]) -> MappingSetDataFrame:
    """
    Converts a dataframe to a MappingSetDataFrame
    :param df:
    :param curie_map:
    :param meta:
    :return: MappingSetDataFrame
    """
    if not curie_map:
        raise Exception(f'No valid curie_map provided')

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
        # logging.info(f'Row={mdict}')
        m = _prepare_mapping(Mapping(**mdict))
        mlist.append(m)
    for k, v in bad_attrs.items():
        logging.warning(f'No attr for {k} [{v} instances]')
    ms.mappings = mlist
    for k, v in meta.items():
        if k != 'curie_map':
            ms[k] = v
    doc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(doc)


def is_extract_property(p, properties):
    if not properties or p in properties:
        return True
    else:
        return False


def from_obographs(jsondoc: Dict, curie_map: Dict[str, str], meta: Dict[str, str]) -> MappingSetDataFrame:
    """
    Converts a obographs json object to an SSSOM data frame

    Args:
        jsondoc: The JSON object representing the ontology in obographs format
        curie_map: The curie map to be used
        meta: Any additional metadata that needs to be added to the resulting SSSOM data frame

    Returns:
        An SSSOM data frame (MappingSetDataFrame)

    """
    if not curie_map:
        raise Exception(f'No valid curie_map provided')

    ms = MappingSet()
    mlist = []
    bad_attrs = {}

    allowed_properties = [
        "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
        "http://www.w3.org/2004/02/skos/core#exactMatch",
        "http://www.w3.org/2004/02/skos/core#broadMatch",
        "http://www.w3.org/2004/02/skos/core#closeMatch",
        "http://www.w3.org/2004/02/skos/core#narrowMatch",
        "http://www.w3.org/2004/02/skos/core#relatedMatch"
    ]

    if 'graphs' in jsondoc:
        for g in jsondoc['graphs']:
            if 'nodes' in g:
                for n in g['nodes']:
                    nid = n['id']
                    if 'lbl' in n:
                        label = n['lbl']
                    else:
                        label = ""
                    if 'meta' in n:
                        if 'xrefs' in n['meta']:
                            for xref in n['meta']['xrefs']:
                                xref_id = xref['val']
                                mdict = {}
                                try:
                                    mdict['subject_id'] = curie(nid, curie_map)
                                    mdict['object_id'] = curie(xref_id, curie_map)
                                    mdict['subject_label'] = label
                                    mdict['predicate_id'] = "oboInOwl:hasDbXref"
                                    mdict['match_type'] = "Unspecified"
                                    mlist.append(Mapping(**mdict))
                                except NoCURIEException as e:
                                    logging.warning(e)
                        if 'basicPropertyValues' in n['meta']:
                            for basicPropertyBalue in n['meta']['basicPropertyValues']:
                                pred = basicPropertyBalue['pred']
                                if pred in allowed_properties:
                                    xref_id = basicPropertyBalue['val']
                                    mdict = {}
                                    try:
                                        mdict['subject_id'] = curie(nid, curie_map)
                                        mdict['object_id'] = curie(xref_id, curie_map)
                                        mdict['subject_label'] = label
                                        mdict['predicate_id'] = curie(pred, curie_map)
                                        mdict['match_type'] = "Unspecified"
                                        mlist.append(Mapping(**mdict))
                                    except NoCURIEException as e:
                                        logging.warning(e)
    else:
        raise Exception(f'No graphs element in obographs file, wrong format?')

    ms.mappings = mlist
    if meta:
        for k, v in meta.items():
            if k != 'curie_map':
                ms[k] = v
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


def from_owl_graph(g: Graph, curie_map: Dict[str, str], meta: Dict[str, str]) -> MappingSetDataFrame:
    """
    Converts a dataframe to a MappingSetDataFrame
    :param g: A Graph object (rdflib)
    :param curie_map:
    :param meta: an optional set of metadata elements
    :return: MappingSetDataFrame
    """
    if not curie_map:
        raise Exception(f'No valid curie_map provided')

    ms = MappingSet()
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


def from_rdf_graph(g: Graph, curie_map: Dict[str, str], meta: Dict[str, str],
                   mapping_predicates: Set[str] = None) -> MappingSetDataFrame:
    """
    Converts a dataframe to a MappingSetDataFrame
    :param g: A Graph object (rdflib)
    :param curie_map:
    :param meta: an optional set of metadata elements
    :return: MappingSetDataFrame
    """
    if not curie_map:
        raise Exception(f'No valid curie_map provided')
    if mapping_predicates is None:
        mapping_predicates = get_default_mapping_predicates()
    ms = MappingSet()
    for s, p, o in g.triples((None, None, None)):
        if isinstance(s, URIRef):
            try:
                p_id = curie(p, curie_map)
                if p_id in mapping_predicates:
                    s_id = curie(s, curie_map)
                    if isinstance(o, URIRef):
                        o_id = curie(o, curie_map)
                        m = Mapping(subject_id=s_id,
                                    object_id=o_id,
                                    predicate_id=p_id)
                        ms.mappings.append(m)
            except NoCURIEException as e:
                logging.warning(e)
    mdoc = MappingSetDocument(mapping_set=ms, curie_map=curie_map)
    return to_mapping_set_dataframe(mdoc)


# Utilities (reading)
# All from_* should return MappingSetDataFrame

def get_default_mapping_predicates():
    return {
        'oio:hasDbXref',
        'skos:exactMatch',
        'skos:narrowMatch',
        'skos:broadMatch',
        'skos:exactMatch',
        'skos:closeMatch',
        'owl:sameAs',
        'owl:equivalentClass',
        'owl:equivalentProperty'
    }


def get_parsing_function(input_format, filename):
    if input_format is None:
        input_format = get_file_extension(filename)
    if input_format == 'tsv':
        return from_tsv
    elif input_format == 'rdf':
        return from_rdf
    elif input_format == 'owl':
        return from_owl
    elif input_format == 'alignment-api-xml':
        return from_alignment_xml
    elif input_format == 'obographs-json':
        return from_obographs_json
    elif input_format == 'json-ld':
        return from_jsonld
    elif input_format == 'json':
        raise Exception(f'LinkML JSON not yet implemented, did you mean json-ld or obographs-json.')
    else:
        raise Exception(f'Unknown input format: {input_format}')


def _prepare_mapping(mapping: Mapping):
    p = mapping.predicate_id
    if p == "sssom:superClassOf":
        mapping.predicate_id = "rdfs:subClassOf"
        return _swap_object_subject(mapping)
    return mapping


def _swap_object_subject(mapping):
    members = [attr.replace("subject_", "") for attr in dir(mapping) if
               not callable(getattr(mapping, attr)) and not attr.startswith("__") and attr.startswith("subject_")]
    for var in members:
        subject_val = getattr(mapping, "subject_" + var)
        object_val = getattr(mapping, "object_" + var)
        setattr(mapping, "subject_" + var, object_val)
        setattr(mapping, "object_" + var, subject_val)
    return mapping


def _read_metadata_from_table(filename: str):
    with open(filename, 'r') as s:
        yamlstr = ""
        for line in s:
            if line.startswith("#"):
                yamlstr += re.sub('^#', '', line)
            else:
                break
        if yamlstr:
            meta = yaml.safe_load(yamlstr)
            logging.info(f'Meta={meta}')
            return meta
    return {}


def is_valid_mapping(m):
    return True


class NoCURIEException(Exception):
    pass

def curie(uri: str, curie_map):
    if is_curie(uri):
        return uri
    for prefix in curie_map:
        uri_prefix = curie_map[prefix]
        if uri.startswith(uri_prefix):
            remainder = uri.replace(uri_prefix, "")
            return f"{prefix}:{remainder}"
    raise NoCURIEException(f"{uri} does not follow any known prefixes")


def _cell_element_values(cell_node, curie_map: dict) -> Mapping:
    mdict = {}
    for child in cell_node.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            try:
                if child.nodeName == "entity1":
                    mdict["subject_id"] = curie(child.getAttribute('rdf:resource'), curie_map)
                elif child.nodeName == "entity2":
                    mdict["object_id"] = curie(child.getAttribute('rdf:resource'), curie_map)
                elif child.nodeName == "measure":
                    mdict["confidence"] = child.firstChild.nodeValue
                elif child.nodeName == "relation":
                    relation = child.firstChild.nodeValue
                    if relation == "=":
                        mdict["predicate_id"] = "owl:equivalentClass"
                    else:
                        logging.warning(f"{relation} not a recognised relation type.")
                else:
                    logging.warning(f"Unsupported alignment api element: {child.nodeName}")
            except NoCURIEException as e:
                logging.warning(e)

    m = Mapping(**mdict)
    if is_valid_mapping(m):
        return m


def to_mapping_set_document(msdf: MappingSetDataFrame) -> MappingSetDocument:
    """
    Converts a MappingSetDataFrame to a MappingSetDocument
    :param msdf:
    :return: MappingSetDocument
    """
    if not msdf.prefixmap:
        raise Exception(f'No valid curie_map provided')

    mlist = []
    ms = MappingSet()
    bad_attrs = {}
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
        logging.warning(f'No attr for {k} [{v} instances]')
    ms.mappings = mlist
    for k, v in msdf.metadata.items():
        if k != 'curie_map':
            ms[k] = v
    return MappingSetDocument(mapping_set=ms, curie_map=msdf.prefixmap)


def split_dataframe(msdf: MappingSetDataFrame):
    df = msdf.df
    subject_prefixes = set(df['subject_id'].str.split(':', 1, expand=True)[0])
    object_prefixes = set(df['object_id'].str.split(':', 1, expand=True)[0])
    relations = set(df['predicate_id'])
    return split_dataframe_by_prefix(msdf=msdf, subject_prefixes=subject_prefixes, object_prefixes=object_prefixes,
                                     relations=relations)


def split_dataframe_by_prefix(msdf: MappingSetDataFrame, subject_prefixes, object_prefixes, relations):
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

                dfs = df[(df['subject_id'].str.startswith(pre_subj + ":"))
                         & (df['predicate_id'] == rel)
                         & (df['object_id'].str.startswith(pre_obj + ":"))]
                if pre_subj in curie_map and pre_obj in curie_map and len(dfs) > 0:
                    cm = {pre_subj: curie_map[pre_subj], pre_obj: curie_map[pre_obj], relpre: curie_map[relpre]}
                    msdf = from_dataframe(dfs, curie_map=cm, meta=meta)
                    splitted[split_name] = msdf
                else:
                    print(
                        f"Not adding {split_name} because there is a missing prefix ({pre_subj}, {pre_obj}), "
                        f"or no matches ({len(dfs)} matches found)")
    return splitted
