import logging
import os
import re
from typing import Dict, Set
from xml.dom import minidom, Node
from xml.dom.minidom import Document
import json

import pandas as pd
import yaml
from rdflib import Graph, BNode, Literal, URIRef

from .sssom_document import MappingSet, Mapping, MappingSetDocument

from .util import RDF_FORMATS
from .datamodel_util import MappingSetDataFrame, get_file_extension, to_mapping_set_dataframe

from sssom.datamodel_util import read_pandas

cwd = os.path.abspath(os.path.dirname(__file__))



# Readers (from file)


def from_tsv(filename: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """

    df = read_pandas(filename)
    if meta is None:
        meta = _read_metadata_from_table(filename)
    if 'curie_map' in meta:
        logging.info("Context provided, but SSSOM file provides its own CURIE map. "
                     "CURIE map from context is disregarded.")
        curie_map = meta['curie_map']
    msdf = from_dataframe(df, curie_map=curie_map, meta=meta)
    #msdf = to_mapping_set_dataframe(doc) # Creates a MappingSetDataFrame object
    return msdf
    


def from_rdf(filename: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    g = Graph()
    file_format = guess_file_format(filename)
    g.parse(filename, format=file_format)
    msdf = from_rdf_graph(g, curie_map, meta)
    #msdf = to_mapping_set_dataframe(doc) # Creates a MappingSetDataFrame object
    return msdf


def from_owl(filename: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    g = Graph()
    file_format = guess_file_format(filename)
    g.parse(filename, format=file_format)
    msdf = from_owl_graph(g, curie_map, meta)
    #msdf = to_mapping_set_dataframe(doc) # Creates a MappingSetDataFrame object
    return msdf

def from_obographs_json(filename: str, curie_map: Dict[str, str] = None, meta: Dict[str, str] = None, properties: list = []) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """


    with open(filename) as json_file:
        jsondoc = json.load(json_file)

    msdf = from_obographs(jsondoc, curie_map, meta)
    #msdf = to_mapping_set_dataframe(doc) # Creates a MappingSetDataFrame object
    return msdf


def guess_file_format(filename):
    extension = get_file_extension(filename)
    if extension in ["owl", "rdf"]:
        return "xml"
    elif extension in RDF_FORMATS:
        return extension
    else:
        raise Exception(f"File extension {extension} does not correspond to a legal file format")


def from_alignment_xml(filename: str, curie_map: Dict[str, str] = None,
                       meta: Dict[str, str] = None) -> MappingSetDataFrame:
    """
    parses a TSV -> MappingSetDocument -> MappingSetDataFrame
    """
    logging.info("Loading from alignment API")
    xmldoc = minidom.parse(filename)
    msdf = from_alignment_minidom(xmldoc, curie_map, meta)
    return msdf


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
    mdoc =  MappingSetDocument(mapping_set=ms, curie_map=curie_map)
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
    Converts a dataframe to a MappingSetDataFrame
    :param g: A Graph object (rdflib)
    :param curie_map:
    :param meta: an optional set of metadata elements
    :return: MappingSetDataFrame
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
                    if 'meta' in n:
                        if 'xrefs' in n['meta']:
                            for xref in n['meta']['xrefs']:
                                xref_id = xref['val']
                                mdict = {}
                                mdict['subject_id'] = curie(nid, curie_map)
                                mdict['object_id'] = curie(xref_id, curie_map)
                                mdict['predicate_id'] = "oboInOwl:hasDbXref"
                                mlist.append(Mapping(**mdict))
                        if 'basicPropertyValues' in n['meta']:
                            for basicPropertyBalue in n['meta']['basicPropertyValues']:
                                pred = basicPropertyBalue['pred']
                                if pred in allowed_properties:
                                    xref_id = basicPropertyBalue['val']
                                    mdict = {}
                                    mdict['subject_id'] = curie(nid, curie_map)
                                    mdict['object_id'] = curie(xref_id, curie_map)
                                    mdict['predicate_id'] = curie(pred, curie_map)
                                    mlist.append(Mapping(**mdict))
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


def from_rdf_graph(g: Graph, curie_map: Dict[str, str], meta: Dict[str, str], mapping_predicates: Set[str] = None) -> MappingSetDataFrame:
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
    for s,p,o in g.triples((None, None, None)):
        if isinstance(s, URIRef):
            p_id = curie(p, curie_map)
            if p_id in mapping_predicates:
                s_id = curie(s, curie_map)
                if isinstance(o, URIRef):
                    o_id = curie(o, curie_map)
                    m = Mapping(subject_id=s_id,
                                object_id=o_id,
                                predicate_id=p_id)
                    ms.mappings.append(m)
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
    elif input_format == 'json':
        return from_obographs_json
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


def curie(uri: str, curie_map):
    for prefix in curie_map:
        uri_prefix = curie_map[prefix]
        if uri.startswith(uri_prefix):
            remainder = uri.replace(uri_prefix, "")
            return f"{prefix}:{remainder}"
    return uri


def _cell_element_values(cell_node, curie_map: dict) -> Mapping:
    mdict = {}
    for child in cell_node.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
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
    m = Mapping(**mdict)
    if is_valid_mapping(m):
        return m

def to_mapping_set_document(msdf:MappingSetDataFrame) -> MappingSetDocument:
    """
    Converts a MappingSetDataFrame to a MappingSetDocument
    :param msdf:
    :return: MappingSetDocument
    """
    if not msdf.metadata['curie_map']:
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
    return MappingSetDocument(mapping_set=ms, curie_map=msdf.metadata['curie_map'])