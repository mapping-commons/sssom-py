import pandas as pd
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from .sssom_document import MappingSet, Mapping, MappingSetDocument, Entity
from .sssom_datamodel import slots, SSSOM_NS
from .context import get_jsonld_context

from jsonasobj import as_json_obj, as_json
from biolinkml.utils.yamlutils import DupCheckYamlLoader, as_json_object as yaml_to_json
from typing import Dict
import json
import yaml
import logging
import os
import re

cwd = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONTEXT_PATH = f'{cwd}/../schema/sssom.context.jsonld'

RDF_FORMATS=['ttl', 'turtle', 'nt']
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_ANNOTATION_PROPERTY = "http://www.w3.org/2002/07/owl#AnnotationProperty"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_EQUIV_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
OWL_EQUIV_OBJECTPROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"

def to_tsv(df : pd.DataFrame, filename: str) -> None:
    """
    dataframe 2 tsv
    big hacky, does not export anything in the header.
    should take something more general, like dict, mappinhset class object
    Saves a dataframe. TODO: header
    """
    return df.to_csv(filename, sep="\t", index=False)

def convert_file(input: str, output:str = None, input_format:str = None, output_format:str = None, context_path=None):
    """
    converts from one format to another
    :param input:
    :param output:
    :param input_format:
    :param output_format:
    :param context_path:
    :return:
    """
    if input_format is None:
        input_format = guess_format(input)
    if output_format is None:
        output_format = guess_format(output)
    if input_format == 'tsv':
        doc = from_tsv(input)
    else:
        raise Exception(f'Unknown input format: {input_format}')

    if output_format in RDF_FORMATS:
        g = to_rdf(doc, context_path=context_path)
        g.serialize(destination=output, format=output_format)
    else:
        raise Exception(f'Unknown output format: {output_format}')

def guess_format(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) > 0:
        return parts[-1]
    else:
        raise Exception(f'Cannot guess format from {filename}')

def tsv_to_dataframe(filename: str) -> pd.DataFrame:
    """
    wrapper to pd.read_csv that handles comment lines correctly
    :param filename:
    :return:
    """
    # TODO: this is awkward... check if there is a more elegant way to filter
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile("r+") as tmp:
        with open(filename, "r") as f:
            for line in f:
                if not line.startswith('#'):
                    tmp.write(line + "\n")
        tmp.seek(0)
        return pd.read_csv(tmp, sep="\t").fillna("")

def from_tsv(filename: str) -> MappingSetDocument:
    """
    parses a TSV to a MappingSetDocument
    """
    df = tsv_to_dataframe(filename)
    curie_map = {}
    with open(filename, 'r') as s:
        in_curie_map = False
        yamlstr = ""
        for line in s:
            if line.startswith("#"):
                yamlstr += re.sub('^#', '', line)
            else:
                break
        meta = yaml.safe_load(yamlstr)
        logging.info(f'Meta={meta}')
        curie_map = meta['curie_map']
    return from_dataframe(df, curie_map=curie_map, meta=meta)


def swap_object_subject(mapping):
    members = [attr.replace("subject_", "") for attr in dir(mapping) if not callable(getattr(mapping, attr)) and not attr.startswith("__") and attr.startswith("subject_")]
    for var in members:
        subject_val = getattr(mapping,"subject_"+var)
        object_val = getattr(mapping, "object_" + var)
        setattr(mapping,"subject_"+var,object_val)
        setattr(mapping, "object_" + var, subject_val)
    return mapping


def prepare_mapping(mapping: Mapping):
    p = mapping.predicate_id
    if p == "sssom:superClassOf":
        mapping.predicate_id = "rdfs:subClassOf"
        return swap_object_subject(mapping)
    return mapping


def from_dataframe(df: pd.DataFrame, curie_map: Dict[str,str], meta: Dict[str,str]) -> MappingSetDocument:
    """
    Converts a dataframe to a MappingSetDocument
    :param df:
    :param curie_map:
    :return: MappingSetDocument
    """
    mlist = []
    ms = MappingSet()
    bad_attrs = {}
    for _, row in df.iterrows():
        mdict = {}
        for k,v in row.items():
            ok = False
            #if k.endswith('_id'): # TODO: introspect
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
        #logging.info(f'Row={mdict}')
        m = prepare_mapping(Mapping(**mdict))
        mlist.append(m)
    for k,v in bad_attrs.items():
        logging.warning(f'No attr for {k} [{v} instances]')
    ms.mappings = mlist
    for k,v in meta.items():
        if k != 'curie_map':
            ms[k] = v
    return MappingSetDocument(mapping_set=ms, curie_map=curie_map)


def inject_annotation_properties(graph: Graph, elements):
    for var in [slot for slot in dir(slots) if not callable(getattr(slots, slot)) and not slot.startswith("__")]:
        slot = getattr(slots, var)
        if slot.name in elements:
            graph.add((URIRef(slot.uri), URIRef(RDF_TYPE), URIRef(OWL_ANNOTATION_PROPERTY)))

def to_rdf(doc: MappingSetDocument, graph: Graph = Graph(), context_path=None) -> Graph:
    """
    Converts to RDF
    :param df:
    :return:
    """

    for k,v in doc.curie_map.items():
        graph.namespace_manager.bind(k, URIRef(v))



    if context_path is not None:
        with open(context_path, 'r') as f:
            cntxt = json.load(f)
    else:
        cntxt = json.loads(get_jsonld_context())
        # see whether I can do this proper;y
        # can we bundle a json ld context in a pypi disro

    if True:
        for k, v in doc.curie_map.items():
            cntxt['@context'][k] = v
        jsonobj = yaml_to_json(doc.mapping_set, cntxt)

        #for m in doc.mapping_set.mappings:
        #    if m.subject_id not in jsonobj:
        #        jsonobj[m.subject_id] = {}
        #    if m.predicate_id not in jsonobj[m.subject_id]:
        #        jsonobj[m.subject_id][m.predicate_id] = []
        #    jsonobj[m.subject_id][m.predicate_id].append(m.object_id)
        #    print(f'T {m.subject_id} = {jsonobj[m.subject_id]}')
        # TODO: should be covered by context?
        elements = []
        for m in jsonobj['mappings']:
            m['@type'] = 'owl:Axiom'
            for field in m:
                if m[field]:
                    if not field.startswith("@"):
                        elements.append(field)
        jsonld = json.dumps(as_json_obj(jsonobj))
        graph.parse(data=jsonld, format="json-ld")
        elements = list(set(elements))
        # assert reified triple
        inject_annotation_properties(graph, elements)

        for axiom in graph.subjects(RDF.type, OWL.Axiom):
            logging.info(f'Axiom: {axiom}')
            for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
                for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                    for o in graph.objects(subject=axiom, predicate=OWL.annotatedTarget):
                        if p.toPython() == OWL_EQUIV_CLASS:
                            graph.add((s, URIRef(RDF_TYPE), URIRef(OWL_CLASS)))
                            graph.add((o, URIRef(RDF_TYPE), URIRef(OWL_CLASS)))
                        elif p.toPython() == OWL_EQUIV_OBJECTPROPERTY:
                            graph.add((o, URIRef(RDF_TYPE), URIRef(OWL_OBJECT_PROPERTY)))
                            graph.add((s, URIRef(RDF_TYPE), URIRef(OWL_OBJECT_PROPERTY)))
                        graph.add((s, p, o))
                        if p.toPython().startswith("http://example.org/sssom/"):
                            print("YESYES")
                            # prefix commons has that working
                            graph.add((p, URIRef(RDF_TYPE), URIRef(OWL_ANNOTATION_PROPERTY)))

        #for m in doc.mapping_set.mappings:
        #    graph.add( (URIRef(m.subject_id), URIRef(m.predicate_id), URIRef(m.object_id)))
        return graph

