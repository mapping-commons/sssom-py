import json
import logging
import yaml
import os

import pandas as pd
from linkml.utils.yamlutils import as_json_object as yaml_to_json
from jsonasobj import as_json_obj, JsonObj
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF
import json

from .sssom_datamodel import slots
from .sssom_document import MappingSetDocument
from .datamodel_util import get_file_extension
from .util import RDF_FORMATS
from .context import get_jsonld_context

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_ANNOTATION_PROPERTY = "http://www.w3.org/2002/07/owl#AnnotationProperty"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_EQUIV_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
OWL_EQUIV_OBJECTPROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"
SSSOM_NS = "http://w3id.org/sssom/"

cwd = os.path.abspath(os.path.dirname(__file__))


# Writers


def write_tsv(msdoc: MappingSetDocument, filename: str, fileformat="tsv", context_path=None) -> None:
    """
    dataframe 2 tsv
    """

    if fileformat == "csv":
        sep = ","
    elif fileformat == "tsv":
        sep = "\t"
    else:
        raise Exception(f'Unknown table format: {fileformat}, should be one of tsv or csv')

    df = to_dataframe(msdoc, context_path)
    meta = extract_global_metadata(msdoc)
    if os.path.isfile(filename):
        os.remove(filename)
    f = open(filename, 'a')
    if meta:
        mapping_data_string = yaml.dump(meta)
        for line in mapping_data_string.splitlines():
            f.write("#" + line + "\n")
    df.to_csv(f, sep=sep, index=False)
    f.close()


def write_rdf(msdoc: MappingSetDocument, filename: str, fileformat="xml", context_path=None) -> None:
    """
    dataframe 2 tsv
    """
    graph = to_rdf_graph(msdoc, context_path)
    graph.serialize(destination=filename, format=fileformat)


def write_owl(msdoc: MappingSetDocument, filename: str, fileformat="xml", context_path=None) -> None:
    """
    dataframe 2 tsv
    """
    graph = to_owl_graph(msdoc, context_path)
    graph.serialize(destination=filename, format=fileformat)


def write_json(msdoc: MappingSetDocument, filename: str, fileformat="jsonld", context_path=None) -> None:
    """
    dataframe 2 tsv
    """
    if fileformat == "jsonld":
        data = to_jsonld_dict(msdoc, context_path)
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)
    else:
        raise Exception(f"Unknown json format: {fileformat}, currently only jsonld supported")


# Converters

def to_dataframe(doc: MappingSetDocument, context_path=None) -> pd.DataFrame:
    data = []
    for mapping in doc.mapping_set.mappings:
        mdict = mapping.__dict__
        m = {}
        for key in mdict:
            if mdict[key]:
                m[key] = mdict[key]
        data.append(m)
    df = pd.DataFrame(data=data)
    return df


def to_owl_graph(doc: MappingSetDocument, context_path=None) -> Graph:
    """
    Converts to RDF - OWL flavour
    :param doc: A MappingSetDocument object
    :param context_path: An optional context path for the MappingSet
    :return:
    """
    graph = Graph()
    for k, v in doc.curie_map.items():
        graph.namespace_manager.bind(k, URIRef(v))

    if context_path is not None:
        with open(context_path, 'r') as f:
            cntxt = json.load(f)
    else:
        cntxt = get_jsonld_context()
        # see whether I can do this proper;y
        # can we bundle a json ld context in a pypi disro

    if True:
        for k, v in doc.curie_map.items():
            cntxt['@context'][k] = v
        jsonobj = yaml_to_json(doc.mapping_set, json.dumps(cntxt))

        # for m in doc.mapping_set.mappings:
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
        _inject_annotation_properties(graph, elements)

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
                        if p.toPython().startswith(SSSOM_NS):
                            # prefix commons has that working
                            graph.add((p, URIRef(RDF_TYPE), URIRef(OWL_ANNOTATION_PROPERTY)))

        # for m in doc.mapping_set.mappings:
        #    graph.add( (URIRef(m.subject_id), URIRef(m.predicate_id), URIRef(m.object_id)))
        return graph


def to_rdf_graph(doc: MappingSetDocument, context_path=None) -> Graph:
    """
    Converts to RDF
    :param doc: A MappingSetDocument object
    :param context_path: An optional context path for the MappingSet
    :return:
    """
    graph = Graph()
    for k, v in doc.curie_map.items():
        graph.namespace_manager.bind(k, URIRef(v))

    if context_path is not None:
        with open(context_path, 'r') as f:
            cntxt = json.load(f)
    else:
        cntxt = get_jsonld_context()
        # see whether I can do this proper;y
        # can we bundle a json ld context in a pypi disro

    if True:
        for k, v in doc.curie_map.items():
            cntxt['@context'][k] = v
        jsonobj = yaml_to_json(doc.mapping_set, json.dumps(cntxt))

        # for m in doc.mapping_set.mappings:
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
        # elements = list(set(elements))
        # assert reified triple
        # _inject_annotation_properties(graph, elements)

        for axiom in graph.subjects(RDF.type, OWL.Axiom):
            logging.info(f'Axiom: {axiom}')
            for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
                for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                    for o in graph.objects(subject=axiom, predicate=OWL.annotatedTarget):
                        graph.add((s, p, o))

        # for m in doc.mapping_set.mappings:
        #    graph.add( (URIRef(m.subject_id), URIRef(m.predicate_id), URIRef(m.object_id)))
        return graph


def to_jsonld_dict(doc: MappingSetDocument, context_path=None) -> Graph:
    """
    Converts to RDF
    :param doc: A MappingSetDocument object
    :param context_path: An optional context path for the MappingSet
    :return:
    """
    g = to_rdf_graph(doc, context_path)
    s = g.serialize(format='json-ld', indent=4)
    return json.loads(s)


def get_writer_function(output_format, output):
    if output_format is None:
        output_format = get_file_extension(output)

    if output_format == 'tsv':
        return write_tsv, output_format
    elif output_format in RDF_FORMATS:
        return write_rdf, output_format
    elif output_format == 'rdf':
        return write_rdf, 'xml'
    elif output_format == 'json':
        return write_json, 'jsonld'
    elif output_format == 'owl':
        return write_owl, 'xml'
    else:
        raise Exception(f'Unknown output format: {output_format}')


def extract_global_metadata(msdoc: MappingSetDocument):
    meta = {'curie_map': msdoc.curie_map}
    ms_meta = msdoc.mapping_set
    for key in [slot for slot in dir(slots) if not callable(getattr(slots, slot)) and not slot.startswith("__")]:
        slot = getattr(slots, key).name
        if slot not in ["mappings"] and slot in ms_meta:
            if ms_meta[slot]:
                meta[key] = ms_meta[slot]
    return meta


def _inject_annotation_properties(graph: Graph, elements):
    for var in [slot for slot in dir(slots) if not callable(getattr(slots, slot)) and not slot.startswith("__")]:
        slot = getattr(slots, var)
        if slot.name in elements:
            if slot.uri.startswith(SSSOM_NS):
                graph.add((URIRef(slot.uri), URIRef(RDF_TYPE), URIRef(OWL_ANNOTATION_PROPERTY)))
