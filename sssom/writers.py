import json
import logging
import os
import sys

import pandas as pd
import yaml
from jsonasobj2 import JsonObj
from linkml_runtime.dumpers import JSONDumper, RDFDumper
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from .parsers import to_mapping_set_document
from .sssom_datamodel import slots
from .util import MappingSetDataFrame, prepare_context_from_curie_map, URI_SSSOM_MAPPINGS, DEFAULT_MAPPING_SET_ID, SSSOM_URI_PREFIX
from .util import RDF_FORMATS, SSSOM_DEFAULT_RDF_SERIALISATION, MAPPING_SET_ID
from .util import get_file_extension

# noinspection PyProtectedMember

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_ANNOTATION_PROPERTY = "http://www.w3.org/2002/07/owl#AnnotationProperty"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_EQUIV_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
OWL_EQUIV_OBJECTPROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"
SSSOM_NS = SSSOM_URI_PREFIX

cwd = os.path.abspath(os.path.dirname(__file__))


# Writers


def write_table(
        msdf: MappingSetDataFrame, filename: str, serialisation="tsv"
) -> None:
    """
    dataframe 2 tsv
    """

    sep = _get_separator(serialisation)

    # df = to_dataframe(msdf)

    if msdf.metadata is not None:
        meta = {k: v for k, v in msdf.metadata.items()}
    else:
        meta = {}
    if msdf.prefixmap is not None:
        meta["curie_map"] = msdf.prefixmap

    lines = yaml.safe_dump(meta).split("\n")
    lines = [f"# {line}" for line in lines if line != ""]
    s = msdf.df.to_csv(sep=sep, index=False)
    lines = lines + [s]

    if filename and filename != "-":
        if os.path.isfile(filename):
            os.remove(filename)
        f = open(filename, "a")
        for line in lines:
            f.write(line + "\n")
        f.close()
    else:
        # stdout the result for now
        for line in lines:
            sys.stdout.write("#" + line + "\n")


def write_rdf(msdf: MappingSetDataFrame, filename: str, serialisation=SSSOM_DEFAULT_RDF_SERIALISATION) -> None:
    """
    dataframe 2 tsv
    """

    if serialisation not in RDF_FORMATS:
        logging.warning(f"Serialisation {serialisation} is not supported, "
                        f"using {SSSOM_DEFAULT_RDF_SERIALISATION} instead.")
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION

    graph = to_rdf_graph(msdf=msdf)
    graph.serialize(filename, format=serialisation)


def write_json(msdf: MappingSetDataFrame, filename: str, serialisation="json") -> None:
    """
    dataframe 2 tsv
    """
    if serialisation == "json":
        data = to_json(msdf)
        # doc = to_mapping_set_document(msdf)
        # context = prepare_context_from_curie_map(doc.curie_map)
        # data = JSONDumper().dumps(doc.mapping_set, contexts=context)
        with open(filename, "w") as outfile:
            json.dump(data, outfile, indent='  ')

    else:
        raise Exception(
            f"Unknown json format: {serialisation}, currently only json supported"
        )


def write_owl(msdf: MappingSetDataFrame, filename: str, serialisation=SSSOM_DEFAULT_RDF_SERIALISATION) -> None:
    if serialisation not in RDF_FORMATS:
        logging.warning(f"Serialisation {serialisation} is not supported, "
                        f"using {SSSOM_DEFAULT_RDF_SERIALISATION} instead.")
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION

    graph = to_owl_graph(msdf)
    graph.serialize(destination=filename, format=serialisation)


# Converters
# Converters convert a mappingsetdataframe to an object of the supportes types (json, pandas dataframe)


def to_dataframe(msdf: MappingSetDataFrame) -> pd.DataFrame:
    data = []

    doc = to_mapping_set_document(msdf)

    for mapping in doc.mapping_set.mappings:
        mdict = mapping.__dict__
        m = {}
        for key in mdict:
            if mdict[key]:
                m[key] = mdict[key]
        data.append(m)
    df = pd.DataFrame(data=data)
    return df


def to_owl_graph(msdf: MappingSetDataFrame) -> Graph:
    """

    Args:
        msdf: The MappingSetDataFrame (SSSOM table)

    Returns:
        an rdfib Graph obect

    """

    graph = to_rdf_graph(msdf=msdf)

    if MAPPING_SET_ID in msdf.metadata:
        mapping_set_id = msdf.metadata[MAPPING_SET_ID]
    else:
        mapping_set_id = DEFAULT_MAPPING_SET_ID

    sparql_prefixes = """
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

"""
    queries = []

    queries.append(sparql_prefixes + """
    INSERT {
      ?c rdf:type owl:Class .
      ?d rdf:type owl:Class .
    }
    WHERE {
     ?c owl:equivalentClass ?d .
    }
    """)

    queries.append(sparql_prefixes + """
    DELETE {
      ?o rdf:type sssom:MappingSet .  
    }
    INSERT {
      ?o rdf:type owl:Ontology .
    }
    WHERE {
     ?o rdf:type sssom:MappingSet .
    }
    """)

    queries.append(sparql_prefixes + """
    DELETE {
      ?o sssom:mappings ?mappings .  
    }
    WHERE {
     ?o sssom:mappings ?mappings .
    }
    """)

    queries.append(sparql_prefixes + """
    INSERT {
        ?p rdf:type owl:AnnotationProperty .  
    }
    WHERE {
        ?o a owl:Axiom ;
        ?p ?v .
        FILTER(?p!=rdf:type)
    }
    """)

    for query in queries:
        graph.update(query)

    return graph


def to_rdf_graph(msdf: MappingSetDataFrame) -> Graph:
    """

    Args:
        msdf:

    Returns:

    """
    doc = to_mapping_set_document(msdf)
    cntxt = prepare_context_from_curie_map(doc.curie_map)
    # json_obj = to_json(msdf)
    # g = Graph()
    # g.load(json_obj, format="json-ld")
    # print(g.serialize(format="xml"))

    graph = _temporary_as_rdf_graph(element=doc.mapping_set, contexts=cntxt, namespaces=doc.curie_map)
    return graph


def _temporary_as_rdf_graph(element, contexts, namespaces=None) -> Graph:
    # TODO needs to be replaced by RDFDumper().as_rdf_graph(element=doc.mapping_set, contexts=cntxt)
    graph = RDFDumper().as_rdf_graph(element=element, contexts=contexts)
    # print(graph.serialize(fmt="turtle").decode())

    # Adding some stuff that the default RDF serialisation does not do:
    # Direct triples

    for k, v in namespaces.items():
        graph.bind(k, v)

    for s, p, o in graph.triples((None, URIRef(URI_SSSOM_MAPPINGS), None)):
        graph.add((o, URIRef(RDF_TYPE), OWL.Axiom))

    for axiom in graph.subjects(RDF.type, OWL.Axiom):
        for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
            for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                for o in graph.objects(
                        subject=axiom, predicate=OWL.annotatedTarget
                ):
                    graph.add((s, p, o))
    return graph


def to_json(msdf: MappingSetDataFrame) -> JsonObj:
    """

    Args:
        msdf: A SSSOM Data Table

    Returns:
        The standard SSSOM json representation
    """

    doc = to_mapping_set_document(msdf)
    context = prepare_context_from_curie_map(doc.curie_map)
    data = JSONDumper().dumps(doc.mapping_set, contexts=context)
    json_obj = json.loads(data)
    return json_obj


# Support methods


def get_writer_function(output_format, output):
    if output_format is None:
        output_format = get_file_extension(output)

    if output_format == "tsv":
        return write_table, output_format
    elif output_format in RDF_FORMATS:
        return write_rdf, output_format
    elif output_format == "rdf":
        return write_rdf, SSSOM_DEFAULT_RDF_SERIALISATION
    elif output_format == "json":
        return write_json, output_format
    elif output_format == "owl":
        return write_owl, SSSOM_DEFAULT_RDF_SERIALISATION
    else:
        raise Exception(f"Unknown output format: {output_format}")


def write_tables(sssom_dict, output_dir):
    """

    Args:
        sssom_dict:
        output_dir:

    Returns:

    """
    for split_id in sssom_dict:
        sssom_file = os.path.join(output_dir, f"{split_id}.sssom.tsv")
        msdf = sssom_dict[split_id]
        write_table(msdf=msdf, filename=sssom_file)
        logging.info(f"Writing {sssom_file} complete!")


def _inject_annotation_properties(graph: Graph, elements):
    for var in [
        slot
        for slot in dir(slots)
        if not callable(getattr(slots, slot)) and not slot.startswith("__")
    ]:
        slot = getattr(slots, var)
        if slot.name in elements:
            if slot.uri.startswith(SSSOM_NS):
                graph.add(
                    (
                        URIRef(slot.uri),
                        URIRef(RDF_TYPE),
                        URIRef(OWL_ANNOTATION_PROPERTY),
                    )
                )


def _get_separator(serialisation):
    if serialisation == "csv":
        sep = ","
    elif serialisation == "tsv":
        sep = "\t"
    else:
        raise Exception(
            f"Unknown table format: {serialisation}, should be one of tsv or csv"
        )
    return sep
