import json
import logging
import os
import sys

import pandas as pd
import yaml
from jsonasobj2 import JsonObj, as_json_obj
from linkml_runtime.dumpers import JSONDumper, RDFDumper
from rdflib.namespace import OWL, RDF
from linkml_runtime.utils.yamlutils import (
    as_json_object,
)
from rdflib import Graph, URIRef

from .context import get_default_metadata, get_jsonld_context
from .parsers import to_mapping_set_document
from .sssom_datamodel import slots
from .util import MappingSetDataFrame, extract_global_metadata
from .util import RDF_FORMATS, SSSOM_DEFAULT_RDF_SERIALISATION
from .util import get_file_extension

# noinspection PyProtectedMember

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_ANNOTATION_PROPERTY = "http://www.w3.org/2002/07/owl#AnnotationProperty"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_EQUIV_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
OWL_EQUIV_OBJECTPROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"
SSSOM_NS = "http://w3id.org/sssom/"

cwd = os.path.abspath(os.path.dirname(__file__))


# Writers


def write_tsv(
        msdf: MappingSetDataFrame, filename: str, serialisation="tsv"
) -> None:  # , context_path=None) -> None:
    """
    dataframe 2 tsv
    """

    if serialisation == "csv":
        sep = ","
    elif serialisation == "tsv":
        sep = "\t"
    else:
        raise Exception(
            f"Unknown table format: {serialisation}, should be one of tsv or csv"
        )

    msdoc = to_mapping_set_document(msdf)
    df = to_dataframe(msdf)
    meta = extract_global_metadata(msdoc)
    if filename and filename != "-":
        if os.path.isfile(filename):
            os.remove(filename)
        f = open(filename, "a")
        if meta:
            mapping_data_string = yaml.dump(meta)
            for line in mapping_data_string.splitlines():
                f.write("#" + line + "\n")
        df.to_csv(f, sep=sep, index=False)
        f.close()
    else:
        # stdout the result for now
        if meta:
            mapping_data_string = yaml.dump(meta)
            for line in mapping_data_string.splitlines():
                sys.stdout.write("#" + line + "\n")
        df.to_csv(sys.stdout, sep=sep, index=False)


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
        doc = to_mapping_set_document(msdf)
        context = _prepare_context_from_curie_map(doc.curie_map)
        data = JSONDumper().dumps(doc.mapping_set, contexts=context)
        with open(filename, "w") as outfile:
            print(data, file=outfile)

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
        msdf:

    Returns:

    """

    graph = to_rdf_graph(msdf=msdf)

    sparql_prefixes = """
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

"""

    sparql_equiv_class = sparql_prefixes + """
INSERT {
  ?c rdf:type owl:Class .
  ?d rdf:type owl:Class .
}
WHERE {
 ?c owl:equivalentClass ?d .
}
"""
    print(sparql_equiv_class)
    graph.update(sparql_equiv_class)

    # jsonobj = yaml_to_json(doc.mapping_set, json.dumps(cntxt))

    # # for m in doc.mapping_set.mappings:
    # #    if m.subject_id not in jsonobj:
    # #        jsonobj[m.subject_id] = {}
    # #    if m.predicate_id not in jsonobj[m.subject_id]:
    # #        jsonobj[m.subject_id][m.predicate_id] = []
    # #    jsonobj[m.subject_id][m.predicate_id].append(m.object_id)
    # #    print(f'T {m.subject_id} = {jsonobj[m.subject_id]}')
    # # TODO: THIS IS BROKEN NOW AND NEEDS PROPER THINKING. THING FROM SCRATCH
    # elements = []
    # for m in jsonobj["mappings"]:
    #     m["@type"] = "owl:Axiom"
    #     for field in m:
    #         if m[field]:
    #             if not field.startswith("@"):
    #                 elements.append(field)
    # jsonld = json.dumps(as_json_obj(jsonobj))
    # logging.warning(f"to_owl_graph:jsonlod={jsonld}")
    # graph.parse(data=jsonld, format="json-ld")
    # elements = list(set(elements))
    # # assert reified triple
    # _inject_annotation_properties(graph, elements)
    #
    # for axiom in graph.subjects(RDF.type, OWL.Axiom):
    #     logging.info(f"Axiom: {axiom}")
    #     for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
    #         for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
    #             for o in graph.objects(subject=axiom, predicate=OWL.annotatedTarget):
    #                 if p.toPython() == OWL_EQUIV_CLASS:
    #                     graph.add((s, URIRef(RDF_TYPE), URIRef(OWL_CLASS)))
    #                     graph.add((o, URIRef(RDF_TYPE), URIRef(OWL_CLASS)))
    #                 elif p.toPython() == OWL_EQUIV_OBJECTPROPERTY:
    #                     graph.add((o, URIRef(RDF_TYPE), URIRef(OWL_OBJECT_PROPERTY)))
    #                     graph.add((s, URIRef(RDF_TYPE), URIRef(OWL_OBJECT_PROPERTY)))
    #                 graph.add((s, p, o))
    #                 if p.toPython().startswith(SSSOM_NS):
    #                     # prefix commons has that working
    #                     graph.add(
    #                         (p, URIRef(RDF_TYPE), URIRef(OWL_ANNOTATION_PROPERTY))
    #                     )

    # for m in doc.mapping_set.mappings:
    #    graph.add( (URIRef(m.subject_id), URIRef(m.predicate_id), URIRef(m.object_id)))
    logging.warning(f"to_owl_graph:g={graph}")
    return graph


def to_rdf_graph(msdf: MappingSetDataFrame) -> Graph:
    """

    Args:
        msdf:

    Returns:

    """
    doc = to_mapping_set_document(msdf)
    cntxt = _prepare_context_from_curie_map(doc.curie_map)

    # TODO following line needs to be replaced by:
    graph = RDFDumper().as_rdf_graph(element=doc.mapping_set, contexts=cntxt)
    #graph = _temporary_as_rdf_graph(element=doc.mapping_set, contexts=cntxt, namespaces=doc.curie_map)
    return graph


def _temporary_as_rdf_graph(element, contexts, namespaces) -> Graph:
    # TODO needs to be replaced by RDFDumper().as_rdf_graph(element=doc.mapping_set, contexts=cntxt)
    graph = Graph()

    for k, v in namespaces.items():
        graph.namespace_manager.bind(k, URIRef(v))

    if not contexts:
        raise Exception(f"ERROR: No context provided to as_rdf_graph().")

    if "@context" not in contexts:
        contexts["@context"] = dict()

    for k, v in namespaces.items():
        contexts["@context"][k] = v

    jsonobj = as_json_object(element, contexts)


    # for m in doc.mapping_set.mappings:
    #    if m.subject_id not in jsonobj:
    #        jsonobj[m.subject_id] = {}
    #    if m.predicate_id not in jsonobj[m.subject_id]:
    #        jsonobj[m.subject_id][m.predicate_id] = []
    #    jsonobj[m.subject_id][m.predicate_id].append(m.object_id)
    #    print(f'T {m.subject_id} = {jsonobj[m.subject_id]}')
    # TODO: should be covered by context?
    # elements = []
    # for m in jsonobj["mappings"]:
    #     m["@type"] = "owl:Axiom"
    #     for field in m:
    #         if m[field]:
    #             if not field.startswith("@"):
    #                 elements.append(field)
    jsonld = json.dumps(as_json_obj(jsonobj))
    graph.parse(data=jsonld, format="json-ld")
    # elements = list(set(elements))
    # assert reified triple
    # _inject_annotation_properties(graph, elements)

    for axiom in graph.subjects(RDF.type, OWL.Axiom):
        logging.info(f"Axiom: {axiom}")
        for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
            for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                for o in graph.objects(
                        subject=axiom, predicate=OWL.annotatedTarget
                ):
                    graph.add((s, p, o))

    # for m in doc.mapping_set.mappings:
    #    graph.add( (URIRef(m.subject_id), URIRef(m.predicate_id), URIRef(m.object_id)))
    return graph

def to_json(msdf: MappingSetDataFrame) -> JsonObj:
    """

    Args:
        msdf: A SSSOM Data Table

    Returns:
        The standard SSSOM json representation
    """
    doc = to_mapping_set_document(msdf)
    context = _prepare_context_from_curie_map(doc.curie_map)
    return as_json_object(doc.mapping_set, context)


# Support methods


def get_writer_function(output_format, output):
    if output_format is None:
        output_format = get_file_extension(output)

    if output_format == "tsv":
        return write_tsv, output_format
    elif output_format in RDF_FORMATS:
        return write_rdf, output_format
    elif output_format == "rdf":
        return write_rdf, SSSOM_DEFAULT_RDF_SERIALISATION
    elif output_format == "json":
        return write_json, "json"
    elif output_format == "owl":
        return write_owl, SSSOM_DEFAULT_RDF_SERIALISATION
    else:
        raise Exception(f"Unknown output format: {output_format}")


def write_tsvs(sssom_dict, output_dir):
    """

    Args:
        sssom_dict:
        output_dir:

    Returns:

    """
    for split_id in sssom_dict:
        sssom_file = os.path.join(output_dir, f"{split_id}.sssom.tsv")
        msdf = sssom_dict[split_id]
        write_tsv(msdf=msdf, filename=sssom_file)
        print(f"Writing {sssom_file} complete!")


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


def _prepare_context_from_curie_map(curie_map: dict):
    meta, default_curie_map = get_default_metadata()
    context = get_jsonld_context()
    if not curie_map:
        curie_map = default_curie_map

    for k, v in curie_map.items():
        if isinstance(v, str):
            if k not in context["@context"]:
                context["@context"][k] = v
            else:
                if context["@context"][k] != v:
                    logging.info(
                        f"{k} namespace is already in the context, ({context['@context'][k]}, "
                        f"but with a different value than {v}. Overwriting!"
                    )
                    context["@context"][k] = v
    return json.dumps(context)
