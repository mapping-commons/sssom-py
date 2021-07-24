import json
import logging
import os
import sys

import pandas as pd
import yaml
from jsonasobj import as_json_obj
from linkml_runtime.utils.yamlutils import as_json_object as yaml_to_json
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from .parsers import to_mapping_set_document
from .sssom_datamodel import slots
from .util import MappingSetDataFrame, extract_global_metadata
from .util import RDF_FORMATS
from .util import get_file_extension

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
        msdf: MappingSetDataFrame, filename: str, fileformat="tsv"
) -> None:  # , context_path=None) -> None:
    """
    dataframe 2 tsv
    """

    if fileformat == "csv":
        sep = ","
    elif fileformat == "tsv":
        sep = "\t"
    else:
        raise Exception(
            f"Unknown table format: {fileformat}, should be one of tsv or csv"
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


def write_rdf(
        msdf: MappingSetDataFrame, filename: str, fileformat="xml"
) -> None:
    """
    dataframe 2 tsv
    """
    graph = to_rdf_graph(msdf)
    graph.serialize(destination=filename, format=fileformat)


def write_owl(
        msdf: MappingSetDataFrame, filename: str, fileformat="xml") -> None:
    """
    dataframe 2 tsv
    """
    graph = to_owl_graph(msdf)
    graph.serialize(destination=filename, format=fileformat)


def write_json(
        msdf: MappingSetDataFrame, filename: str, fileformat="jsonld") -> None:
    """
    dataframe 2 tsv
    """
    _prepare_context_from_curie_map()
    if fileformat == "jsonld":
        data = json.dumps(to_jsonld(msdf), indent=4)
        with open(filename, "w") as outfile:
            print(data, file=outfile)
    elif fileformat == "json":
        doc = to_mapping_set_document(msdf)
        context = _prepare_context_from_curie_map(doc.curie_map)
        data = JSONDumper().dumps(doc.mapping_set, contexts=context)
        with open(filename, "w") as outfile:
            print(data, file=outfile)

    else:
        raise Exception(
            f"Unknown json format: {fileformat}, currently only jsonld supported"
        )
    with open(filename, "w") as outfile:
        print(f"Purchase Amount: {data}", file=outfile)


# Converters


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

    doc = to_mapping_set_document(msdf)

    graph = Graph()
    for k, v in doc.curie_map.items():
        graph.namespace_manager.bind(k, URIRef(v))

    cntxt = _prepare_context_from_curie_map(doc.curie_map)
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
    for m in jsonobj["mappings"]:
        m["@type"] = "owl:Axiom"
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
        logging.info(f"Axiom: {axiom}")
        for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
            for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                for o in graph.objects(
                        subject=axiom, predicate=OWL.annotatedTarget
                ):
                    if p.toPython() == OWL_EQUIV_CLASS:
                        graph.add((s, URIRef(RDF_TYPE), URIRef(OWL_CLASS)))
                        graph.add((o, URIRef(RDF_TYPE), URIRef(OWL_CLASS)))
                    elif p.toPython() == OWL_EQUIV_OBJECTPROPERTY:
                        graph.add(
                            (o, URIRef(RDF_TYPE), URIRef(OWL_OBJECT_PROPERTY))
                        )
                        graph.add(
                            (s, URIRef(RDF_TYPE), URIRef(OWL_OBJECT_PROPERTY))
                        )
                    graph.add((s, p, o))
                    if p.toPython().startswith(SSSOM_NS):
                        # prefix commons has that working
                        graph.add(
                            (p, URIRef(RDF_TYPE), URIRef(OWL_ANNOTATION_PROPERTY))
                        )

    # for m in doc.mapping_set.mappings:
    #    graph.add( (URIRef(m.subject_id), URIRef(m.predicate_id), URIRef(m.object_id)))
    return graph


def _prepare_context_from_curie_map(curie_map: dict):
    return {"@context": curie_map}


def to_rdf_graph(msdf: MappingSetDataFrame) -> Graph:
    """

    Args:
        msdf:

    Returns:

    """
    doc = to_mapping_set_document(msdf)
    cntxt = _prepare_context_from_curie_map(doc.curie_map)
    return RDFDumper().as_rdf_graph(doc.mapping_set, contexts=cntxt)


def to_jsonld(msdf: MappingSetDataFrame) -> dict:
    """

    Args:
        msdf: A SSSOM Data Table

    Returns:
        A JSON Object (dictionary) with the JSONLD representation of the SSSOM Table

    """

    g = to_rdf_graph(msdf)
    json_obj = json.loads(g.serialize(format="jsonld").decode())
    return json_obj


def to_json(msdf: MappingSetDataFrame) -> dict:
    """

    Args:
        msdf: A SSSOM Data Table

    Returns:
        The standard SSSOM json representation
    """
    doc = to_mapping_set_document(msdf)
    context = _prepare_context_from_curie_map(doc.curie_map)
    return as_json_object(doc.mapping_set, context)


def get_writer_function(output_format, output):
    if output_format is None:
        output_format = get_file_extension(output)

    if output_format == "tsv":
        return write_tsv, output_format
    elif output_format in RDF_FORMATS:
        return write_rdf, output_format
    elif output_format == "rdf":
        return write_rdf, "xml"
    elif output_format == "json":
        return write_json, "jsonld"
    elif output_format == "owl":
        return write_owl, "xml"
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
