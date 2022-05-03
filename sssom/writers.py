"""Serialization functions for SSSOM."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, Union

import pandas as pd
import yaml
from jsonasobj2 import JsonObj
from linkml_runtime.dumpers import JSONDumper, rdflib_dumper
from linkml_runtime.utils.schemaview import SchemaView
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from .constants import SCHEMA_YAML
from .parsers import to_mapping_set_document
from .sssom_datamodel import slots
from .util import (
    PREFIX_MAP_KEY,
    RDF_FORMATS,
    SSSOM_DEFAULT_RDF_SERIALISATION,
    SSSOM_URI_PREFIX,
    URI_SSSOM_MAPPINGS,
    MappingSetDataFrame,
    get_file_extension,
    prepare_context_str,
)

# noinspection PyProtectedMember

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_ANNOTATION_PROPERTY = "http://www.w3.org/2002/07/owl#AnnotationProperty"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_EQUIV_OBJECTPROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"
SSSOM_NS = SSSOM_URI_PREFIX

# Writers

MSDFWriter = Callable[[MappingSetDataFrame, TextIO], None]


def write_table(msdf: MappingSetDataFrame, file: TextIO, serialisation="tsv") -> None:
    """Write a mapping set dataframe to the file as a table."""
    if msdf.df is None:
        raise TypeError

    sep = _get_separator(serialisation)

    # df = to_dataframe(msdf)

    meta: Dict[str, Any] = {}
    if msdf.metadata is not None:
        meta.update(msdf.metadata)
    if msdf.prefix_map is not None:
        meta[PREFIX_MAP_KEY] = msdf.prefix_map

    lines = yaml.safe_dump(meta).split("\n")
    lines = [f"# {line}" for line in lines if line != ""]
    s = msdf.df.to_csv(sep=sep, index=False)
    lines = lines + [s]
    for line in lines:
        print(line, file=file)


def write_rdf(
    msdf: MappingSetDataFrame,
    file: TextIO,
    serialisation: Optional[str] = None,
) -> None:
    """Write a mapping set dataframe to the file as RDF."""
    if serialisation is None:
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION
    elif serialisation not in RDF_FORMATS:
        logging.warning(
            f"Serialisation {serialisation} is not supported, "
            f"using {SSSOM_DEFAULT_RDF_SERIALISATION} instead."
        )
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION

    graph = to_rdf_graph(msdf=msdf)
    t = graph.serialize(format=serialisation, encoding="utf-8")
    print(t.decode(), file=file)


def write_json(msdf: MappingSetDataFrame, output: TextIO, serialisation="json") -> None:
    """Write a mapping set dataframe to the file as JSON."""
    if serialisation == "json":
        data = to_json(msdf)
        json.dump(data, output, indent=2)

    else:
        raise ValueError(
            f"Unknown json format: {serialisation}, currently only json supported"
        )


def write_owl(
    msdf: MappingSetDataFrame,
    file: TextIO,
    serialisation=SSSOM_DEFAULT_RDF_SERIALISATION,
) -> None:
    """Write a mapping set dataframe to the file as OWL."""
    if serialisation not in RDF_FORMATS:
        logging.warning(
            f"Serialisation {serialisation} is not supported, "
            f"using {SSSOM_DEFAULT_RDF_SERIALISATION} instead."
        )
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION

    graph = to_owl_graph(msdf)
    t = graph.serialize(format=serialisation, encoding="utf-8")
    print(t.decode(), file=file)


# Converters
# Converters convert a mappingsetdataframe to an object of the supportes types (json, pandas dataframe)


def to_dataframe(msdf: MappingSetDataFrame) -> pd.DataFrame:
    """Convert a mapping set dataframe to a dataframe."""
    data = []
    doc = to_mapping_set_document(msdf)
    if doc.mapping_set.mappings is None:
        raise TypeError
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
    """Convert a mapping set dataframe to OWL in an RDF graph."""
    graph = to_rdf_graph(msdf=msdf)

    for _s, _p, o in graph.triples((None, URIRef(URI_SSSOM_MAPPINGS), None)):
        graph.add((o, URIRef(RDF_TYPE), OWL.Axiom))

    for axiom in graph.subjects(RDF.type, OWL.Axiom):
        for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
            for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                for o in graph.objects(subject=axiom, predicate=OWL.annotatedTarget):
                    graph.add((s, p, o))

    # if MAPPING_SET_ID in msdf.metadata:
    #    mapping_set_id = msdf.metadata[MAPPING_SET_ID]
    # else:
    #    mapping_set_id = DEFAULT_MAPPING_SET_ID

    sparql_prefixes = """
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

"""
    queries = []

    queries.append(
        sparql_prefixes
        + """
    INSERT {
      ?c rdf:type owl:Class .
      ?d rdf:type owl:Class .
    }
    WHERE {
     ?c owl:equivalentClass ?d .
    }
    """
    )

    queries.append(
        sparql_prefixes
        + """
        INSERT {
          ?c rdf:type owl:ObjectProperty .
          ?d rdf:type owl:ObjectProperty .
        }
        WHERE {
         ?c owl:equivalentProperty ?d .
        }
        """
    )

    queries.append(
        sparql_prefixes
        + """
    DELETE {
      ?o rdf:type sssom:MappingSet .
    }
    INSERT {
      ?o rdf:type owl:Ontology .
    }
    WHERE {
     ?o rdf:type sssom:MappingSet .
    }
    """
    )

    queries.append(
        sparql_prefixes
        + """
    DELETE {
      ?o sssom:mappings ?mappings .
    }
    WHERE {
     ?o sssom:mappings ?mappings .
    }
    """
    )

    queries.append(
        sparql_prefixes
        + """
    INSERT {
        ?p rdf:type owl:AnnotationProperty .
    }
    WHERE {
        ?o a owl:Axiom ;
        ?p ?v .
        FILTER(?p!=rdf:type && ?p!=owl:annotatedProperty && ?p!=owl:annotatedTarget && ?p!=owl:annotatedSource)
    }
    """
    )

    for query in queries:
        graph.update(query)

    return graph


def to_rdf_graph(msdf: MappingSetDataFrame) -> Graph:
    """Convert a mapping set dataframe to an RDF graph."""
    doc = to_mapping_set_document(msdf)
    # cntxt = prepare_context(doc.prefix_map)

    # rdflib_dumper.dump(
    #     element=doc.mapping_set,
    #     schemaview=SchemaView(os.path.join(os.getcwd(), "schema/sssom.yaml")),
    #     prefix_map=msdf.prefix_map,
    #     to_file="sssom.ttl",
    # )
    # graph = Graph()
    # graph = graph.parse("sssom.ttl", format="ttl")

    # os.remove("sssom.ttl")  # remove the intermediate file.
    graph = rdflib_dumper.as_rdf_graph(
        element=doc.mapping_set,
        schemaview=SchemaView(SCHEMA_YAML),
        prefix_map=msdf.prefix_map,
    )
    return graph


def to_json(msdf: MappingSetDataFrame) -> JsonObj:
    """Convert a mapping set dataframe to a JSON object."""
    doc = to_mapping_set_document(msdf)
    context = prepare_context_str(doc.prefix_map)
    data = JSONDumper().dumps(doc.mapping_set, contexts=context)
    json_obj = json.loads(data)
    return json_obj


# Support methods


def get_writer_function(
    *, output_format: Optional[str] = None, output: TextIO
) -> Tuple[MSDFWriter, str]:
    """Get appropriate writer function based on file format.

    :param output: Output file
    :param output_format: Output file format, defaults to None
    :raises ValueError: Unknown output format
    :return: Type of writer function
    """
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
        raise ValueError(f"Unknown output format: {output_format}")


def write_tables(
    sssom_dict: Dict[str, MappingSetDataFrame], output_dir: Union[str, Path]
) -> None:
    """Write table from MappingSetDataFrame object.

    :param sssom_dict: Dictionary of MappingSetDataframes
    :param output_dir: The directory in which the derived SSSOM files are written
    """
    # FIXME documentation does not actually describe what this is doing
    # FIXME explanation of sssom_dict does not make sense
    # FIXME sssom_dict is a bad variable name
    output_dir = Path(output_dir).resolve()
    for split_id, msdf in sssom_dict.items():
        path = output_dir.joinpath(f"{split_id}.sssom.tsv")
        with path.open("w") as file:
            write_table(msdf, file)
        logging.info(f"Writing {path} complete!")


def _inject_annotation_properties(graph: Graph, elements) -> None:
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


def _get_separator(serialisation: Optional[str] = None) -> str:
    if serialisation == "csv":
        sep = ","
    elif serialisation == "tsv" or serialisation is None:
        sep = "\t"
    else:
        raise ValueError(
            f"Unknown table format: {serialisation}, should be one of tsv or csv"
        )
    return sep
