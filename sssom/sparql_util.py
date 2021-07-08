from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, OWL, SKOS
from typing import Dict, Set, List, Optional
from dataclasses import dataclass
from sssom.util import MappingSetDataFrame
import pandas as pd
import logging


@dataclass
class EndpointConfig:
    url: str = None
    graph: URIRef = None
    predmap: Dict[str, str] = None
    predicates: Optional[List[str]] = None
    limit: Optional[int] = None
    curie_map: Optional[Dict[str, str]] = None
    include_object_labels: bool = False


def query_mappings(config: EndpointConfig) -> MappingSetDataFrame:
    """
    Query a SPARQL endpoint to obtain a set of mapping
    """
    sparql = SPARQLWrapper(config.url)
    if config.graph is None:
        g = "?g"
    else:
        g = config.graph
        if isinstance(g, str):
            g = URIRef(g)
        g = g.n3()
    preds = config.predicates
    if preds is None:
        preds = {SKOS.exactMatch, SKOS.closeMatch}
    else:
        preds = [expand_curie(p, config) for p in preds]
    predstr = " ".join([p.n3() for p in preds])
    limitstr = ""
    if config.limit is not None:
        limitstr = f"LIMIT {config.limit}"
    cols = [
        "subject_id",
        "subject_label",
        "predicate_id",
        "object_id",
        "mapping_provider",
    ]
    if config.include_object_labels:
        cols.insert(-1, "object_label")
    colstr = " ".join([f"?{c}" for c in cols])
    olq = (
        "OPTIONAL { ?object_id rdfs:label ?object_label }"
        if config.include_object_labels
        else ""
    )
    q = f"""
    PREFIX rdfs: {RDFS.uri.n3()} 
    SELECT {colstr}
    WHERE {{
        GRAPH {g} {{ 
          VALUES ?predicate_id {{ {predstr} }} .
          ?subject_id ?predicate_id ?object_id .
          ?subject_id rdfs:label ?subject_label 
        }} .
        {olq}
        BIND({g} as ?mapping_provider)
    }} {limitstr}
    """
    logging.info(q)
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    rows = []
    for result in results["results"]["bindings"]:
        row = {k: v["value"] for k, v in result.items()}
        rows.append(curiefy_row(row, config))
    df = pd.DataFrame(rows)
    return MappingSetDataFrame(df=df, prefixmap=config.curie_map)


def curiefy_row(row: Dict[str, str], config: EndpointConfig) -> Dict[str, str]:
    new_row = {}
    for k, v in row.items():
        new_row[k] = contract_uri(v, config)
    return new_row


def contract_uri(uristr: str, config: EndpointConfig) -> str:
    if config.curie_map is None:
        return uristr
    for k, v in config.curie_map.items():
        if uristr.startswith(v):
            return uristr.replace(v, f"{k}:")
    return uristr


def expand_curie(curie: str, config: EndpointConfig) -> URIRef:
    if config.curie_map is None:
        return URIRef(curie)
    for k, v in config.curie_map.items():
        prefix = f"{k}:"
        if curie.startswith(prefix):
            return URIRef(curie.replace(prefix, v))
    return URIRef(uristr)
