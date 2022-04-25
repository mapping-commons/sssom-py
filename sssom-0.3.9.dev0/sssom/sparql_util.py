"""Utilities for querying mappings with SPARQL."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import pandas as pd
from rdflib import URIRef
from rdflib.namespace import RDFS, SKOS
from SPARQLWrapper import JSON, SPARQLWrapper

from .util import MappingSetDataFrame

__all__ = [
    "EndpointConfig",
    "query_mappings",
]


@dataclass
class EndpointConfig:
    """A container for a SPARQL endpoint's configuration."""

    url: str
    graph: URIRef
    predmap: Dict[str, str]
    predicates: Optional[List[str]]
    limit: Optional[int]
    prefix_map: Optional[Dict[str, str]]
    include_object_labels: bool = False


def query_mappings(config: EndpointConfig) -> MappingSetDataFrame:
    """Query a SPARQL endpoint to obtain a set of mappings."""
    sparql = SPARQLWrapper(config.url)
    if config.graph is None:
        g = "?g"
    elif isinstance(config.graph, str):
        g = URIRef(config.graph).n3()
    else:
        g = config.graph.n3()
    if config.predicates is None:
        predicates = [SKOS.exactMatch, SKOS.closeMatch]
    else:
        predicates = [
            expand_curie(predicate, config) for predicate in config.predicates
        ]
    predstr = " ".join(URIRef(predicate).n3() for predicate in predicates)
    if config.limit is not None:
        limitstr = f"LIMIT {config.limit}"
    else:
        limitstr = ""
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
    q = f"""\
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
        rows.append(curiefy_values(row, config))
    df = pd.DataFrame(rows)
    if config.prefix_map is None:
        raise TypeError
    return MappingSetDataFrame(df=df, prefix_map=config.prefix_map)


def curiefy_values(row: Mapping[str, str], config: EndpointConfig) -> Dict[str, str]:
    """Convert all values in the dict from URIs to CURIEs.

    :param row: A dictionary of string keys to URIs
    :param config: Configuration
    :return: A dictionary of string keys to CURIEs
    """
    return {k: contract_uri(v, config) for k, v in row.items()}


def contract_uri(uri: str, config: EndpointConfig) -> str:
    """Replace the URI with a CURIE based on the prefix map in the given configuration.

    :param uri: A uniform resource identifier
    :param config: Configuration
    :return: A CURIE if it's able to contract, otherwise return the original URI
    """
    if config.prefix_map is None:
        return uri
    for k, v in config.prefix_map.items():
        if uri.startswith(v):
            return uri.replace(v, f"{k}:")
    return uri


def expand_curie(curie: str, config: EndpointConfig) -> URIRef:
    """Expand a CURIE to a URI.

    :param curie: CURIE
    :param config: Configuration
    :return: URI of CURIE
    """
    if config.prefix_map is None:
        return URIRef(curie)
    for k, v in config.prefix_map.items():
        prefix = f"{k}:"
        if curie.startswith(prefix):
            return URIRef(curie.replace(prefix, v))
    return URIRef(curie)
