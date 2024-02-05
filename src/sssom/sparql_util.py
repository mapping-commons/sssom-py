"""Utilities for querying mappings with SPARQL."""

import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Optional

import pandas as pd
from curies import Converter
from rdflib import URIRef
from rdflib.namespace import RDFS, SKOS
from SPARQLWrapper import JSON, SPARQLWrapper

from .util import MappingSetDataFrame, safe_compress

__all__ = [
    "EndpointConfig",
    "query_mappings",
]


@dataclass
class EndpointConfig:
    """A container for a SPARQL endpoint's configuration."""

    url: str
    graph: URIRef
    converter: Converter
    predmap: Dict[str, str]
    predicates: Optional[List[str]]
    limit: Optional[int]
    include_object_labels: bool = False


def query_mappings(config: EndpointConfig) -> MappingSetDataFrame:
    """Query a SPARQL endpoint to obtain a set of mappings."""
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
            URIRef(config.converter.expand_strict(predicate)) for predicate in config.predicates
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
    colstr = " ".join(f"?{c}" for c in cols)
    olq = "OPTIONAL { ?object_id rdfs:label ?object_label }" if config.include_object_labels else ""
    sparql = dedent(
        f"""\
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
    )
    logging.info(sparql)

    sparql_wrapper = SPARQLWrapper(config.url, returnFormat=JSON)
    sparql_wrapper.setQuery(sparql)
    results = sparql_wrapper.query().convert()
    df = pd.DataFrame(
        [
            {key: safe_compress(v["value"], config.converter) for key, v in result.items()}
            for result in results["results"]["bindings"]
        ]
    )
    return MappingSetDataFrame.with_converter(df=df, converter=config.converter)
