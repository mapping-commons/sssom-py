"""Rewriting functionality for RDFlib graphs."""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Sequence, cast

import curies
from linkml_runtime.utils.metamodelcore import URIorCURIE
from rdflib import Graph, Node, URIRef
from sssom_schema import EntityReference, Mapping

from .parsers import to_mapping_set_document
from .util import MappingSetDataFrame

__all__ = [
    "rewire_graph",
]


def _precedence_index(curie: str, converter: curies.Converter, precedence: Sequence[str]) -> int:
    """Find how strongly ``precedence`` prefers a CURIE, by the position of its prefix.

    :param curie: A candidate replacement.
    :param converter: The converter that parses the CURIE.
    :param precedence: Prefixes in order of preference.

    :return: The position of the prefix in ``precedence``, or the length of ``precedence`` when
        the prefix is not listed, so that a smaller number is a stronger preference.
    """
    prefix = converter.parse_curie(curie, strict=True).prefix
    return precedence.index(prefix) if prefix in precedence else len(precedence)


def rewire_graph(
    g: Graph,
    mset: MappingSetDataFrame,
    subject_to_object: bool = True,
    precedence: Optional[List[str]] = None,
) -> int:
    """Rewire an RDF Graph replacing using equivalence mappings."""
    mdoc = to_mapping_set_document(mset)
    if mdoc.mapping_set.mappings is None:
        raise TypeError

    converter: curies.Converter = mdoc.converter
    # Map each node to its candidate replacements, in the order of their first mapping.
    candidates: defaultdict[URIorCURIE, List[URIorCURIE]] = defaultdict(set)
    for m in mdoc.mapping_set.mappings:
        if not isinstance(m, Mapping):
            continue
        if m.predicate_id in {"owl:equivalentClass", "owl:equivalentProperty"}:
            if subject_to_object:
                src, tgt = m.subject_id, m.object_id
            else:
                src, tgt = m.object_id, m.subject_id
            if not isinstance(src, EntityReference) or not isinstance(tgt, EntityReference):
                raise TypeError
            candidates[src].add(tgt)

    key = partial(_precedence_index, converter, precedence, tgt)

    # Choose only after reading every mapping, so that only distinct candidates tie.
    rewire_map: Dict[URIorCURIE, URIorCURIE] = {}
    for src, targets in candidates.items():
        if len(targets) == 1:
            rewire_map[src] = next(iter(targets))
            continue
        message = f"Ambiguous: {src} -> {' vs '.join(sorted(targets))}"
        if not precedence:
            raise ValueError(message)
        logging.info(message)
        # When two keys are equal, min keeps the first, so the earlier mapping wins.
        rewire_map[src] = min(targets, key=key)
        logging.info(f"{rewire_map[src]} has precedence, due to {precedence}")

    uri_ref_rewire_map: Dict[URIRef, URIRef] = {
        URIRef(converter.expand_strict(k)): URIRef(converter.expand_strict(v))
        for k, v in rewire_map.items()
    }

    def rewire_node(n: Node) -> Node:
        """Rewire node."""
        if not isinstance(n, URIRef):
            return n
        elif n not in uri_ref_rewire_map:
            return n
        else:
            return uri_ref_rewire_map[n]

    triples: list[tuple[Node, Node, Node]] = []
    new_triples: list[tuple[Node, Node, Node]] = []
    num_changed = 0
    for raw_triple in g.triples((None, None, None)):
        triples.append(raw_triple)
        rewired_triple = cast(tuple[Node, Node, Node], tuple(rewire_node(x) for x in raw_triple))
        new_triples.append(rewired_triple)
        if rewired_triple != raw_triple:
            num_changed += 1
    for raw_triple in triples:
        g.remove(raw_triple)
    for raw_triple in new_triples:
        g.add(raw_triple)
    return num_changed
