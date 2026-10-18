"""Rewriting functionality for RDFlib graphs."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, cast

import curies
from linkml_runtime.utils.metamodelcore import URIorCURIE
from rdflib import Graph, Node, URIRef
from sssom_schema import EntityReference, Mapping

from .constants import OWL_EQUIVALENT_CLASS, OWL_EQUIVALENT_PROPERTY, PREDICATE_MODIFIER_NOT
from .parsers import to_mapping_set_document
from .util import MappingSetDataFrame

__all__ = [
    "rewire_graph",
]

# Rewiring uses only the mappings with these predicates.
_REWIRE_PREDICATE_CURIES = {
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
}


def rewire_graph(
    g: Graph,
    mset: MappingSetDataFrame,
    *,
    subject_to_object: bool = True,
    precedence: Optional[List[str]] = None,
    strict: bool = False,
) -> int:
    """Rewire an RDF graph in place, replacing nodes by their equivalents in a mapping set.

    :param g: The graph to rewire.
    :param mset: The mapping set. Rewiring uses only its ``owl:equivalentClass`` and
        ``owl:equivalentProperty`` mappings.
    :param subject_to_object: If true, replace the subject of each mapping by its object. If
        false, replace the object by the subject.
    :param precedence: Prefixes in order of preference, which choose between the candidates for
        a node.
    :param strict: If true, raise an error when no mapping is left to rewire by, instead of
        logging a warning.

    :return: The number of triples that changed.
    :raises ValueError: If a node has more than one candidate and no ``precedence`` is given, or
        if ``strict`` is true and no mapping is left to rewire by.
    """
    mdoc = to_mapping_set_document(mset)
    if mdoc.mapping_set.mappings is None:
        raise TypeError

    converter: curies.Converter = mdoc.converter
    rewire_map: Dict[URIorCURIE, URIorCURIE] = {}
    for m in mdoc.mapping_set.mappings:
        if not isinstance(m, Mapping):
            continue
        # A negated mapping records that the two entities do not match.
        if m.predicate_modifier == PREDICATE_MODIFIER_NOT:
            continue
        if m.predicate_id in _REWIRE_PREDICATE_CURIES:
            if subject_to_object:
                src, tgt = m.subject_id, m.object_id
            else:
                src, tgt = m.object_id, m.subject_id
            if not isinstance(src, EntityReference) or not isinstance(tgt, EntityReference):
                raise TypeError
            if src in rewire_map:
                curr_tgt = rewire_map[src]
                logging.info(f"Ambiguous: {src} -> {tgt} vs {curr_tgt}")
                if precedence:
                    curr_pfx = converter.parse_curie(curr_tgt, strict=True).prefix
                    tgt_pfx = converter.parse_curie(tgt, strict=True).prefix
                    if tgt_pfx in precedence:
                        if curr_pfx not in precedence or precedence.index(
                            tgt_pfx
                        ) < precedence.index(curr_pfx):
                            rewire_map[src] = tgt
                            logging.info(f"{tgt} has precedence, due to {precedence}")
                else:
                    raise ValueError(f"Ambiguous: {src} -> {tgt} vs {curr_tgt}")
            else:
                rewire_map[src] = tgt

    # Warn, or fail when strict, rather than return the graph unchanged and say nothing.
    if not rewire_map:
        message = (
            f"No mapping to rewire by with the predicates {', '.join(_REWIRE_PREDICATE_CURIES)}"
        )
        if strict:
            raise ValueError(message)
        logging.warning(f"{message}, so the graph is unchanged")

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
