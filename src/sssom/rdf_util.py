"""Rewriting functionality for RDFlib graphs."""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from typing import Collection, Dict, Optional, Sequence, cast

import curies
from linkml_runtime.utils.metamodelcore import URIorCURIE
from rdflib import Graph, Node, URIRef
from sssom_schema import EntityReference, Mapping

from .constants import (
    NO_TERM_FOUND,
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
    PREDICATE_MODIFIER_NOT,
    SKOS_BROAD_MATCH,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_NARROW_MATCH,
    SKOS_RELATED_MATCH,
)
from .parsers import to_mapping_set_document
from .util import MappingSetDataFrame

__all__ = [
    "rewire_graph",
]

# Rewiring uses the mappings with these predicates when no predicates are given.
_REWIRE_PREDICATE_CURIES = {
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
}

# The strength rank of each predicate, where 0 is the strongest. A predicate not listed here
# ranks below all of them.
_PREDICATE_RANK: Dict[str, int] = {
    OWL_EQUIVALENT_CLASS: 0,
    OWL_EQUIVALENT_PROPERTY: 0,
    SKOS_EXACT_MATCH: 1,
    SKOS_CLOSE_MATCH: 2,
    SKOS_BROAD_MATCH: 3,
    SKOS_NARROW_MATCH: 3,
    SKOS_RELATED_MATCH: 4,
}
_UNRANKED = max(_PREDICATE_RANK.values()) + 1


def _precedence_index(curie: str, *, converter: curies.Converter, precedence: Sequence[str]) -> int:
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
    *,
    subject_to_object: bool = True,
    precedence: Optional[Sequence[str]] = None,
    predicates: Optional[Collection[str]] = None,
    strict: bool = False,
) -> int:
    """Rewire an RDF graph in place, replacing each mapped node by its mapping partner.

    When a node has several candidates, the one under the strongest predicate wins. The order,
    strongest first, is the OWL equivalences, ``skos:exactMatch``, ``skos:closeMatch``,
    ``skos:broadMatch`` and ``skos:narrowMatch``, ``skos:relatedMatch``, then any other. Mappings
    that agree on a candidate count once, under their strongest predicate. Only the equivalences
    keep the graph's meaning: a ``skos:broadMatch`` mapping puts the broader entity even in the
    triples that describe the narrower one. Negated mappings, and mappings to
    ``sssom:NoTermFound``, are skipped.

    :param g: The graph to rewire.
    :param mset: The mapping set to rewire by.
    :param subject_to_object: If true, replace the subject of each mapping by its object. If
        false, replace the object by the subject.
    :param precedence: Prefixes in order of preference, which choose between equally strong
        candidates for a node. When it does not decide, the earlier mapping wins.
    :param predicates: The mapping predicates to rewire by, as CURIEs or IRIs. Defaults to
        ``owl:equivalentClass`` and ``owl:equivalentProperty``.
    :param strict: If true, raise an error when no mapping is left to rewire by, instead of
        logging a warning.

    :return: The number of triples that changed.
    :raises TypeError: If the mapping set has no mappings, or a mapping in use has a subject or
        object that is not an entity reference.
    :raises ValueError: If equally strong candidates tie and no ``precedence`` is given, or if
        ``strict`` is true and no mapping is left to rewire by.
    """
    mdoc = to_mapping_set_document(mset)
    if mdoc.mapping_set.mappings is None:
        raise TypeError

    converter: curies.Converter = mdoc.converter
    if predicates is None:
        predicates = _REWIRE_PREDICATE_CURIES
    # Compare predicates as IRIs, so that a CURIE and its IRI count as the same predicate.
    predicate_iris = {converter.expand(predicate, passthrough=True) for predicate in predicates}
    rank_by_iri = {
        converter.expand(predicate, passthrough=True): rank
        for predicate, rank in _PREDICATE_RANK.items()
    }

    # Map each node to its candidate replacements, in the order of their first mapping, each with
    # the rank of its strongest mapping.
    candidates: defaultdict[URIorCURIE, Dict[URIorCURIE, int]] = defaultdict(dict)
    for m in mdoc.mapping_set.mappings:
        if not isinstance(m, Mapping):
            continue
        # A negated mapping records that the two entities do not match.
        if m.predicate_modifier == PREDICATE_MODIFIER_NOT:
            continue
        # A mapping to sssom:NoTermFound records that an entity has no match.
        if NO_TERM_FOUND in (m.subject_id, m.object_id):
            continue
        predicate_iri = converter.expand(m.predicate_id, passthrough=True)
        if predicate_iri not in predicate_iris:
            continue
        if subject_to_object:
            src, tgt = m.subject_id, m.object_id
        else:
            src, tgt = m.object_id, m.subject_id
        if not isinstance(src, EntityReference) or not isinstance(tgt, EntityReference):
            raise TypeError
        rank = rank_by_iri.get(predicate_iri, _UNRANKED)
        # A candidate keeps the rank of its strongest mapping, so agreeing mappings count once.
        candidates[src][tgt] = min(rank, candidates[src].get(tgt, rank))

    # Warn, or fail when strict, rather than return the graph unchanged and say nothing.
    if not candidates:
        message = f"No mapping to rewire by with the predicates {', '.join(sorted(predicates))}"
        if strict:
            raise ValueError(message)
        logging.warning(f"{message}, so the graph is unchanged")

    if precedence is not None:
        key = partial(_precedence_index, converter=converter, precedence=precedence)
    else:
        key = None

    # Choose only after reading every mapping, so that only distinct candidates tie and the order
    # of the rows does not matter.
    rewire_map: Dict[URIorCURIE, URIorCURIE] = {}
    for subject_curie, ranked in candidates.items():
        # Only the candidates under the strongest predicate compete for the node.
        strongest = min(ranked.values())
        tied = [tgt for tgt, rank in ranked.items() if rank == strongest]
        if len(tied) == 1:
            rewire_map[subject_curie] = tied[0]
            continue
        message = f"Ambiguous: {subject_curie} -> {' vs '.join(sorted(tied))}"
        if not precedence:
            # TODO make erroring opt-in, can use lexical sort
            #  to be deterministic otherwise
            raise ValueError(message)
        logging.info(message)
        # When two keys are equal, min keeps the first, so the earlier mapping wins.
        rewire_map[subject_curie] = min(tied, key=key)
        logging.info(f"{rewire_map[subject_curie]} has precedence, due to {precedence}")

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
