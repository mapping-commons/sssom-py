"""Rewriting functionality for RDFlib graphs."""

from __future__ import annotations

import logging
from typing import Collection, Dict, FrozenSet, Optional, Sequence, cast

import curies
from linkml_runtime.utils.metamodelcore import URIorCURIE
from rdflib import Graph, Node, URIRef
from sssom_schema import EntityReference, Mapping

from .constants import (
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

#: The predicates that :func:`rewire_graph` uses when it is given none: the OWL equivalences.
_DEFAULT_PREDICATES: FrozenSet[str] = frozenset({OWL_EQUIVALENT_CLASS, OWL_EQUIVALENT_PROPERTY})

#: The strength rank of each predicate, where 0 is the strongest. A predicate not listed here
#: ranks below all of them.
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


def _to_iri(converter: curies.Converter, reference: str) -> str:
    """Expand a CURIE to an IRI so that both spellings of a predicate compare equal.

    :param converter: The converter whose prefix map is used for the expansion.
    :param reference: A CURIE or an IRI.
    :return: The IRI, or ``reference`` unchanged when it is an IRI already or cannot be expanded.
    """
    return converter.expand(reference, passthrough=True) or reference


def _precedence_index(converter: curies.Converter, precedence: Sequence[str], curie: str) -> int:
    """Find how strongly ``precedence`` prefers a CURIE, by the position of its prefix.

    :param converter: The converter that parses the CURIE.
    :param precedence: Prefixes in order of preference.
    :param curie: A candidate replacement.
    :return: The position of the prefix in ``precedence``, or the length of ``precedence`` when
        the prefix is not listed, so that a smaller number is a stronger preference.
    """
    prefix = converter.parse_curie(curie, strict=True).prefix
    return precedence.index(prefix) if prefix in precedence else len(precedence)


def rewire_graph(
    g: Graph,
    mset: MappingSetDataFrame,
    subject_to_object: bool = True,
    precedence: Optional[Sequence[str]] = None,
    predicates: Optional[Collection[str]] = None,
    strict: bool = False,
) -> int:
    """Rewire an RDF graph in place, replacing each mapped entity with its mapping partner.

    Each mapping whose predicate is in ``predicates`` replaces its subject with its object, or its
    object with its subject when ``subject_to_object`` is False. Negated mappings are skipped. Only
    the equivalences keep what the graph says: a ``skos:broadMatch`` mapping puts the broader entity
    even in the triples that describe the narrower one.

    When a node has several candidates, the one under the strongest predicate wins. The order,
    strongest first, is the OWL equivalences, ``skos:exactMatch``, ``skos:closeMatch``,
    ``skos:broadMatch`` and ``skos:narrowMatch``, ``skos:relatedMatch``, then any other. Mappings
    that agree on a candidate count once, under their strongest predicate. ``precedence`` breaks a
    tie in favour of the candidate whose prefix it lists first, and when that does not decide, the
    earlier mapping wins. Without ``precedence``, a tie raises :class:`ValueError`.

    :param g: The graph to rewire.
    :param mset: The mapping set to rewire by.
    :param subject_to_object: If False, replace objects with subjects instead.
    :param precedence: Prefixes in order of preference, to break a tie.
    :param predicates: The mapping predicates to use, as CURIEs or IRIs. Defaults to the OWL
        equivalences.
    :param strict: If True, raise :class:`ValueError` instead of a warning when no mapping is left
        to rewire by.
    :return: The number of triples that changed.
    :raises TypeError: If the mapping set has no mappings, or a mapping in use has a subject or
        object that is not an entity reference.
    :raises ValueError: On a tie without ``precedence``, or under ``strict`` when no mapping is left
        to rewire by.
    """
    mdoc = to_mapping_set_document(mset)
    if mdoc.mapping_set.mappings is None:
        raise TypeError

    converter: curies.Converter = mdoc.converter
    if predicates is None:
        predicates = _DEFAULT_PREDICATES
    # Compare predicates as IRIs, so that a CURIE and its IRI count as the same predicate.
    predicate_iris = {_to_iri(converter, p) for p in predicates}
    rank_by_iri = {_to_iri(converter, p): rank for p, rank in _PREDICATE_RANK.items()}

    # The candidate replacements of each node, each with the rank of its strongest mapping.
    candidates: Dict[URIorCURIE, Dict[URIorCURIE, int]] = {}
    for m in mdoc.mapping_set.mappings:
        if not isinstance(m, Mapping):
            continue
        # A negated mapping records that the two entities do not match.
        if m.predicate_modifier == PREDICATE_MODIFIER_NOT:
            continue
        predicate_iri = _to_iri(converter, str(m.predicate_id))
        if predicate_iri not in predicate_iris:
            continue
        if subject_to_object:
            src, tgt = m.subject_id, m.object_id
        else:
            src, tgt = m.object_id, m.subject_id
        if not isinstance(src, EntityReference) or not isinstance(tgt, EntityReference):
            raise TypeError
        rank = rank_by_iri.get(predicate_iri, _UNRANKED)
        targets = candidates.setdefault(src, {})
        # A candidate keeps the rank of its strongest mapping, so agreeing mappings count once.
        targets[tgt] = min(rank, targets.get(tgt, rank))

    # Choose only after reading every mapping, so that mappings that agree count once and the
    # order of the rows does not matter.
    rewire_map: Dict[URIorCURIE, URIorCURIE] = {}
    for src, targets in candidates.items():
        # Only the candidates under the strongest predicate compete for the node.
        strongest = min(targets.values())
        tied = [tgt for tgt, rank in targets.items() if rank == strongest]
        if len(tied) == 1:
            rewire_map[src] = tied[0]
            continue
        logging.info(f"Ambiguous: {src} -> {' vs '.join(tied)}")
        if not precedence:
            raise ValueError(f"Ambiguous: {src} -> {' vs '.join(tied)}")
        # On equal keys, min keeps the first, so the earlier mapping wins.
        rewire_map[src] = min(tied, key=lambda tgt: _precedence_index(converter, precedence, tgt))
        logging.info(f"{rewire_map[src]} has precedence, due to {precedence}")

    # Say when nothing will change, instead of returning the graph unchanged.
    if not rewire_map:
        message = f"No mapping to rewire by with the predicates {', '.join(sorted(predicates))}"
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
