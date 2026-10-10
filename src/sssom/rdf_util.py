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
    SKOS_BROAD_MATCH,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_NARROW_MATCH,
    SKOS_RELATED_MATCH,
)
from .parsers import to_mapping_set_document
from .util import MappingSetDataFrame

__all__ = [
    "EQUIVALENCE_PREDICATES",
    "HIERARCHICAL_PREDICATES",
    "REWIRE_FLAVORS",
    "SKOS_MATCH_PREDICATES",
    "rewire_graph",
]

#: The predicates :func:`rewire_graph` honours when it is given none: the OWL equivalences.
EQUIVALENCE_PREDICATES: FrozenSet[str] = frozenset({OWL_EQUIVALENT_CLASS, OWL_EQUIVALENT_PROPERTY})

#: Every SKOS mapping predicate.
SKOS_MATCH_PREDICATES: FrozenSet[str] = frozenset(
    {SKOS_EXACT_MATCH, SKOS_CLOSE_MATCH, SKOS_BROAD_MATCH, SKOS_NARROW_MATCH, SKOS_RELATED_MATCH}
)

#: Predicates whose two sides are not interchangeable. A mapping with one of these predicates is
#: only ever applied from its subject to its object, never the other way round.
HIERARCHICAL_PREDICATES: FrozenSet[str] = frozenset({SKOS_BROAD_MATCH, SKOS_NARROW_MATCH})

#: Named sets of predicates for :func:`rewire_graph` and the ``--flavor`` option of
#: ``sssom rewire``. Each set contains the one before it.
REWIRE_FLAVORS: Dict[str, FrozenSet[str]] = {
    "equivalence": EQUIVALENCE_PREDICATES,
    "exact": EQUIVALENCE_PREDICATES | {SKOS_EXACT_MATCH},
    "exact-broad": EQUIVALENCE_PREDICATES | {SKOS_EXACT_MATCH, SKOS_BROAD_MATCH},
    "any": EQUIVALENCE_PREDICATES | SKOS_MATCH_PREDICATES,
}

#: How strongly each predicate binds its two sides, strongest first. When a node has candidate
#: replacements under several predicates, the one under the strongest predicate wins.
#: Predicates not listed here rank below all of them.
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


def rewire_graph(
    g: Graph,
    mset: MappingSetDataFrame,
    subject_to_object: bool = True,
    precedence: Optional[Sequence[str]] = None,
    predicates: Optional[Collection[str]] = None,
) -> int:
    """Rewire an RDF graph in place, replacing each mapped entity by its mapping partner.

    Only mappings whose predicate is in ``predicates`` are used. Each one replaces its subject by
    its object wherever the subject occurs in the graph, or its object by its subject when
    ``subject_to_object`` is False. The hierarchical predicates in :data:`HIERARCHICAL_PREDICATES`
    (``skos:broadMatch`` and ``skos:narrowMatch``) are never reversed: their mappings are applied
    from subject to object only and are skipped when ``subject_to_object`` is False.

    When a node has several candidate replacements, the candidate under the strongest predicate
    wins: OWL equivalence, then ``skos:exactMatch``, then ``skos:closeMatch``, then
    ``skos:broadMatch`` and ``skos:narrowMatch``, then ``skos:relatedMatch``, then any other
    predicate. Between candidates under equally strong predicates ``precedence`` decides: a
    candidate whose prefix is listed beats one whose prefix is not, an earlier prefix beats a
    later one, and otherwise the candidate from the earlier mapping stays. Without ``precedence``
    such a tie raises :class:`ValueError`. Several mappings that agree on the replacement are not
    a tie.

    :param g: The graph to rewire. It is modified in place.
    :param mset: The mapping set whose mappings drive the rewiring.
    :param subject_to_object: If True (the default), replace subjects by objects. If False,
        replace objects by subjects for the symmetric predicates and skip the hierarchical ones.
    :param precedence: Prefixes in order of preference, used to decide between candidate
        replacements under equally strong predicates.
    :param predicates: The mapping predicates to honour, as CURIEs or IRIs. Defaults to
        :data:`EQUIVALENCE_PREDICATES`, that is ``owl:equivalentClass`` and
        ``owl:equivalentProperty``. The named sets in :data:`REWIRE_FLAVORS` are convenient values.
    :return: The number of triples that changed.
    :raises TypeError: If the mapping set has no mappings, or a mapping that is used has a subject
        or object that is not an entity reference.
    :raises ValueError: If a node has several candidate replacements under equally strong
        predicates and no ``precedence`` is given.
    """
    mdoc = to_mapping_set_document(mset)
    if mdoc.mapping_set.mappings is None:
        raise TypeError

    converter: curies.Converter = mdoc.converter
    if predicates is None:
        predicates = EQUIVALENCE_PREDICATES
    predicate_iris = {_to_iri(converter, p) for p in predicates}
    hierarchical_iris = {_to_iri(converter, p) for p in HIERARCHICAL_PREDICATES}
    rank_by_iri = {_to_iri(converter, p): rank for p, rank in _PREDICATE_RANK.items()}

    rewire_map: Dict[URIorCURIE, URIorCURIE] = {}
    rank_map: Dict[URIorCURIE, int] = {}
    for m in mdoc.mapping_set.mappings:
        if not isinstance(m, Mapping):
            continue
        predicate_iri = _to_iri(converter, str(m.predicate_id))
        if predicate_iri not in predicate_iris:
            continue
        if predicate_iri in hierarchical_iris and not subject_to_object:
            logging.info(
                f"Skipping {m.predicate_id} mapping of {m.subject_id} to {m.object_id}: "
                "hierarchical mappings are not applied in reverse"
            )
            continue
        if subject_to_object:
            src, tgt = m.subject_id, m.object_id
        else:
            src, tgt = m.object_id, m.subject_id
        if not isinstance(src, EntityReference) or not isinstance(tgt, EntityReference):
            raise TypeError
        rank = rank_by_iri.get(predicate_iri, _UNRANKED)
        if src not in rewire_map:
            rewire_map[src] = tgt
            rank_map[src] = rank
            continue
        curr_tgt = rewire_map[src]
        if tgt == curr_tgt:
            # The same replacement under another predicate, or a repeated mapping, is no conflict.
            continue
        curr_rank = rank_map[src]
        if rank != curr_rank:
            if rank < curr_rank:
                logging.info(f"{tgt} replaces {curr_tgt} for {src}: {m.predicate_id} is stronger")
                rewire_map[src] = tgt
                rank_map[src] = rank
            continue
        logging.info(f"Ambiguous: {src} -> {tgt} vs {curr_tgt}")
        if not precedence:
            raise ValueError(f"Ambiguous: {src} -> {tgt} vs {curr_tgt}")
        curr_pfx = converter.parse_curie(curr_tgt, strict=True).prefix
        tgt_pfx = converter.parse_curie(tgt, strict=True).prefix
        if tgt_pfx in precedence:
            if curr_pfx not in precedence or precedence.index(tgt_pfx) < precedence.index(curr_pfx):
                rewire_map[src] = tgt
                logging.info(f"{tgt} has precedence, due to {precedence}")

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
