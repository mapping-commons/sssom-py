"""Rewriting functionality for RDFlib graphs."""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Sequence, cast

import curies
from linkml_runtime.utils.metamodelcore import URIorCURIE
from rdflib import Graph, Node, URIRef
from sssom_schema import EntityReference, Mapping

from .constants import (
    NO_TERM_FOUND,
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
    PREDICATE_MODIFIER_NOT,
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
    predicates: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> int:
    """Rewire an RDF graph in place, replacing each mapped node by its mapping partner.

    When a node has several candidates, the one under the predicate listed first in
    ``predicates`` wins, so the order of ``predicates`` is the order of preference. Mappings that
    agree on a candidate count once, under the most preferred of their predicates. Only the
    equivalences keep the graph's meaning: a ``skos:broadMatch`` mapping puts the broader entity
    even in the triples that describe the narrower one. Negated mappings, and mappings to
    ``sssom:NoTermFound``, are skipped.

    :param g: The graph to rewire.
    :param mset: The mapping set to rewire by.
    :param subject_to_object: If true, replace the subject of each mapping by its object. If
        false, replace the object by the subject.
    :param precedence: Prefixes in order of preference, which choose between equally preferred
        candidates for a node. When it does not decide, the earlier mapping wins.
    :param predicates: The mapping predicates to rewire by, as CURIEs or IRIs, in order of
        preference. A single string counts as one predicate. Defaults to ``owl:equivalentClass``
        and ``owl:equivalentProperty``, which are equally preferred.
    :param strict: If true, raise an error when no mapping is left to rewire by, instead of
        logging a warning.

    :return: The number of triples that changed.
    :raises TypeError: If the mapping set has no mappings, or a mapping in use has a subject or
        object that is not an entity reference.
    :raises curies.api.ExpansionError: If a chosen predicate, or the predicate of a mapping, does
        not expand.
    :raises ValueError: If equally preferred candidates tie and no ``precedence`` is given, or if
        ``strict`` is true and no mapping is left to rewire by.
    """
    mdoc = to_mapping_set_document(mset)
    if mdoc.mapping_set.mappings is None:
        raise TypeError

    converter: curies.Converter = mdoc.converter
    # The default predicates are equally preferred, so precedence decides between them.
    equally_preferred = predicates is None
    if predicates is None:
        predicates = sorted(_REWIRE_PREDICATE_CURIES)
    elif isinstance(predicates, str):
        # A str is a Sequence[str] too, so it would otherwise be read character by character.
        predicates = [predicates]
    # Rank each predicate by its place in the order of preference, where 0 is the most preferred,
    # and keep the first rank of a predicate given twice. Compare predicates as IRIs, so that a
    # CURIE and its IRI count as the same predicate.
    rank_by_iri: Dict[str, int] = {}
    for position, predicate_uri_or_curie in enumerate(predicates):
        predicate_iri = converter.expand_or_standardize(predicate_uri_or_curie, strict=True)
        rank_by_iri.setdefault(predicate_iri, 0 if equally_preferred else position)

    # Map each node to its candidate replacements, in the order of their first mapping, each with
    # the best rank among its mappings.
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
        # if strict expansion errors, then there's a mapping set integrity
        # problem that the rewire workflow shouldn't handle. curies does not expand a subclass of
        # str, such as an EntityReference, so pass a str.
        rank = rank_by_iri.get(converter.expand_or_standardize(str(m.predicate_id), strict=True))
        if rank is None:
            continue
        if subject_to_object:
            src, tgt = m.subject_id, m.object_id
        else:
            src, tgt = m.object_id, m.subject_id
        if not isinstance(src, EntityReference) or not isinstance(tgt, EntityReference):
            raise TypeError
        # A candidate keeps the best rank among its mappings, so agreeing mappings count once.
        candidates[src][tgt] = min(rank, candidates[src].get(tgt, rank))

    # Warn, or fail when strict, rather than return the graph unchanged and say nothing.
    if not candidates:
        message = f"No mapping to rewire by with the predicates {', '.join(predicates)}"
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
        # Only the candidates under the most preferred predicate compete for the node.
        best = min(ranked.values())
        tied = [tgt for tgt, rank in ranked.items() if rank == best]
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
