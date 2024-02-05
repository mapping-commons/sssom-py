"""Rewriting functionality for RDFlib graphs."""

import logging
from typing import Any, Dict, List, Optional

from linkml_runtime.utils.metamodelcore import URIorCURIE
from rdflib import Graph, URIRef
from sssom_schema import EntityReference, Mapping

from .parsers import to_mapping_set_document
from .util import MappingSetDataFrame

__all__ = [
    "rewire_graph",
]


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

    converter = mdoc.converter
    rewire_map: Dict[URIorCURIE, URIorCURIE] = {}
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
            if src in rewire_map:
                curr_tgt = rewire_map[src]
                logging.info(f"Ambiguous: {src} -> {tgt} vs {curr_tgt}")
                if precedence:
                    curr_pfx, _ = converter.parse_curie(curr_tgt)
                    tgt_pfx, _ = converter.parse_curie(tgt)
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

    uri_ref_rewire_map: Dict[URIRef, URIRef] = {
        URIRef(converter.expand_strict(k)): URIRef(converter.expand_strict(v))
        for k, v in rewire_map.items()
    }

    def rewire_node(n: Any):
        """Rewire node."""
        if isinstance(n, URIRef):
            if n in uri_ref_rewire_map:
                return uri_ref_rewire_map[n]
            else:
                return n
        else:
            return n

    triples = []
    new_triples = []
    num_changed = 0
    for t in g.triples((None, None, None)):
        t2 = [rewire_node(x) for x in t]
        triples.append(t)
        new_triples.append(tuple(t2))
        if t2 != t:
            num_changed += 1
    for t in triples:
        g.remove(t)
    for t in new_triples:
        g.add(t)
    return num_changed
