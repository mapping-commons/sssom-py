import logging
from typing import List

from rdflib import Graph, URIRef
from rdflib.plugins.memory import Any

from .parsers import to_mapping_set_document
from .sssom_datamodel import Mapping
from .util import MappingSetDataFrame


def rewire_graph(
    g: Graph,
    mset: MappingSetDataFrame,
    subject_to_object=True,
    precedence: List[str] = None,
) -> str:
    """
    rewires an RDF Graph replacing using equivalence mappings
    """
    pm = mset.prefixmap
    mdoc = to_mapping_set_document(mset)
    rewire_map = {}

    def expand_curie(curie: str):
        pfx, local = curie.split(":")
        return URIRef(f"{pm[pfx]}{local}")

    for m in mdoc.mapping_set.mappings:
        m: Mapping
        if m.predicate_id in {"owl:equivalentClass", "owl:equivalentProperty"}:
            if subject_to_object:
                src, tgt = m.subject_id, m.object_id
            else:
                src, tgt = m.object_id, m.subject_id
            if src in rewire_map:
                curr_tgt = rewire_map[src]
                logging.info(f"Ambiguous: {src} -> {tgt} vs {curr_tgt}")
                if precedence:
                    curr_pfx, _ = curr_tgt.split(":")
                    tgt_pfx, _ = tgt.split(":")
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
    rewire_map = {expand_curie(k): expand_curie(v) for k, v in rewire_map.items()}

    def rewire_node(n: Any):
        if isinstance(n, URIRef):
            if n in rewire_map:
                return rewire_map[n]
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
