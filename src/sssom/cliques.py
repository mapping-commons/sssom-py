"""Utilities for identifying and working with cliques/SCCs in mappings graphs."""

import hashlib
import statistics
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Optional, Set

import pandas as pd
from sssom_schema import Mapping, MappingSet

from sssom.constants import (
    DEFAULT_LICENSE,
    OWL_DIFFERENT_FROM,
    OWL_EQUIVALENT_CLASS,
    RDFS_SUBCLASS_OF,
    SKOS_BROAD_MATCH,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_NARROW_MATCH,
    SSSOM_SUPERCLASS_OF,
    SSSOM_URI_PREFIX,
)

from .parsers import to_mapping_set_document
from .sssom_document import MappingSetDocument
from .util import MappingSetDataFrame

if TYPE_CHECKING:
    import networkx


def to_digraph(msdf: MappingSetDataFrame) -> "networkx.DiGraph":
    """Convert to a graph where the nodes are entities' CURIEs and edges are their mappings."""
    import networkx as nx

    doc = to_mapping_set_document(msdf)
    g = nx.DiGraph()
    if doc.mapping_set.mappings is not None:
        for mapping in doc.mapping_set.mappings:
            if not isinstance(mapping, Mapping):
                raise TypeError
            s = mapping.subject_id
            o = mapping.object_id
            p = mapping.predicate_id
            # TODO: this is copypastad from export_ptable

            pi = None

            if p == OWL_EQUIVALENT_CLASS:
                pi = 2
            elif p == SKOS_EXACT_MATCH:
                pi = 2
            elif p == SKOS_CLOSE_MATCH:
                # TODO: consider distributing
                pi = 2
            elif p == RDFS_SUBCLASS_OF:
                pi = 0
            elif p == SKOS_BROAD_MATCH:
                pi = 0
            elif p == SSSOM_SUPERCLASS_OF:
                pi = 1
            elif p == SKOS_NARROW_MATCH:
                pi = 1
            elif p == OWL_DIFFERENT_FROM:
                pi = 3

            if pi == 0:
                g.add_edge(o, s)
            elif pi == 1:
                g.add_edge(s, o)
            elif pi == 2:
                g.add_edge(s, o)
                g.add_edge(o, s)
    return g


def split_into_cliques(msdf: MappingSetDataFrame) -> List[MappingSetDocument]:
    """Split a MappingSetDataFrames documents corresponding to a strongly connected components of the associated graph.

    :param msdf: MappingSetDataFrame object
    :raises TypeError: If Mappings is not of type List
    :raises TypeError: If each mapping is not of type Mapping
    :raises TypeError: If Mappings is not of type List
    :return: List of MappingSetDocument objects
    """
    import networkx as nx

    doc = to_mapping_set_document(msdf)
    graph = to_digraph(msdf)
    components_it = nx.algorithms.components.strongly_connected_components(graph)
    components = sorted(components_it, key=len, reverse=True)

    curie_to_component = {}
    for i, component in enumerate(components):
        for curie in component:
            curie_to_component[curie] = i
    documents = [
        MappingSetDocument(
            converter=doc.converter,
            mapping_set=MappingSet(
                mapping_set_id=f"{SSSOM_URI_PREFIX}mappings/{uuid.uuid4()}",
                license=doc.mapping_set.license or DEFAULT_LICENSE,
            ),
        )
        for _ in components
    ]

    if not isinstance(doc.mapping_set.mappings, list):
        raise TypeError
    for mapping in doc.mapping_set.mappings:
        if not isinstance(mapping, Mapping):
            raise TypeError
        subject_document = documents[curie_to_component[mapping.subject_id]]
        if not isinstance(subject_document.mapping_set.mappings, list):
            raise TypeError
        subject_document.mapping_set.mappings.append(mapping)
    return documents


def group_values(d: Dict[str, str]) -> Dict[str, List[str]]:
    """Group all keys in the dictionary that share the same value."""
    rv: DefaultDict[str, List[str]] = defaultdict(list)
    for k, v in d.items():
        rv[v].append(k)
    return dict(rv)


def get_src(src: Optional[str], curie: str):
    """Get prefix of subject/object in the MappingSetDataFrame.

    :param src: Source
    :param curie: CURIE
    :return: Source
    """
    if src is None:
        return curie.split(":")[0]
    else:
        return src


def summarize_cliques(doc: MappingSetDataFrame):
    """Summarize stats on a clique doc."""
    cliquedocs = split_into_cliques(doc)
    items = []
    for cdoc in cliquedocs:
        mappings = cdoc.mapping_set.mappings
        if mappings is None:
            continue
        members: Set[str] = set()
        members_names: Set[str] = set()
        confs: List[float] = []
        id2src: Dict[str, str] = {}
        for mapping in mappings:
            if not isinstance(mapping, Mapping):
                raise TypeError
            sub = str(mapping.subject_id)
            obj = str(mapping.object_id)
            id2src[sub] = get_src(mapping.subject_source, sub)
            id2src[obj] = get_src(mapping.object_source, obj)
            members.add(sub)
            members.add(obj)
            members_names.add(str(mapping.subject_label))
            members_names.add(str(mapping.object_label))
            if mapping.confidence is not None:
                confs.append(mapping.confidence)
        src2ids = group_values(id2src)
        mstr = "|".join(members)
        md5 = hashlib.md5(mstr.encode("utf-8")).hexdigest()  # noqa:S303
        item = {
            "id": md5,
            "num_mappings": len(mappings),
            "num_members": len(members),
            "members": mstr,
            "members_labels": "|".join(members_names),
            "max_confidence": max(confs),
            "min_confidence": min(confs),
            "avg_confidence": statistics.mean(confs),
            "sources": "|".join(src2ids.keys()),
            "num_sources": len(src2ids.keys()),
        }
        for s, ids in src2ids.items():
            item[s] = "|".join(ids)
        conflated = False
        total_conflated = 0
        all_conflated = True
        src_counts = []
        for s, ids in src2ids.items():
            n = len(ids)
            item[f"{s}_count"] = n
            item[f"{s}_conflated"] = n > 1
            if n > 1:
                conflated = True
                total_conflated += 1
            else:
                all_conflated = False
            src_counts.append(n)

        item["is_conflated"] = conflated
        item["is_all_conflated"] = all_conflated
        item["total_conflated"] = total_conflated
        item["proportion_conflated"] = total_conflated / len(src2ids.items())
        item["conflation_score"] = (min(src_counts) - 1) * len(src2ids.items()) + (
            statistics.harmonic_mean(src_counts) - 1
        )
        item["members_count"] = sum(src_counts)
        item["min_count_by_source"] = min(src_counts)
        item["max_count_by_source"] = max(src_counts)
        item["avg_count_by_source"] = statistics.mean(src_counts)
        item["harmonic_mean_count_by_source"] = statistics.harmonic_mean(src_counts)
        # item['geometric_mean_conflated'] = statistics.geometric_mean(conflateds) py3.8
        items.append(item)
    df = pd.DataFrame(items)
    return df
