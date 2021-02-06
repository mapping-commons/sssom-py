import networkx as nx

from .sssom_datamodel import slots, MappingSet
from .sssom_document import MappingSetDocument

def to_networkx(doc: MappingSetDocument) -> nx.DiGraph:
    """
    converts a MappingSetDocument to a networkx DiGraph
    """
    g = nx.DiGraph()
    M = {
        'owl:subClassOf',
    }
    for mapping in doc.mapping_set.mappings:
        s = mapping.subject_id
        o = mapping.object_id
        p = mapping.predicate_id
        # TODO: this is copypastad from export_ptable
        if p == 'owl:equivalentClass':
            pi = 2
        elif p == 'skos:exactMatch':
            pi = 2
        elif p == 'skos:closeMatch':
            # TODO: consider distributing
            pi = 2
        elif p == 'owl:subClassOf':
            pi = 0
        elif p == 'skos:broadMatch':
            pi = 0
        elif p == 'inverseOf(owl:subClassOf)':
            pi = 1
        elif p == 'skos:narrowMatch':
            pi = 1
        elif p == 'owl:differentFrom':
            pi = 3
        elif p == 'dbpedia-owl:different':
            pi = 3
        if pi == 0:
            g.add_edge(o, s)
        elif pi == 1:
            g.add_edge(s, o)
        elif pi == 2:
            g.add_edge(s, o)
            g.add_edge(o, s)
    return g

def split_into_cliques(doc: MappingSetDocument):
    g = to_networkx(doc)
    gen = nx.algorithms.components.strongly_connected_components(g)

    node_to_comp = {}
    comp_id = 0
    newdocs = []
    for comp in sorted(gen, key=len, reverse=True):
        for n in comp:
            node_to_comp[n] = comp_id
        comp_id += 1
        newdocs.append(MappingSetDocument(curie_map=doc.curie_map,
                                          mapping_set=MappingSet(mappings=[])))


    for m in doc.mapping_set.mappings:
        comp_id = node_to_comp[m.subject_id]
        subdoc = newdocs[comp_id]
        subdoc.mapping_set.mappings.append(m)
    return newdocs