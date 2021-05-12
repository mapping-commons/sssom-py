from sssom.datamodel_util import MappingSetDataFrame, to_mapping_set_dataframe
from sssom.parsers import to_mapping_set_document
import networkx as nx
import pandas as pd
import hashlib
import statistics

from .sssom_datamodel import slots, MappingSet
from .sssom_document import MappingSetDocument

def to_networkx(msdf: MappingSetDataFrame) -> nx.DiGraph:
    """
    converts a MappingSetDocument to a networkx DiGraph
    """

    doc = to_mapping_set_document(msdf)
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

def split_into_cliques(msdf: MappingSetDataFrame):

    doc = to_mapping_set_document(msdf)
    g = to_networkx(msdf)
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

def invert_dict(d : dict) -> dict:
    invdict = {}
    for k,v in d.items():
        if v not in invdict:
            invdict[v] = []
        invdict[v].append(k)
    return invdict

def get_src(src, id):
    if src is None:
        return id.split(':')[0]
    else:
        return src

def summarize_cliques(doc: MappingSetDocument):
    """
    summary stats on a clique doc
    """
    cliquedocs = split_into_cliques(doc)
    df = pd.DataFrame()
    items = []
    for cdoc in cliquedocs:
        ms = cdoc.mapping_set.mappings
        members = set()
        members_names = set()
        confs = []
        id2src = {}
        for m in ms:
            sub = m.subject_id
            obj = m.object_id
            subsrc = get_src(m.subject_source, sub)
            objsrc = get_src(m.object_source, obj)
            id2src[sub] = subsrc
            id2src[obj] = objsrc
            members.add(sub)
            members.add(obj)
            members_names.add(str(m.subject_label))
            members_names.add(str(m.object_label))
            confs.append(m.confidence)
        src2ids = invert_dict(id2src)
        mstr = "|".join(members)
        md5 = hashlib.md5(mstr.encode('utf-8')).hexdigest()
        item = {
            'id': md5,
            'num_mappings': len(ms),
            'num_members': len(members),
            'members': mstr,
            'members_labels': "|".join(members_names),
            'max_confidence': max(confs),
            'min_confidence': min(confs),
            'avg_confidence': statistics.mean(confs),
            'sources': '|'.join(src2ids.keys()),
            'num_sources': len(src2ids.keys())
        }
        for s,ids in src2ids.items():
            item[s] = '|'.join(ids)
        conflated = False
        total_conflated = 0
        all_conflated = True
        src_counts = []
        for s,ids in src2ids.items():
            n = len(ids)
            item[f'{s}_count'] = n
            item[f'{s}_conflated'] = n > 1
            if n > 1:
                conflated = True
                total_conflated += 1
            else:
                all_conflated = False
            src_counts.append(n)

        item['is_conflated'] = conflated
        item['is_all_conflated'] = all_conflated
        item['total_conflated'] = total_conflated
        item['proportion_conflated'] = total_conflated / len(src2ids.items())
        item['conflation_score'] = (min(src_counts)-1) * len(src2ids.items()) + (statistics.harmonic_mean(src_counts)  -1)
        item['members_count'] = sum(src_counts)
        item['min_count_by_source'] = min(src_counts)
        item['max_count_by_source'] = max(src_counts)
        item['avg_count_by_source'] = statistics.mean(src_counts)
        item['harmonic_mean_count_by_source'] = statistics.harmonic_mean(src_counts)
        ## item['geometric_mean_conflated'] = statistics.geometric_mean(conflateds) py3.8
        items.append(item)
    df = pd.DataFrame(items)
    return df
