import pandas as pd

SUBJECT_ID = 'subject_id'
SUBJECT_LABEL = 'subject_label'
OBJECT_ID = 'object_id'
OBJECT_LABEL = 'object_label'
PREDICATE_ID = 'predicate_id'
CONFIDENCE = 'confidence'

def parse(filename):
    """
    parses a TSV to a pandas frame
    """
    return pd.read_csv(filename, sep="\t", comment="#")

def collapse(df):
    """
    collapses rows with same S/P/O and combines confidence
    """
    df2 = df.groupby([SUBJECT_ID,PREDICATE_ID,OBJECT_ID])[CONFIDENCE].apply(max).reset_index()
    return df2
    
def export_ptable(df, priors=[0.02, 0.02, 0.02, 0.02]):
    """
    ptable
    """
    df = collapse(df)
    pmap = {}
    for _, row in df.iterrows():
        p = row[PREDICATE_ID]
        if p == 'owl:equivalentClass':
            pi = 2
        elif p == 'owl:subClassOf':
            pi = 0
        elif p == 'inverseOf(owl:subClassOf)':
            pi = 1
        else:
            continue
        s = row[SUBJECT_ID]
        o = row[OBJECT_ID]
        pair = (s,o)
        if id not in pmap:
            pmap[pair] = priors
        pmap[pair][pi] = row[CONFIDENCE]
    rows = []
    for pair, pvals in pmap.items():
        sump = sum(pvals)
        if sump >= 1 :
            pvals = [p/sump for p in pvals]
        else:
            extra = (1-sump)/4
            pvals = [p+extra for p in pvals]
        pvalsj = '\t'.join(str(p) for p in pvals)
        row = f'{pair[0]}\t{pair[1]}\t{pvalsj}'
        print(row)
        
            
