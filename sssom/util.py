import pandas as pd
import random
import hashlib

# TODO: use sssom_datamodel
SUBJECT_ID = 'subject_id'
SUBJECT_LABEL = 'subject_label'
OBJECT_ID = 'object_id'
OBJECT_LABEL = 'object_label'
PREDICATE_ID = 'predicate_id'
CONFIDENCE = 'confidence'
SUBJECT_CATEGORY = 'subject_category'
OBJECT_CATEGORY = 'object_category'


def parse(filename) -> pd.DataFrame:
    """
    parses a TSV to a pandas frame
    """
    #return from_tsv(filename)
    return pd.read_csv(filename, sep="\t", comment="#")

def collapse(df):
    """
    collapses rows with same S/P/O and combines confidence
    """
    df2 = df.groupby([SUBJECT_ID,PREDICATE_ID,OBJECT_ID])[CONFIDENCE].apply(max).reset_index()
    return df2

def filter_redundant_rows(df : pd.DataFrame) -> pd.DataFrame:
    """
    removes rows if there is another row with same S/O and higher confidence

    :param df:
    :return:
    """

    df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: x + random.random() / 10000)
    dfmax = df.groupby([SUBJECT_ID, OBJECT_ID])[CONFIDENCE].apply(max).reset_index()
    max_conf = {}
    for index, row in dfmax.iterrows():
        max_conf[(row[SUBJECT_ID], row[OBJECT_ID])] = row[CONFIDENCE]
    #return df[df[CONFIDENCE] >= max_conf((df[SUBJECT_ID], df[OBJECT_ID]))]
    return df[df.apply(lambda x: x[CONFIDENCE] >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID])], axis=1)]

def remove_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where no match is found. TODO: https://github.com/OBOFoundry/SSSOM/issues/28
    :param df:
    :return:
    """
    return df[df[PREDICATE_ID] != 'noMatch']
    
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
        
RDF_FORMATS=['ttl', 'turtle', 'nt']

def guess_format(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) > 0:
        f_format = parts[-1]
        if f_format == "rdf":
            return "xml"
        elif f_format == "owl":
            return "xml"
        else:
            return f_format
    else:
        raise Exception(f'Cannot guess format from {filename}')

def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()