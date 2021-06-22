import hashlib
import logging
import random
import sys
import contextlib

from typing import Dict, List

import pandas as pd

from sssom.datamodel_util import MappingSetDiff, EntityPair
from sssom.sssom_datamodel import Entity

# TODO: use sssom_datamodel
SUBJECT_ID = "subject_id"
SUBJECT_LABEL = "subject_label"
OBJECT_ID = "object_id"
OBJECT_LABEL = "object_label"
PREDICATE_ID = "predicate_id"
CONFIDENCE = "confidence"
SUBJECT_CATEGORY = "subject_category"
OBJECT_CATEGORY = "object_category"
SUBJECT_SOURCE = "subject_source"
OBJECT_SOURCE = "object_source"
COMMENT = "comment"
MAPPING_PROVIDER = "mapping_provider"


def parse(filename) -> pd.DataFrame:
    """
    parses a TSV to a pandas frame
    """
    # return from_tsv(filename)
    logging.info(f"Parsing {filename}")
    return pd.read_csv(filename, sep="\t", comment="#")
    # return read_pandas(filename)


def collapse(df):
    """
    collapses rows with same S/P/O and combines confidence
    """
    df2 = (
        df.groupby([SUBJECT_ID, PREDICATE_ID, OBJECT_ID])[CONFIDENCE]
        .apply(max)
        .reset_index()
    )
    return df2


def filter_redundant_rows(df: pd.DataFrame, ignore_predicate=False) -> pd.DataFrame:
    """
    removes rows if there is another row with same S/O and higher confidence

    :param df:
    :return:
    """
    # tie-breaker
    df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: x + random.random() / 10000)
    if ignore_predicate:
        key = [SUBJECT_ID, OBJECT_ID]
    else:
        key = [SUBJECT_ID, OBJECT_ID, PREDICATE_ID]
    dfmax = df.groupby(key)[CONFIDENCE].apply(max).reset_index()
    max_conf = {}
    for index, row in dfmax.iterrows():
        if ignore_predicate:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID])] = row[CONFIDENCE]
        else:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID], row[PREDICATE_ID])] = row[
                CONFIDENCE
            ]
    if ignore_predicate:
        return df[
            df.apply(
                lambda x: x[CONFIDENCE] >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID])],
                axis=1,
            )
        ]
    else:
        return df[
            df.apply(
                lambda x: x[CONFIDENCE]
                >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID], x[PREDICATE_ID])],
                axis=1,
            )
        ]


def remove_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where no match is found. TODO: https://github.com/OBOFoundry/SSSOM/issues/28
    :param df:
    :return:
    """
    return df[df[PREDICATE_ID] != "noMatch"]


def create_entity(row, id: str, mappings: Dict) -> Entity:
    e = Entity(id=id)
    for k, v in mappings.items():
        if k in e:
            e[k] = v
    return e


def group_mappings(df: pd.DataFrame) -> Dict[EntityPair, List]:
    """
    group mappings by EntityPairs
    """
    mappings: Dict = {}
    for _, row in df.iterrows():
        sid = row[SUBJECT_ID]
        oid = row[OBJECT_ID]
        s = create_entity(
            row,
            sid,
            {
                "label": SUBJECT_LABEL,
                "category": SUBJECT_CATEGORY,
                "source": SUBJECT_SOURCE,
            },
        )
        o = create_entity(
            row,
            oid,
            {
                "label": OBJECT_LABEL,
                "category": OBJECT_CATEGORY,
                "source": OBJECT_SOURCE,
            },
        )
        pair = EntityPair(s, o)
        if pair not in mappings:
            mappings[pair] = []
        mappings[pair].append(row)
    return mappings


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> MappingSetDiff:
    """
    Perform a diff between two SSSOM dataframes

    Currently does not discriminate between mappings with different predicates
    """
    mappings1 = group_mappings(df1.copy())
    mappings2 = group_mappings(df2.copy())
    tuples1 = set(mappings1.keys())
    tuples2 = set(mappings2.keys())
    d = MappingSetDiff()
    d.unique_tuples1 = tuples1.difference(tuples2)
    d.unique_tuples2 = tuples2.difference(tuples1)
    d.common_tuples = tuples1.intersection(tuples2)
    all_tuples = tuples1.union(tuples2)
    all_ids = set()
    for t in all_tuples:
        all_ids.update({t.subject_entity.id, t.object_entity.id})
    rows = []
    for t in d.unique_tuples1:
        for r in mappings1[t]:
            r[COMMENT] = "UNIQUE_1"
        rows += mappings1[t]
    for t in d.unique_tuples2:
        for r in mappings2[t]:
            r[COMMENT] = "UNIQUE_2"
        rows += mappings2[t]
    for t in d.common_tuples:
        new_rows = mappings1[t] + mappings2[t]
        for r in new_rows:
            r[COMMENT] = "COMMON_TO_BOTH"
        rows += new_rows
    # for r in rows:
    #    r['other'] = 'synthesized sssom file'
    d.combined_dataframe = pd.DataFrame(rows)
    return d


@contextlib.contextmanager
def smart_open(filename=None):
    # https://stackoverflow.com/questions/17602878/how-to-handle-both-with-open-and-sys-stdout-nicely
    if filename and filename != "-":
        fh = open(filename, "w")
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def dataframe_to_ptable(
    df: pd.DataFrame, priors=[0.02, 0.02, 0.02, 0.02], inverse_factor: float = 0.5
):
    """
    exports kboom ptable
    :param df: SSSOM dataframe
    :param inverse_factor: relative weighting of probability of inverse of predicate
    :return:
    """
    df = collapse(df)
    rows = []
    for _, row in df.iterrows():
        s = row[SUBJECT_ID]
        o = row[OBJECT_ID]
        c = row[CONFIDENCE]
        # confidence of inverse
        # e.g. if Pr(super) = 0.2, then Pr(sub) = (1-0.2) * IF
        ic = (1.0 - c) * inverse_factor
        # residual confidence
        rc = (1 - (c + ic)) / 2.0

        p = row[PREDICATE_ID]
        if p == "owl:equivalentClass":
            pi = 2
        elif p == "skos:exactMatch":
            pi = 2
        elif p == "skos:closeMatch":
            # TODO: consider distributing
            pi = 2
        elif p == "owl:subClassOf":
            pi = 0
        elif p == "skos:broadMatch":
            pi = 0
        elif p == "inverseOf(owl:subClassOf)":
            pi = 1
        elif p == "skos:narrowMatch":
            pi = 1
        elif p == "owl:differentFrom":
            pi = 3
        elif p == "dbpedia-owl:different":
            pi = 3
        else:
            raise Exception(f"Unknown predicate {p}")

        if pi == 0:
            # subClassOf
            ps = (c, ic, rc, rc)
        elif pi == 1:
            # superClassOf
            ps = (ic, c, rc, rc)
        elif pi == 2:
            # equivalent
            ps = (rc, rc, c, ic)
        elif pi == 3:
            # sibling
            ps = (rc, rc, ic, c)
        else:
            raise Exception(f"pi: {pi}")
        row = [s, o] + [str(p) for p in ps]
        rows.append(row)
    return rows


RDF_FORMATS = ["ttl", "turtle", "nt"]


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()
