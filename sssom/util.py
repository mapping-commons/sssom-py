import contextlib
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from io import FileIO, StringIO
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from urllib.request import urlopen

import numpy as np
import pandas as pd
import validators
import yaml

from .context import get_default_metadata, get_jsonld_context
from .sssom_datamodel import Entity, slots
from .sssom_document import MappingSetDocument

SSSOM_READ_FORMATS = [
    "tsv",
    "rdf",
    "owl",
    "alignment-api-xml",
    "obographs-json",
    "json",
]
SSSOM_EXPORT_FORMATS = ["tsv", "rdf", "owl", "json"]

SSSOM_DEFAULT_RDF_SERIALISATION = "turtle"

SSSOM_URI_PREFIX = "http://w3id.org/sssom/"

# TODO: use sssom_datamodel (Mapping Class)
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
MATCH_TYPE = "match_type"
HUMAN_CURATED_MATCH_TYPE = "HumanCurated"
MAPPING_SET_ID = "mapping_set_id"
DEFAULT_MAPPING_SET_ID = f"{SSSOM_URI_PREFIX}mappings/default"

URI_SSSOM_MAPPINGS = f"{SSSOM_URI_PREFIX}mappings"

#: The 3 columns whose combination would be used as primary keys while merging/grouping
KEY_FEATURES = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID]


@dataclass
class MappingSetDataFrame:
    """
    A collection of mappings represented as a DataFrame, together with additional metadata
    """

    df: Optional[pd.DataFrame] = None  # Mappings
    prefixmap: Dict[str, Any] = None  # maps CURIE prefixes to URI bases
    metadata: Optional[Dict[str, str]] = None  # header metadata excluding prefixes

    def merge(
        self, msdf2: "MappingSetDataFrame", inplace: bool = True
    ) -> "MappingSetDataFrame":
        """Merges two MappingSetDataframes

        Args:
            msdf: Secondary MappingSetDataFrame (self => primary)
            inplace:
                if true, msdf2 is merged into the calling MappingSetDataFrame, if false, it simply return
                the merged data frame.

        Returns:
            MappingSetDataFrame: Merged MappingSetDataFrame
        """
        msdf = merge_msdf(msdf1=self, msdf2=msdf2)
        if inplace:
            self.df = msdf.df
            self.prefixmap = msdf.prefixmap
            self.metadata = msdf.metadata
            # FIXME should return self if inplace
        return msdf

    def __str__(self):
        description = "SSSOM data table \n"
        description += f"Number of mappings: {len(self.df.index)} \n"
        description += f"Number of prefixes: {len(self.prefixmap)} \n"
        description += f"Metadata: {json.dumps(self.metadata)} \n"
        description += "\nFirst rows of data: \n"
        description += self.df.head().to_string() + "\n"
        description += "\nLast rows of data: \n"
        description += self.df.tail().to_string() + "\n"
        return description

    def clean_prefix_map(self):
        prefixes_in_map = get_prefixes_used_in_table(self.df)
        new_prefixes = dict()
        missing_prefix = []
        for prefix in prefixes_in_map:
            if prefix in self.prefixmap:
                new_prefixes[prefix] = self.prefixmap[prefix]
            else:
                logging.warning(
                    f"{prefix} is used in the data frame but does not exist in prefix map"
                )
                missing_prefix.append(prefix)
        if missing_prefix:
            self.df = filter_out_prefixes(self.df, missing_prefix)
        self.prefixmap = new_prefixes


@dataclass
class EntityPair:
    """
    A tuple of entities.

    Note that (e1,e2) == (e2,e1)
    """

    subject_entity: Entity
    object_entity: Entity

    def __hash__(self):
        if self.subject_entity.id <= self.object_entity.id:
            t = self.subject_entity.id, self.object_entity.id
        else:
            t = self.object_entity.id, self.subject_entity.id
        return hash(t)


@dataclass
class MappingSetDiff:
    """
    represents a difference between two mapping sets

    Currently this is limited to diffs at the level of entity-pairs.
    For example, if file1 has A owl:equivalentClass B, and file2 has A skos:closeMatch B,
    this is considered a mapping in common.
    """

    unique_tuples1: Optional[Set[EntityPair]] = None
    unique_tuples2: Optional[Set[EntityPair]] = None
    common_tuples: Optional[Set[EntityPair]] = None

    combined_dataframe: Optional[pd.DataFrame] = None
    """
    Dataframe that combines with left and right dataframes with information injected into
    the comment column
    """


@dataclass
class MetaTSVConverter:
    """
    converts SSSOM/sssom_metadata.tsv
    DO NOT USE, NOW DEPRECATED!
    """

    df: Optional[pd.DataFrame] = None

    def load(self, filename) -> None:
        """
        loads from folder
        :return:
        """
        # self.df = pd.read_csv(filename, sep="\t", comment="#").fillna("")
        self.df = read_pandas(filename)

    def convert(self) -> Dict[str, Any]:
        if self.df is None:
            raise RuntimeError("dataframe is not loaded properly")
        # note that 'mapping' is both a metaproperty and a property of this model...
        cslots = {
            "mappings": {
                "description": "Contains a list of mapping objects",
                "range": "mapping",
                "multivalued": True,
            },
            "id": {"description": "CURIE or IRI identifier", "identifier": True},
        }
        classes: Dict[str, Any] = {
            "mapping set": {
                "description": "Represents a set of mappings",
                "slots": ["mappings"],
            },
            "mapping": {
                "description": "Represents an individual mapping between a pair of entities",
                "slots": [],
                "class_uri": "owl:Axiom",
            },
            "entity": {
                "description": "Represents any entity that can be mapped, such as an OWL class or SKOS concept",
                "mappings": ["rdf:Resource"],
                "slots": ["id"],
            },
        }
        obj = {
            "id": "http://w3id.org/sssom",
            "description": "Datamodel for Simple Standard for Sharing Ontology Mappings (SSSOM)",
            "imports": ["linkml:types"],
            "prefixes": {
                "linkml": "https://w3id.org/linkml/",
                "sssom": "http://w3id.org/sssom/",
            },
            "see_also": ["https://github.com/OBOFoundry/SSSOM"],
            "default_curi_maps": ["semweb_context"],
            "default_prefix": "sssom",
            "slots": cslots,
            "classes": classes,
        }
        for _, row in self.df.iterrows():
            eid = row["Element ID"]
            if eid == "ID":
                continue
            eid = eid.replace("sssom:", "")
            dt = row["Datatype"]
            if dt == "xsd:double":
                dt = "double"
            elif eid.endswith("_id") or eid.endswith("match_field"):
                dt = "entity"
            else:
                dt = "string"

            slot = {"description": row["Description"]}
            ep = row["Equivalent property"]
            if ep != "":
                slot["mappings"] = [ep]
            if row["Required"] == 1:
                slot["required"] = True

            slot["range"] = dt
            cslots[eid] = slot
            slot_uri = None
            if eid == "subject_id":
                slot_uri = "owl:annotatedSource"
            elif eid == "object_id":
                slot_uri = "owl:annotatedTarget"
            elif eid == "predicate_id":
                slot_uri = "owl:annotatedProperty"
            if slot_uri is not None:
                slot["slot_uri"] = slot_uri
            scope = row["Scope"]
            if "G" in scope:
                classes["mapping set"]["slots"].append(eid)
            if "L" in scope:
                classes["mapping"]["slots"].append(eid)
        return obj

    def convert_and_save(self, fn: str) -> None:
        obj = self.convert()
        with open(fn, "w") as stream:
            yaml.safe_dump(obj, stream, sort_keys=False)


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


def sort_sssom_columns(columns: list) -> list:
    # Ideally, the order of the sssom column names is parsed strictly from sssom.yaml

    logging.warning("SSSOM sort columns not implemented")
    columns.sort()
    return columns


def sort_sssom(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(
        by=sort_sssom_columns(list(df.columns)), ascending=False, inplace=True
    )
    return df


def filter_redundant_rows(df: pd.DataFrame, ignore_predicate=False) -> pd.DataFrame:
    """
    removes rows if there is another row with same S/O and higher confidence

    Args:
        df: data frame to filter
        ignore_predicate: if true, the predicate_id column is ignored
    """
    # tie-breaker
    # create a 'sort' method and then replce the following line by sort()
    df = sort_sssom(df)
    # df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: x + random.random() / 10000)
    df, nan_df = assign_default_confidence(df)
    if ignore_predicate:
        key = [SUBJECT_ID, OBJECT_ID]
    else:
        key = [SUBJECT_ID, OBJECT_ID, PREDICATE_ID]
    dfmax: pd.DataFrame
    dfmax = df.groupby(key, as_index=False)[CONFIDENCE].apply(max).drop_duplicates()
    max_conf: Dict[Any, Any] = {}
    for _, row in dfmax.iterrows():
        if ignore_predicate:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID])] = row[CONFIDENCE]
        else:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID], row[PREDICATE_ID])] = row[
                CONFIDENCE
            ]
    if ignore_predicate:
        df = df[
            df.apply(
                lambda x: x[CONFIDENCE] >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID])],
                axis=1,
            )
        ]
    else:
        df = df[
            df.apply(
                lambda x: x[CONFIDENCE]
                >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID], x[PREDICATE_ID])],
                axis=1,
            )
        ]
    # We are preserving confidence = NaN rows without making assumptions.
    # This means that there are potential duplicate mappings
    return_df = df.append(nan_df).drop_duplicates()
    return return_df


def assign_default_confidence(df: pd.DataFrame):
    # Get rows having numpy.NaN as confidence
    if df is not None and "confidence" not in df.columns:
        df["confidence"] = np.NaN

    nan_df = df[df["confidence"].isna()]
    if nan_df is None:
        nan_df = pd.DataFrame(columns=df.columns)
    return df, nan_df


def remove_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where no match is found. TODO: https://github.com/OBOFoundry/SSSOM/issues/28
    :param df:
    :return:
    """
    return df[df[PREDICATE_ID] != "noMatch"]


def create_entity(row, eid: str, mappings: Dict) -> Entity:
    logging.warning(f"create_entity() has row parameter ({row}), but not used.")
    e = Entity(id=eid)
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


def dataframe_to_ptable(df: pd.DataFrame, priors=None, inverse_factor: float = 0.5):
    """Export a KBOOM table.

    Args:
        df:
        priors:
        inverse_factor:

    Returns:
        List of rows
    """
    if not priors:
        priors = [0.02, 0.02, 0.02, 0.02]
    else:
        logging.warning(
            f"Priors given ({priors}), but not being used by dataframe_to_ptable() method."
        )

    df = collapse(df)
    rows = []
    for _, row in df.iterrows():
        subject_id = row[SUBJECT_ID]
        object_id = row[OBJECT_ID]
        confidence = row[CONFIDENCE]
        # confidence of inverse
        # e.g. if Pr(super) = 0.2, then Pr(sub) = (1-0.2) * IF
        inverse_confidence = (1.0 - confidence) * inverse_factor
        residual_confidence = (1 - (confidence + inverse_confidence)) / 2.0

        predicate = row[PREDICATE_ID]
        if predicate == "owl:equivalentClass":
            predicate_type = PREDICATE_EQUIVALENT
        elif predicate == "skos:exactMatch":
            predicate_type = PREDICATE_EQUIVALENT
        elif predicate == "skos:closeMatch":
            # TODO: consider distributing
            predicate_type = PREDICATE_EQUIVALENT
        elif predicate == "owl:subClassOf":
            predicate_type = PREDICATE_SUBCLASS
        elif predicate == "skos:broadMatch":
            predicate_type = PREDICATE_SUBCLASS
        elif predicate == "inverseOf(owl:subClassOf)":
            predicate_type = PREDICATE_SUPERCLASS
        elif predicate == "skos:narrowMatch":
            predicate_type = PREDICATE_SUPERCLASS
        elif predicate == "owl:differentFrom":
            predicate_type = PREDICATE_SIBLING
        elif predicate == "dbpedia-owl:different":
            predicate_type = PREDICATE_SIBLING
        else:
            raise ValueError(f"Unhandled predicate: {predicate}")

        if predicate_type == PREDICATE_SUBCLASS:
            ps = (
                confidence,
                inverse_confidence,
                residual_confidence,
                residual_confidence,
            )
        elif predicate_type == PREDICATE_SUPERCLASS:
            ps = (
                inverse_confidence,
                confidence,
                residual_confidence,
                residual_confidence,
            )
        elif predicate_type == PREDICATE_EQUIVALENT:
            ps = (
                residual_confidence,
                residual_confidence,
                confidence,
                inverse_confidence,
            )
        elif predicate_type == PREDICATE_SIBLING:
            ps = (
                residual_confidence,
                residual_confidence,
                inverse_confidence,
                confidence,
            )
        else:
            raise ValueError(f"predicate: {predicate_type}")
        row = [subject_id, object_id] + [str(p) for p in ps]
        rows.append(row)
    return rows


PREDICATE_SUBCLASS = 0
PREDICATE_SUPERCLASS = 1
PREDICATE_EQUIVALENT = 2
PREDICATE_SIBLING = 3

RDF_FORMATS = {"ttl", "turtle", "nt", "xml"}


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        f: FileIO
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def merge_msdf(
    msdf1: MappingSetDataFrame,
    msdf2: MappingSetDataFrame,
    reconcile: bool = True,
) -> MappingSetDataFrame:
    """
    Merging msdf2 into msdf1,
    if reconcile=True, then dedupe(remove redundant lower confidence mappings) and
        reconcile (if msdf contains a higher confidence _negative_ mapping,
        then remove lower confidence positive one. If confidence is the same,
        prefer HumanCurated. If both HumanCurated, prefer negative mapping).

    Args:
        msdf1 (MappingSetDataFrame): The primary MappingSetDataFrame
        msdf2 (MappingSetDataFrame): The secondary MappingSetDataFrame
        reconcile (bool, optional): [description]. Defaults to True.

    Returns:
        MappingSetDataFrame: Merged MappingSetDataFrame.
    """
    # Inject metadata of msdf into df
    msdf1 = inject_metadata_into_df(msdf=msdf1)
    msdf2 = inject_metadata_into_df(msdf=msdf2)

    merged_msdf = MappingSetDataFrame()
    # If msdf2 has a DataFrame
    if msdf1.df is not None and msdf2.df is not None:
        # 'outer' join in pandas == FULL JOIN in SQL
        merged_msdf.df = msdf1.df.merge(msdf2.df, how="outer")
    else:
        merged_msdf.df = msdf1.df
    # merge the non DataFrame elements
    merged_msdf.prefixmap = dict_merge(
        source=msdf2.prefixmap, target=msdf1.prefixmap, dict_name="prefixmap"
    )
    # After a Slack convo with @matentzn, commented out below.
    # merged_msdf.metadata = dict_merge(msdf2.metadata, msdf1.metadata, 'metadata')

    """if inplace:
            msdf1.prefixmap = merged_msdf.prefixmap
            msdf1.metadata = merged_msdf.metadata
            msdf1.df = merged_msdf.df"""

    if reconcile:
        merged_msdf.df = filter_redundant_rows(merged_msdf.df)
        merged_msdf.df = deal_with_negation(merged_msdf.df)  # deals with negation

    return merged_msdf


def deal_with_negation(df: pd.DataFrame) -> pd.DataFrame:
    """Combine negative and positive rows with matching [SUBJECT_ID, OBJECT_ID, CONFIDENCE] combination
    taking into account the rule that negative trumps positive given equal confidence values.

    Args:
        df (pd.DataFrame): Merged Pandas DataFrame

    Returns:
        pd.DataFrame: Pandas DataFrame with negations addressed
    """

    """
            1. Mappings in mapping1 trump mappings in mapping2 (if mapping2 contains a conflicting mapping in mapping1,
               the one in mapping1 is preserved).
            2. Reconciling means two things
                [i] if the same s,p,o (subject_id, object_id, predicate_id) is present multiple times,
                    only preserve the highest confidence one. If confidence is same, rule 1 (above) applies.
                [ii] If s,!p,o and s,p,o , then prefer higher confidence and remove the other.
                     If same confidence prefer "HumanCurated" .If same again prefer negative.
            3. Prefixes:
                [i] if there is the same prefix in mapping1 as in mapping2, and the prefix URL is different,
                throw an error and fail hard
                    else just merge the two prefix maps
            4. Metadata: same as rule 1.

            #1; #2(i) #3 and $4 are taken care of by 'filtered_merged_df' Only #2(ii) should be performed here.
        """
    # Handle DataFrames with no 'confidence' column (basically adding a np.NaN to all non-numeric confidences)
    df, nan_df = assign_default_confidence(df)

    if df is None:
        raise Exception(
            "The dataframe, after assigning default confidence, appears empty (deal_with_negation"
        )

    #  If s,!p,o and s,p,o , then prefer higher confidence and remove the other.  ###
    negation_df: pd.DataFrame
    negation_df = df.loc[
        df[PREDICATE_ID].str.startswith("!")
    ]  # or df.loc[df['predicate_modifier'] == 'NOT']

    # This step ONLY if 'NOT' is expressed by the symbol '!' in 'predicate_id' #####
    normalized_negation_df = negation_df.reset_index()
    normalized_negation_df[PREDICATE_ID] = normalized_negation_df[
        PREDICATE_ID
    ].str.replace("!", "")
    ########################################################
    normalized_negation_df = normalized_negation_df.drop(["index"], axis=1)

    # remove the NOT rows from the main DataFrame
    condition = negation_df.isin(df)
    positive_df = df.drop(condition.index)
    positive_df = positive_df.reset_index().drop(["index"], axis=1)

    columns_of_interest = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID, CONFIDENCE, MATCH_TYPE]
    negation_subset = normalized_negation_df[columns_of_interest]
    positive_subset = positive_df[columns_of_interest]

    combined_normalized_subset = pd.concat(
        [positive_subset, negation_subset]
    ).drop_duplicates()

    # GroupBy and SELECT ONLY maximum confidence
    max_confidence_df: pd.DataFrame
    max_confidence_df = combined_normalized_subset.groupby(
        KEY_FEATURES, as_index=False
    )[CONFIDENCE].max()

    # If same confidence prefer "HumanCurated".
    reconciled_df_subset = pd.DataFrame(columns=combined_normalized_subset.columns)
    for _, row_1 in max_confidence_df.iterrows():
        match_condition_1 = (
            (combined_normalized_subset[SUBJECT_ID] == row_1[SUBJECT_ID])
            & (combined_normalized_subset[OBJECT_ID] == row_1[OBJECT_ID])
            & (combined_normalized_subset[CONFIDENCE] == row_1[CONFIDENCE])
        )
        # match_condition_1[match_condition_1] gives the list of 'True's.
        # In other words, the rows that match the condition (rules declared).
        # Ideally, there should be 1 row. If not apply an extra rule to look for 'HumanCurated'.
        if len(match_condition_1[match_condition_1].index) > 1:
            match_condition_1 = (
                (combined_normalized_subset[SUBJECT_ID] == row_1[SUBJECT_ID])
                & (combined_normalized_subset[OBJECT_ID] == row_1[OBJECT_ID])
                & (combined_normalized_subset[CONFIDENCE] == row_1[CONFIDENCE])
                & (combined_normalized_subset[MATCH_TYPE] == HUMAN_CURATED_MATCH_TYPE)
            )
            # In spite of this, if match_condition_1 is returning multiple rows, pick any random row from above.
            if len(match_condition_1[match_condition_1].index) > 1:
                match_condition_1 = match_condition_1[match_condition_1].sample()

        reconciled_df_subset = reconciled_df_subset.append(
            combined_normalized_subset.loc[
                match_condition_1[match_condition_1].index, :
            ]
        )

    # Add negations (NOT symbol) back to the PREDICATE_ID
    # NOTE: negative TRUMPS positive if negative and positive with same
    # [SUBJECT_ID, OBJECT_ID, PREDICATE_ID] exist
    for _, row_2 in negation_df.iterrows():
        match_condition_2 = (
            (reconciled_df_subset[SUBJECT_ID] == row_2[SUBJECT_ID])
            & (reconciled_df_subset[OBJECT_ID] == row_2[OBJECT_ID])
            & (reconciled_df_subset[CONFIDENCE] == row_2[CONFIDENCE])
        )
        reconciled_df_subset.loc[
            match_condition_2[match_condition_2].index, PREDICATE_ID
        ] = row_2[PREDICATE_ID]

    reconciled_df = pd.DataFrame(columns=df.columns)
    for _, row_3 in reconciled_df_subset.iterrows():
        match_condition_3 = (
            (df[SUBJECT_ID] == row_3[SUBJECT_ID])
            & (df[OBJECT_ID] == row_3[OBJECT_ID])
            & (df[CONFIDENCE] == row_3[CONFIDENCE])
            & (df[PREDICATE_ID] == row_3[PREDICATE_ID])
        )
        reconciled_df = reconciled_df.append(
            df.loc[match_condition_3[match_condition_3].index, :]
        )
    return_df = reconciled_df.append(nan_df).drop_duplicates()
    return return_df


def dict_merge(
    *,
    source: Optional[Dict[str, Any]] = None,
    target: Dict[str, Any],
    dict_name: str,
) -> Dict[str, Any]:
    """
    Takes 2 MappingSetDataFrame elements (prefixmap OR metadata) and merges source => target

    Args:
        source: MappingSetDataFrame.prefixmap / MappingSetDataFrame.metadata
        target: MappingSetDataFrame.prefixmap / MappingSetDataFrame.metadata
        dict_name: prefixmap or metadata

    Returns:
        Dict: merged MappingSetDataFrame.prefixmap / MappingSetDataFrame.metadata
    """
    if source is not None:
        for k, v in source.items():
            if k not in target:
                if v not in list(target.values()):
                    target[k] = v
                else:
                    common_values = [i for i, val in target.items() if val == v]
                    raise ValueError(
                        f"Value [{v}] is present in {dict_name} for multiple keys [{common_values}]."
                    )
            else:
                if target[k] != v:
                    raise ValueError(
                        f"{dict_name} values in both MappingSetDataFrames for the same key [{k}] are different."
                    )
    return target


def inject_metadata_into_df(msdf: MappingSetDataFrame) -> MappingSetDataFrame:
    """Inject metadata dictionary key-value pair into DataFrame columns in a MappingSetDataFrame.DataFrame.

    Args:
        msdf (MappingSetDataFrame): MappingSetDataFrame with metadata separate.

    Returns:
        MappingSetDataFrame: MappingSetDataFrame with metadata as columns
    """
    if msdf.metadata is not None and msdf.df is not None:
        for k, v in msdf.metadata.items():
            if k not in msdf.df.columns:
                msdf.df[k] = v
    return msdf


def get_file_extension(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) > 0:
        f_format = parts[-1]
        return f_format
    else:
        raise Exception(f"Cannot guess format from {filename}")


def read_csv(filename, comment="#", sep=","):
    if validators.url(filename):
        response = urlopen(filename)
        lines = "".join(
            [
                line.decode("utf-8")
                for line in response
                if not line.decode("utf-8").startswith(comment)
            ]
        )
    else:
        with open(filename, "r") as f:
            lines = "".join([line for line in f if not line.startswith(comment)])
    return pd.read_csv(StringIO(lines), sep=sep)


def read_metadata(filename):
    """
    Read a metadata file (yaml) that is supplied separately from a TSV.

    :param filename: location of file
    :return: two objects, a metadata and a curie_map object
    """
    meta = {}
    curie_map = {}
    with open(filename, "r") as stream:
        try:
            m = yaml.safe_load(stream)
            if "curie_map" in m:
                curie_map = m["curie_map"]
            m.pop("curie_map", None)
            meta = m
        except yaml.YAMLError as exc:
            print(exc)  # FIXME this clobbers the exception. Remove try/except
    return meta, curie_map


def read_pandas(filename: str, sep: Optional[str] = "\t") -> pd.DataFrame:
    """
    Read a tabular data file by wrapping func:`pd.read_csv` to handles comment lines correctly.

    :param filename:
    :param sep: File separator in pandas (\t or ,)
    :return:
    """
    if not sep:
        extension = get_file_extension(filename)
        sep = "\t"
        if extension == "tsv":
            sep = "\t"
        elif extension == "csv":
            sep = ","
        else:
            logging.warning("Cannot automatically determine table format, trying tsv.")

    # from tempfile import NamedTemporaryFile
    # with NamedTemporaryFile("r+") as tmp:
    #    with open(filename, "r") as f:
    #        for line in f:
    #            if not line.startswith('#'):
    #                tmp.write(line + "\n")
    #    tmp.seek(0)
    return read_csv(filename, comment="#", sep=sep).fillna("")


def extract_global_metadata(msdoc: MappingSetDocument):
    meta = {"curie_map": msdoc.curie_map}
    ms_meta = msdoc.mapping_set
    for key in [
        slot
        for slot in dir(slots)
        if not callable(getattr(slots, slot)) and not slot.startswith("__")
    ]:
        slot = getattr(slots, key).name
        if slot not in ["mappings"] and slot in ms_meta:
            if ms_meta[slot]:
                meta[key] = ms_meta[slot]
    return meta


def to_mapping_set_dataframe(doc: MappingSetDocument) -> MappingSetDataFrame:
    ###
    # convert MappingSetDocument into MappingSetDataFrame
    ###
    data = []
    if doc.mapping_set.mappings is not None:
        for mapping in doc.mapping_set.mappings:
            mdict = mapping.__dict__
            m = {}
            for key in mdict:
                if mdict[key]:
                    m[key] = mdict[key]
            data.append(m)
    df = pd.DataFrame(data=data)
    meta = extract_global_metadata(doc)
    meta.pop("curie_map", None)
    msdf = MappingSetDataFrame(df=df, prefixmap=doc.curie_map, metadata=meta)
    return msdf


# to_mapping_set_document is in parser.py in order to avoid circular import errors


class NoCURIEException(ValueError):
    pass


CURIE_RE = re.compile(r"[A-Za-z0-9_]+[:][A-Za-z0-9_]")


def is_curie(string: str) -> bool:
    return bool(CURIE_RE.match(string))


def get_prefix_from_curie(curie: str) -> str:
    if is_curie(curie):
        return curie.split(":")[0]
    else:
        return ""


def curie_from_uri(uri: str, curie_map: Mapping[str, str]):
    if is_curie(uri):
        return uri
    for prefix in curie_map:
        uri_prefix = curie_map[prefix]
        if uri.startswith(uri_prefix):
            remainder = uri.replace(uri_prefix, "")
            return f"{prefix}:{remainder}"
    raise NoCURIEException(f"{uri} does not follow any known prefixes")


def get_prefixes_used_in_table(df: pd.DataFrame):
    prefixes = []
    for col in KEY_FEATURES:
        for v in df[col].values:
            prefixes.append(get_prefix_from_curie(v))
    return list(set(prefixes))


def filter_out_prefixes(df: pd.DataFrame, filter_prefixes) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        # Get list of CURIEs from the 3 columns (KEY_FEATURES) for the row.
        prefixes = {get_prefix_from_curie(curie) for curie in row[KEY_FEATURES]}
        # Confirm if none of the 3 CURIEs in the list above appear in the filter_prefixes list.
        # If TRUE, append row.
        if not any(prefix in prefixes for prefix in filter_prefixes):
            rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=KEY_FEATURES)


def guess_file_format(filename):
    extension = get_file_extension(filename)
    if extension in ["owl", "rdf"]:
        return SSSOM_DEFAULT_RDF_SERIALISATION
    elif extension in RDF_FORMATS:
        return extension
    else:
        raise Exception(
            f"File extension {extension} does not correspond to a legal file format"
        )


def prepare_context_from_curie_map(curie_map: dict):
    meta, default_curie_map = get_default_metadata()
    context = get_jsonld_context()
    if not curie_map:
        curie_map = default_curie_map

    for k, v in curie_map.items():
        if isinstance(v, str):
            if k not in context["@context"]:
                context["@context"][k] = v
            else:
                if context["@context"][k] != v:
                    logging.info(
                        f"{k} namespace is already in the context, ({context['@context'][k]}, "
                        f"but with a different value than {v}. Overwriting!"
                    )
                    context["@context"][k] = v
    return json.dumps(context)


def raise_for_bad_path(file_path: str) -> None:
    if not validators.url(file_path) and not os.path.exists(file_path):
        raise Exception(f"{file_path} is not a valid file path or url.")
