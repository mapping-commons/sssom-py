"""Utility functions."""
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from io import StringIO
from pathlib import Path
from string import punctuation
from typing import (
    Any,
    ChainMap,
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
)
from urllib.request import urlopen

import deprecation
import numpy as np
import pandas as pd
import validators
import yaml
from jsonschema import ValidationError
from linkml_runtime.linkml_model.types import Uriorcurie
from pandas.errors import EmptyDataError

# from .sssom_datamodel import Mapping as SSSOM_Mapping
# from .sssom_datamodel import slots
from sssom_schema import Mapping as SSSOM_Mapping
from sssom_schema import slots

from .constants import (
    COLUMN_INVERT_DICTIONARY,
    COMMENT,
    CONFIDENCE,
    MAPPING_JUSTIFICATION,
    MAPPING_SET_ID,
    MAPPING_SET_SOURCE,
    OBJECT_CATEGORY,
    OBJECT_ID,
    OBJECT_LABEL,
    OBJECT_SOURCE,
    OBO_HAS_DB_XREF,
    OWL_DIFFERENT_FROM,
    OWL_EQUIVALENT_CLASS,
    PREDICATE_ID,
    PREDICATE_INVERT_DICTIONARY,
    PREDICATE_LIST,
    PREDICATE_MODIFIER,
    PREDICATE_MODIFIER_NOT,
    PREFIX_MAP_MODES,
    RDFS_SUBCLASS_OF,
    SCHEMA_YAML,
    SEMAPV,
    SKOS_BROAD_MATCH,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_NARROW_MATCH,
    SKOS_RELATED_MATCH,
    SSSOM_SUPERCLASS_OF,
    SUBJECT_CATEGORY,
    SUBJECT_ID,
    SUBJECT_LABEL,
    SUBJECT_SOURCE,
    UNKNOWN_IRI,
    SSSOMSchemaView,
)
from .context import (
    SSSOM_BUILT_IN_PREFIXES,
    SSSOM_URI_PREFIX,
    get_default_metadata,
    get_jsonld_context,
)
from .sssom_document import MappingSetDocument
from .typehints import Metadata, MetadataType, PrefixMap

#: The key that's used in the YAML section of an SSSOM file
PREFIX_MAP_KEY = "curie_map"

SSSOM_READ_FORMATS = [
    "tsv",
    "rdf",
    "owl",
    "alignment-api-xml",
    "obographs-json",
    "json",
]
SSSOM_EXPORT_FORMATS = ["tsv", "rdf", "owl", "json", "fhir", "ontoportal_json"]

SSSOM_DEFAULT_RDF_SERIALISATION = "turtle"

URI_SSSOM_MAPPINGS = f"{SSSOM_URI_PREFIX}mappings"

#: The 4 columns whose combination would be used as primary keys while merging/grouping
KEY_FEATURES = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID, PREDICATE_MODIFIER]
TRIPLES_IDS = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID]


@dataclass
class MappingSetDataFrame:
    """A collection of mappings represented as a DataFrame, together with additional metadata."""

    df: Optional[pd.DataFrame] = None  # Mappings
    # maps CURIE prefixes to URI bases
    prefix_map: PrefixMap = field(default_factory=dict)
    metadata: Optional[MetadataType] = None  # header metadata excluding prefixes

    def merge(self, *msdfs: "MappingSetDataFrame", inplace: bool = True) -> "MappingSetDataFrame":
        """Merge two MappingSetDataframes.

        :param msdfs: Multiple/Single MappingSetDataFrame(s) to merge with self
        :param inplace: If true, msdf2 is merged into the calling MappingSetDataFrame,
                        if false, it simply return the merged data frame.
        :return: Merged MappingSetDataFrame
        """
        msdf = merge_msdf(self, *msdfs)
        if inplace:
            self.df = msdf.df
            self.prefix_map = msdf.prefix_map
            self.metadata = msdf.metadata
            return self
        else:
            return msdf

    def __str__(self) -> str:  # noqa:D105
        description = "SSSOM data table \n"
        description += f"Number of prefixes: {len(self.prefix_map)} \n"
        if self.metadata is None:
            description += "No metadata available \n"
        else:
            description += f"Metadata: {json.dumps(self.metadata)} \n"
        if self.df is None:
            description += "No dataframe available"
        else:
            description += f"Number of mappings: {len(self.df.index)} \n"
            description += "\nFirst rows of data: \n"
            description += self.df.head().to_string() + "\n"
            description += "\nLast rows of data: \n"
            description += self.df.tail().to_string() + "\n"
        return description

    def clean_prefix_map(self, strict: bool = True) -> None:
        """
        Remove unused prefixes from the internal prefix map based on the internal dataframe.

        :param strict: Boolean if True, errors out if all prefixes in dataframe are not
                       listed in the 'curie_map'.
        :raises ValueError: If prefixes absent in 'curie_map' and strict flag = True
        """
        all_prefixes = []
        prefixes_in_table = get_prefixes_used_in_table(self.df)
        if self.metadata:
            prefixes_in_metadata = get_prefixes_used_in_metadata(self.metadata)
            all_prefixes = list(set(prefixes_in_table + prefixes_in_metadata))
        else:
            all_prefixes = prefixes_in_table

        new_prefixes: PrefixMap = dict()
        missing_prefixes = []
        default_prefix_map = get_default_metadata().prefix_map
        for prefix in all_prefixes:
            if prefix in self.prefix_map:
                new_prefixes[prefix] = self.prefix_map[prefix]
            elif prefix in default_prefix_map:
                new_prefixes[prefix] = default_prefix_map[prefix]
            else:
                logging.warning(
                    f"{prefix} is used in the SSSOM mapping set but it does not exist in the prefix map"
                )
                if prefix != "":
                    missing_prefixes.append(prefix)
                    if not strict:
                        new_prefixes[prefix] = UNKNOWN_IRI + prefix.lower() + "/"

        if missing_prefixes and strict:
            raise ValueError(
                f"{missing_prefixes} are used in the SSSOM mapping set but it does not exist in the prefix map"
            )
            # self.df = filter_out_prefixes(self.df, missing_prefixes)
        self.prefix_map = new_prefixes

    def remove_mappings(self, msdf: "MappingSetDataFrame"):
        """Remove mappings in right msdf from left msdf.

        :param msdf: MappingSetDataframe object to be removed from primary msdf object.
        """
        self.df = (
            pd.merge(
                self.df,
                msdf.df,
                on=KEY_FEATURES,
                how="outer",
                suffixes=("", "_2"),
                indicator=True,
            )
            .query("_merge == 'left_only'")
            .drop("_merge", axis=1)
            .reset_index(drop=True)
        )

        self.df = self.df[self.df.columns.drop(list(self.df.filter(regex=r"_2")))]
        self.clean_prefix_map()


@dataclass
class EntityPair:
    """
    A tuple of entities.

    Note that (e1,e2) == (e2,e1)
    """

    subject_entity: Uriorcurie
    object_entity: Uriorcurie

    def __hash__(self) -> int:  # noqa:D105
        if self.subject_entity <= self.object_entity:
            t = self.subject_entity, self.object_entity
        else:
            t = self.object_entity, self.subject_entity
        return hash(t)


@dataclass
class MappingSetDiff:
    """
    Represents a difference between two mapping sets.

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


def parse(filename: str) -> pd.DataFrame:
    """Parse a TSV to a pandas frame."""
    # return from_tsv(filename)
    logging.info(f"Parsing {filename}")
    return pd.read_csv(filename, sep="\t", comment="#")
    # return read_pandas(filename)


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows with same S/P/O and combines confidence."""
    df2 = df.groupby([SUBJECT_ID, PREDICATE_ID, OBJECT_ID])[CONFIDENCE].apply(max).reset_index()
    return df2


def sort_sssom(df: pd.DataFrame) -> pd.DataFrame:
    """Sort SSSOM by columns.

    :param df: SSSOM DataFrame to be sorted.
    :return: Sorted SSSOM DataFrame
    """
    df.sort_values(by=sorted(df.columns), ascending=False, inplace=True)
    return df


def filter_redundant_rows(df: pd.DataFrame, ignore_predicate: bool = False) -> pd.DataFrame:
    """Remove rows if there is another row with same S/O and higher confidence.

    :param df: Pandas DataFrame to filter
    :param ignore_predicate: If true, the predicate_id column is ignored, defaults to False
    :return: Filtered pandas DataFrame
    """
    # tie-breaker
    # create a 'sort' method and then replce the following line by sort()
    df = sort_sssom(df)
    # df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: x + random.random() / 10000)
    confidence_in_original = CONFIDENCE in df.columns
    df, nan_df = assign_default_confidence(df)
    if ignore_predicate:
        key = [SUBJECT_ID, OBJECT_ID]
    else:
        key = [SUBJECT_ID, OBJECT_ID, PREDICATE_ID]
    dfmax: pd.DataFrame
    dfmax = df.groupby(key, as_index=False)[CONFIDENCE].apply(max).drop_duplicates()
    max_conf: Dict[Tuple[str, ...], float] = {}
    for _, row in dfmax.iterrows():
        if ignore_predicate:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID])] = row[CONFIDENCE]
        else:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID], row[PREDICATE_ID])] = row[CONFIDENCE]
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
                lambda x: x[CONFIDENCE] >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID], x[PREDICATE_ID])],
                axis=1,
            )
        ]
    # We are preserving confidence = NaN rows without making assumptions.
    # This means that there are potential duplicate mappings
    # FutureWarning: The frame.append method is deprecated and
    # will be removed from pandas in a future version.
    # Use pandas.concat instead.
    # return_df = df.append(nan_df).drop_duplicates()
    confidence_reconciled_df = pd.concat([df, nan_df]).drop_duplicates()

    # Reconciling dataframe rows based on the predicates with equal confidence.
    if PREDICATE_MODIFIER in confidence_reconciled_df.columns:
        tmp_df = confidence_reconciled_df[
            [SUBJECT_ID, OBJECT_ID, PREDICATE_ID, CONFIDENCE, PREDICATE_MODIFIER]
        ]
        tmp_df = tmp_df[tmp_df[PREDICATE_MODIFIER] != PREDICATE_MODIFIER_NOT].drop(
            PREDICATE_MODIFIER, axis=1
        )
    else:
        tmp_df = confidence_reconciled_df[[SUBJECT_ID, OBJECT_ID, PREDICATE_ID, CONFIDENCE]]
    tmp_df_grp = tmp_df.groupby([SUBJECT_ID, OBJECT_ID, CONFIDENCE], as_index=False).count()
    tmp_df_grp = tmp_df_grp[tmp_df_grp[PREDICATE_ID] > 1].drop(PREDICATE_ID, axis=1)
    non_predicate_reconciled_df = (
        confidence_reconciled_df.merge(
            tmp_df_grp, on=list(tmp_df_grp.columns), how="left", indicator=True
        )
        .query('_merge == "left_only"')
        .drop(columns="_merge")
    )

    multiple_predicate_df = (
        confidence_reconciled_df.merge(
            tmp_df_grp, on=list(tmp_df_grp.columns), how="right", indicator=True
        )
        .query('_merge == "both"')
        .drop(columns="_merge")
    )

    return_df = non_predicate_reconciled_df
    for _, row in tmp_df_grp.iterrows():
        logic_df = multiple_predicate_df[list(tmp_df_grp.columns)] == row
        concerned_row_index = logic_df[logic_df[list(tmp_df_grp.columns)]].dropna().index
        concerned_df = multiple_predicate_df.iloc[concerned_row_index]
        # Go down the hierarchical list of PREDICATE_LIST and grab the first match
        return_df = pd.concat(
            [get_row_based_on_hierarchy(concerned_df), return_df], axis=0
        ).drop_duplicates()

    if not confidence_in_original:
        return_df = return_df.drop(columns=[CONFIDENCE], axis=1)
    return return_df


def get_row_based_on_hierarchy(df: pd.DataFrame):
    """Get row based on hierarchy of predicates.

    The hierarchy is as follows:
    # owl:equivalentClass
    # owl:equivalentProperty
    # rdfs:subClassOf
    # rdfs:subPropertyOf
    # owl:sameAs
    # skos:exactMatch
    # skos:closeMatch
    # skos:broadMatch
    # skos:narrowMatch
    # oboInOwl:hasDbXref
    # skos:relatedMatch
    # rdfs:seeAlso

    :param df: Dataframe containing multiple predicates for same subject and object.
    :return: Dataframe with a single row which ranks higher in the hierarchy.
    """
    for pred in PREDICATE_LIST:
        hierarchical_df = df[df[PREDICATE_ID] == pred]
        if not hierarchical_df.empty:
            return hierarchical_df


def assign_default_confidence(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign :data:`numpy.nan` to confidence that are blank.

    :param df: SSSOM DataFrame
    :return: A Tuple consisting of the original DataFrame and dataframe consisting of empty confidence values.
    """
    # Get rows having numpy.NaN as confidence
    if df is not None:
        new_df = df.copy()
        if CONFIDENCE not in new_df.columns:
            new_df[CONFIDENCE] = 0.0  # np.NaN
            nan_df = pd.DataFrame(columns=new_df.columns)
        else:
            new_df = df[~df[CONFIDENCE].isna()]
            nan_df = df[df[CONFIDENCE].isna()]
    else:
        ValueError("DataFrame cannot be empty to 'assign_default_confidence'.")
    return new_df, nan_df


def remove_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where no match is found.

    TODO: https://github.com/OBOFoundry/SSSOM/issues/28
    :param df: Pandas DataFrame
    :return: Pandas DataFrame with 'PREDICATE_ID' not 'noMatch'.
    """
    return df[df[PREDICATE_ID] != "noMatch"]


def create_entity(identifier: str, mappings: Dict[str, Any]) -> Uriorcurie:
    """
    Create an Entity object.

    :param identifier: Entity Id
    :param mappings: Mapping dictionary
    :return: An Entity object
    """
    entity = Uriorcurie(identifier)  # Entity(id=identifier)
    for key, value in mappings.items():
        if key in entity:
            entity[key] = value
    return entity


def group_mappings(df: pd.DataFrame) -> Dict[EntityPair, List[pd.Series]]:
    """Group mappings by EntityPairs."""
    mappings: DefaultDict[EntityPair, List[pd.Series]] = defaultdict(list)
    for _, row in df.iterrows():
        subject_entity = create_entity(
            identifier=row[SUBJECT_ID],
            mappings={
                "label": SUBJECT_LABEL,
                "category": SUBJECT_CATEGORY,
                "source": SUBJECT_SOURCE,
            },
        )
        object_entity = create_entity(
            identifier=row[OBJECT_ID],
            mappings={
                "label": OBJECT_LABEL,
                "category": OBJECT_CATEGORY,
                "source": OBJECT_SOURCE,
            },
        )
        mappings[EntityPair(subject_entity, object_entity)].append(row)
    return dict(mappings)


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> MappingSetDiff:
    """Perform a diff between two SSSOM dataframes.

    :param df1: A mapping dataframe
    :param df2: A mapping dataframe
    :returns: A mapping set diff

    .. warning:: currently does not discriminate between mappings with different predicates
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
        all_ids.update({t.subject_entity, t.object_entity})
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
    d.combined_dataframe = pd.DataFrame(rows).drop_duplicates()
    return d


def add_default_confidence(df: pd.DataFrame, confidence: float = None) -> pd.DataFrame:
    """Add `confidence` column to DataFrame if absent and initializes to 0.95.

    If `confidence` column already exists, only fill in the None ones by 0.95.

    :param df: DataFrame whose `confidence` column needs to be filled.
    :return: DataFrame with a complete `confidence` column.
    """
    # df[CONFIDENCE] = df.get(CONFIDENCE, confidence)
    if df.get(CONFIDENCE) is not None:
        df[CONFIDENCE] = confidence * df[CONFIDENCE]
    df.loc[df[CONFIDENCE].isnull(), CONFIDENCE] = confidence
    return df


def dataframe_to_ptable(
    df: pd.DataFrame, *, inverse_factor: float = None, default_confidence: float = None
):
    """Export a KBOOM table.

    :param df: Pandas DataFrame
    :param inverse_factor: Multiplier to (1 - confidence), defaults to 0.5
    :param default_confidence: Default confidence to be assigned if absent.
    :raises ValueError: Predicate value error
    :raises ValueError: Predicate type value error
    :return: List of rows
    """
    if not inverse_factor:
        inverse_factor = 0.5

    if default_confidence:
        df = add_default_confidence(df, default_confidence)

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
        if predicate == OWL_EQUIVALENT_CLASS:
            predicate_type = PREDICATE_EQUIVALENT
        elif predicate == SKOS_EXACT_MATCH:
            predicate_type = PREDICATE_EQUIVALENT
        elif predicate == SKOS_CLOSE_MATCH:
            # TODO: consider distributing
            predicate_type = PREDICATE_EQUIVALENT
        elif predicate == RDFS_SUBCLASS_OF:
            predicate_type = PREDICATE_SUBCLASS
        elif predicate == SKOS_BROAD_MATCH:
            predicate_type = PREDICATE_SUBCLASS
        elif predicate == SSSOM_SUPERCLASS_OF:
            predicate_type = PREDICATE_SUPERCLASS
        elif predicate == SKOS_NARROW_MATCH:
            predicate_type = PREDICATE_SUPERCLASS
        elif predicate == OWL_DIFFERENT_FROM:
            predicate_type = PREDICATE_SIBLING
        # * Added by H2 ############################
        elif predicate == OBO_HAS_DB_XREF:
            predicate_type = PREDICATE_HAS_DBXREF
        elif predicate == SKOS_RELATED_MATCH:
            predicate_type = PREDICATE_RELATED_MATCH
        # * ########################################
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
        # * Added by H2 ############################
        elif predicate_type == PREDICATE_HAS_DBXREF:
            ps = (
                residual_confidence,
                residual_confidence,
                confidence,
                inverse_confidence,
            )
        elif predicate_type == PREDICATE_RELATED_MATCH:
            ps = (
                residual_confidence,
                residual_confidence,
                confidence,
                inverse_confidence,
            )
        # * #########################################
        else:
            raise ValueError(f"predicate: {predicate_type}")
        row = [subject_id, object_id] + [str(p) for p in ps]
        rows.append(row)
    return rows


PREDICATE_SUBCLASS = 0
PREDICATE_SUPERCLASS = 1
PREDICATE_EQUIVALENT = 2
PREDICATE_SIBLING = 3
# * Added by H2 ############################
PREDICATE_HAS_DBXREF = 4
PREDICATE_RELATED_MATCH = 5
# * ########################################

RDF_FORMATS = {"ttl", "turtle", "nt", "xml"}


def sha256sum(path: str) -> str:
    """Calculate the SHA256 hash over the bytes in a file.

    :param path: Filename
    :return: Hashed value
    """
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(path, "rb", buffering=0) as file:
        for n in iter(lambda: file.readinto(mv), 0):  # type: ignore
            h.update(mv[:n])
    return h.hexdigest()


def merge_msdf(
    *msdfs: MappingSetDataFrame,
    reconcile: bool = False,
) -> MappingSetDataFrame:
    """Merge multiple MappingSetDataFrames into one.

    :param msdfs: A Tuple of MappingSetDataFrames to be merged
    :param reconcile: If reconcile=True, then dedupe(remove redundant lower confidence mappings)
        and reconcile (if msdf contains a higher confidence _negative_ mapping,
        then remove lower confidence positive one. If confidence is the same,
        prefer HumanCurated. If both HumanCurated, prefer negative mapping).
        Defaults to True.
    :returns: Merged MappingSetDataFrame.
    """
    merged_msdf = MappingSetDataFrame()

    # Inject metadata of msdf into df
    msdf_with_meta = [inject_metadata_into_df(msdf) for msdf in msdfs]

    # merge df [# 'outer' join in pandas == FULL JOIN in SQL]
    # df_merged = reduce(
    #     lambda left, right: left.merge(right, how="outer", on=list(left.columns)),
    #     [msdf.df for msdf in msdf_with_meta if msdf.df is not None],
    # )
    # Concat is an alternative to merge when columns are not the same.
    df_merged = reduce(
        lambda left, right: pd.concat([left, right], axis=0, ignore_index=True),
        [msdf.df for msdf in msdf_with_meta if msdf.df is not None],
    ).drop_duplicates(ignore_index=True)

    # merge the non DataFrame elements
    prefix_map_list = [msdf.prefix_map for msdf in msdf_with_meta]
    # prefix_map_merged = {k: v for d in prefix_map_list for k, v in d.items()}
    merged_msdf.prefix_map = dict(ChainMap(*prefix_map_list))
    merged_msdf.df = df_merged
    if reconcile:
        merged_msdf.df = filter_redundant_rows(merged_msdf.df)
        if (
            PREDICATE_MODIFIER in merged_msdf.df.columns
            and PREDICATE_MODIFIER_NOT in merged_msdf.df[PREDICATE_MODIFIER]
        ):
            merged_msdf.df = deal_with_negation(merged_msdf.df)  # deals with negation

    # TODO: Add default values for license and mapping_set_id.
    return merged_msdf


def deal_with_negation(df: pd.DataFrame) -> pd.DataFrame:
    """Combine negative and positive rows with matching [SUBJECT_ID, OBJECT_ID, CONFIDENCE] combination.

    Rule: negative trumps positive if modulus of confidence values are equal.

    :param df: Merged Pandas DataFrame
    :return: Pandas DataFrame with negations addressed
    :raises ValueError: If the dataframe is none after assigning default confidence
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
    confidence_in_original = CONFIDENCE in df.columns
    df, nan_df = assign_default_confidence(df)
    if df is None:
        raise ValueError(
            "The dataframe, after assigning default confidence, appears empty (deal_with_negation)"
        )

    #  If s,!p,o and s,p,o , then prefer higher confidence and remove the other.  ###
    negation_df: pd.DataFrame
    negation_df = df.loc[df[PREDICATE_MODIFIER] == PREDICATE_MODIFIER_NOT]
    normalized_negation_df = negation_df.reset_index()

    # This step ONLY if 'NOT' is expressed by the symbol '!' in 'predicate_id' #####
    # normalized_negation_df[PREDICATE_ID] = normalized_negation_df[
    #     PREDICATE_ID
    # ].str.replace("!", "")
    ########################################################
    normalized_negation_df = normalized_negation_df.drop(["index"], axis=1)

    # remove the NOT rows from the main DataFrame
    condition = negation_df.isin(df)
    positive_df = df.drop(condition.index)
    positive_df = positive_df.reset_index().drop(["index"], axis=1)

    columns_of_interest = [
        SUBJECT_ID,
        PREDICATE_ID,
        OBJECT_ID,
        CONFIDENCE,
        MAPPING_JUSTIFICATION,
    ]
    negation_subset = normalized_negation_df[columns_of_interest]
    positive_subset = positive_df[columns_of_interest]

    combined_normalized_subset = pd.concat([positive_subset, negation_subset]).drop_duplicates()

    # GroupBy and SELECT ONLY maximum confidence
    max_confidence_df: pd.DataFrame
    max_confidence_df = combined_normalized_subset.groupby(TRIPLES_IDS, as_index=False)[
        CONFIDENCE
    ].max()

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
                & (
                    combined_normalized_subset[MAPPING_JUSTIFICATION]
                    == SEMAPV.ManualMappingCuration.value
                )
            )
            # In spite of this, if match_condition_1
            # is returning multiple rows, pick any random row from above.
            if len(match_condition_1[match_condition_1].index) > 1:
                match_condition_1 = match_condition_1[match_condition_1].sample()

        # FutureWarning: The frame.append method is deprecated and will be removed
        # from pandas in a future version. Use pandas.concat instead.
        # reconciled_df_subset = reconciled_df_subset.append(
        #     combined_normalized_subset.loc[
        #         match_condition_1[match_condition_1].index, :
        #     ],
        #     ignore_index=True,
        # )
        reconciled_df_subset = pd.concat(
            [
                reconciled_df_subset,
                combined_normalized_subset.loc[match_condition_1[match_condition_1].index, :],
            ],
            ignore_index=True,
        )

    # Add negations (PREDICATE_MODIFIER) back to DataFrame
    # NOTE: negative TRUMPS positive if negative and positive with same
    # [SUBJECT_ID, OBJECT_ID, PREDICATE_ID] exist
    for _, row_2 in negation_df.iterrows():
        match_condition_2 = (
            (reconciled_df_subset[SUBJECT_ID] == row_2[SUBJECT_ID])
            & (reconciled_df_subset[OBJECT_ID] == row_2[OBJECT_ID])
            & (reconciled_df_subset[CONFIDENCE] == row_2[CONFIDENCE])
        )

        reconciled_df_subset.loc[
            match_condition_2[match_condition_2].index, PREDICATE_MODIFIER
        ] = row_2[PREDICATE_MODIFIER]

    if PREDICATE_MODIFIER in reconciled_df_subset.columns:
        reconciled_df_subset[PREDICATE_MODIFIER] = reconciled_df_subset[PREDICATE_MODIFIER].fillna(
            ""
        )

    # .fillna(df) towards the end fills an empty value
    # with a corresponding value from df.
    # This needs to happen because the columns in df
    # not in reconciled_df_subset will be NaN otherwise
    # which is incorrect.
    reconciled_df = df.merge(
        reconciled_df_subset, how="right", on=list(reconciled_df_subset.columns)
    ).fillna(df)

    if nan_df.empty:
        return_df = reconciled_df
    else:
        return_df = reconciled_df.append(nan_df).drop_duplicates()

    if not confidence_in_original:
        return_df = return_df.drop(columns=[CONFIDENCE], axis=1)

    return return_df


def inject_metadata_into_df(msdf: MappingSetDataFrame) -> MappingSetDataFrame:
    """Inject metadata dictionary key-value pair into DataFrame columns in a MappingSetDataFrame.DataFrame.

    :param msdf: MappingSetDataFrame with metadata separate.

    :return: MappingSetDataFrame with metadata as columns
    """
    # TODO Check if 'k' is a valid 'slot' for 'mapping' [sssom.yaml]
    with open(SCHEMA_YAML) as file:
        schema = yaml.safe_load(file)
    slots = schema["classes"]["mapping"]["slots"]
    if msdf.metadata is not None and msdf.df is not None:
        for k, v in msdf.metadata.items():
            if k not in msdf.df.columns and k in slots:
                if k == MAPPING_SET_ID:
                    k = MAPPING_SET_SOURCE
                if isinstance(v, list):
                    v = "|".join(x for x in v)
                msdf.df[k] = str(v)
    return msdf


def get_file_extension(file: Union[str, Path, TextIO]) -> str:
    """Get file extension.

    :param file: File path
    :return: format of the file passed, default tsv
    """
    if isinstance(file, Path):
        if file.suffix:
            return file.suffix.strip(punctuation)
        else:
            logging.warning(
                f"Cannot guess format from {file}, despite appearing to be a Path-like object."
            )
    elif isinstance(file, str):
        filename = file
        parts = filename.split(".")
        if len(parts) > 0:
            f_format = parts[-1]
            return f_format.strip(punctuation)
        else:
            logging.warning(f"Cannot guess format from {filename}")
    logging.info("Cannot guess format extension for this file, assuming TSV.")
    return "tsv"


@deprecation.deprecated(details="Use pandas.read_csv() instead.")
def read_csv(
    filename: Union[str, Path, TextIO], comment: str = "#", sep: str = ","
) -> pd.DataFrame:
    """Read a CSV that contains frontmatter commented by a specific character.

    :param filename: Either the file path, a URL, or a file object to read as a TSV
        with frontmatter
    :param comment: The comment character used for the frontmatter. This isn't the
        same as the comment keyword in :func:`pandas.read_csv` because it only
        works on the first charcter in the line
    :param sep: The separator for the TSV file
    :returns: A pandas dataframe
    """
    if isinstance(filename, TextIO):
        return pd.read_csv(filename, sep=sep)
    if isinstance(filename, Path) or not validators.url(filename):
        with open(filename, "r") as f:
            lines = "".join([line for line in f if not line.startswith(comment)])
    else:
        response = urlopen(filename)
        lines = "".join(
            [
                line.decode("utf-8")
                for line in response
                if not line.decode("utf-8").startswith(comment)
            ]
        )
    try:
        df = pd.read_csv(StringIO(lines), sep=sep, low_memory=False)
    except EmptyDataError as e:
        logging.warning(f"Seems like the dataframe is empty: {e}")
        df = pd.DataFrame(
            columns=[
                SUBJECT_ID,
                SUBJECT_LABEL,
                PREDICATE_ID,
                OBJECT_ID,
                MAPPING_JUSTIFICATION,
            ]
        )

    return df


def read_metadata(filename: str) -> Metadata:
    """Read a metadata file (yaml) that is supplied separately from a TSV."""
    prefix_map = {}
    with open(filename) as file:
        metadata = yaml.safe_load(file)
    if PREFIX_MAP_KEY in metadata:
        prefix_map = metadata.pop(PREFIX_MAP_KEY)
    return Metadata(prefix_map=prefix_map, metadata=metadata)


@deprecation.deprecated(details="Use pandas.read_csv() instead.")
def read_pandas(file: Union[str, Path, TextIO], sep: Optional[str] = None) -> pd.DataFrame:
    """Read a tabular data file by wrapping func:`pd.read_csv` to handles comment lines correctly.

    :param file: The file to read. If no separator is given, this file should be named.
    :param sep: File separator for pandas
    :return: A pandas dataframe
    """
    if sep is None:
        if isinstance(file, Path) or isinstance(file, str):
            extension = get_file_extension(file)
            if extension == "tsv":
                sep = "\t"
            elif extension == "csv":
                sep = ","
            logging.warning(f"Could not guess file extension for {file}")
    df = read_csv(file, comment="#", sep=sep).fillna("")
    return sort_df_rows_columns(df)


def extract_global_metadata(msdoc: MappingSetDocument) -> Dict[str, PrefixMap]:
    """Extract metadata.

    :param msdoc: MappingSetDocument object
    :return: Dictionary containing metadata
    """
    meta = {PREFIX_MAP_KEY: msdoc.prefix_map}
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


# to_mapping_set_document is in parser.py in order to avoid circular import errors
def to_mapping_set_dataframe(doc: MappingSetDocument) -> MappingSetDataFrame:
    """Convert MappingSetDocument into MappingSetDataFrame.

    :param doc: MappingSetDocument object
    :return: MappingSetDataFrame object
    """
    data = []
    slots_with_double_as_range = [
        s
        for s in _get_sssom_schema_object().dict["slots"].keys()
        if _get_sssom_schema_object().dict["slots"][s]["range"] == "double"
    ]
    if doc.mapping_set.mappings is not None:
        for mapping in doc.mapping_set.mappings:
            m = get_dict_from_mapping(mapping)
            data.append(m)
    df = pd.DataFrame(data=data)
    meta = extract_global_metadata(doc)
    meta.pop(PREFIX_MAP_KEY, None)
    # The following 3 lines are to remove columns
    # where all values are blank.
    df.replace("", np.nan, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)  # remove columns with all row = 'None'-s.
    non_double_cols = df.loc[:, ~df.columns.isin(slots_with_double_as_range)]
    non_double_cols = non_double_cols.replace(np.nan, "")
    df[non_double_cols.columns] = non_double_cols
    msdf = MappingSetDataFrame(df=df, prefix_map=doc.prefix_map, metadata=meta)
    msdf.df = sort_df_rows_columns(msdf.df)
    return msdf


def get_dict_from_mapping(map_obj: Union[Any, Dict[Any, Any], SSSOM_Mapping]) -> dict:
    """
    Get information for linkml objects (MatchTypeEnum, PredicateModifierEnum) from the Mapping object and return the dictionary form of the object.

    :param map_obj: Mapping object
    :return: Dictionary
    """
    map_dict = {}
    slots_with_double_as_range = [
        s
        for s in _get_sssom_schema_object().dict["slots"].keys()
        if _get_sssom_schema_object().dict["slots"][s]["range"] == "double"
    ]
    for property in map_obj:
        if map_obj[property] is not None:
            if isinstance(map_obj[property], list):
                # IF object is an enum
                if (
                    _get_sssom_schema_object().dict["slots"][property]["range"]
                    in _get_sssom_schema_object().dict["enums"].keys()
                ):
                    # IF object is a multivalued enum
                    if _get_sssom_schema_object().dict["slots"][property]["multivalued"]:
                        map_dict[property] = "|".join(
                            enum_value.code.text for enum_value in map_obj[property]
                        )
                    # If object is NOT multivalued BUT an enum.
                    else:
                        map_dict[property] = map_obj[property].code.text
                # IF object is NOT an enum but a list
                else:
                    map_dict[property] = "|".join(enum_value for enum_value in map_obj[property])
            # IF object NOT a list
            else:
                # IF object is an enum
                if (
                    _get_sssom_schema_object().dict["slots"][property]["range"]
                    in _get_sssom_schema_object().dict["enums"].keys()
                ):
                    map_dict[property] = map_obj[property].code.text
                else:
                    map_dict[property] = map_obj[property]
        else:
            # IF map_obj[property] is None:
            if property in slots_with_double_as_range:
                map_dict[property] = np.nan
            else:
                map_dict[property] = ""

    return map_dict


class NoCURIEException(ValueError):
    """An exception raised when a CURIE can not be parsed with a given prefix map."""


CURIE_RE = re.compile(r"[A-Za-z0-9_.]+[:][A-Za-z0-9_]")


def is_curie(string: str) -> bool:
    """Check if the string is a CURIE."""
    return bool(CURIE_RE.match(string))


def is_iri(string: str) -> bool:
    """Check if the string is an IRI."""
    return validators.url(string)


def get_prefix_from_curie(curie: str) -> str:
    """Get the prefix from a CURIE."""
    if is_curie(curie):
        return curie.split(":")[0]
    else:
        return ""


def curie_from_uri(uri: str, prefix_map: Mapping[str, str]) -> str:
    """Parse a CURIE from an IRI.

    :param uri: The URI to parse. If this is already a CURIE, return directly.
    :param prefix_map: The prefix map against which the IRI is checked
    :return: A CURIE
    :raises NoCURIEException: if a CURIE can not be parsed

    Example parsing:
    >>> m = {"hgnc.genegroup": "https://example.org/hgnc.genegroup:"}
    >>> curie_from_uri("https://example.org/hgnc.genegroup:1234", {})
    'hgnc.genegroup:1234'

    Example CURIE passthrough:
    >>> curie_from_uri("hgnc:1234", {})
    'hgnc:1234'
    >>> curie_from_uri("hgnc.genegroup:1234", {})
    'hgnc.genegroup:1234'
    """
    # TODO consider replacing with :func:`bioregistry.curie_from_iri`
    # FIXME what if the curie has a subspace in it? RE will fail
    if is_curie(uri):
        return uri
    for prefix in prefix_map:
        uri_prefix = prefix_map[prefix]
        if uri.startswith(uri_prefix):
            remainder = uri.replace(uri_prefix, "")
            curie = f"{prefix}:{remainder}"
            if is_curie(curie):
                return f"{prefix}:{remainder}"
            else:
                logging.warning(f"{prefix}:{remainder} is not a CURIE ... skipping")
                continue
    raise NoCURIEException(f"{uri} does not follow any known prefixes")


def get_prefixes_used_in_table(df: pd.DataFrame) -> List[str]:
    """Get a list of prefixes used in CURIEs in key feature columns in a dataframe."""
    prefixes = list(SSSOM_BUILT_IN_PREFIXES)
    if not df.empty:
        for col in _get_sssom_schema_object().entity_reference_slots:
            if col in df.columns:
                prefixes.extend(list(set(df[col].str.split(":", n=1, expand=True)[0])))
    if "" in prefixes:
        prefixes.remove("")
    return list(set(prefixes))


def get_prefixes_used_in_metadata(meta: MetadataType) -> List[str]:
    """Get a list of prefixes used in CURIEs in the metadata."""
    prefixes = list(SSSOM_BUILT_IN_PREFIXES)
    if meta:
        for v in meta.values():
            if type(v) is list:
                prefixes.extend(
                    [get_prefix_from_curie(x) for x in v if get_prefix_from_curie(x) != ""]
                )
            else:
                pref = get_prefix_from_curie(str(v))
                if pref != "" and not None:
                    prefixes.append(pref)
    return list(set(prefixes))


def filter_out_prefixes(
    df: pd.DataFrame,
    filter_prefixes: List[str],
    features: list = KEY_FEATURES,
    require_all_prefixes: bool = False,
) -> pd.DataFrame:
    """Filter out rows which contains a CURIE with a prefix in the filter_prefixes list.

    :param df: Pandas DataFrame of SSSOM Mapping
    :param filter_prefixes: List of prefixes
    :param features: List of dataframe column names dataframe to consider
    :param require_all_prefixes: If True, all prefixes must be present in a row to be filtered out
    :return: Pandas Dataframe
    """
    filter_prefix_set = set(filter_prefixes)
    rows = []
    selection = all if require_all_prefixes else any

    for _, row in df.iterrows():
        prefixes = {get_prefix_from_curie(curie) for curie in row[features]}
        if not selection(prefix in prefixes for prefix in filter_prefix_set):
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=features)


def filter_prefixes(
    df: pd.DataFrame,
    filter_prefixes: List[str],
    features: list = KEY_FEATURES,
    require_all_prefixes: bool = True,
) -> pd.DataFrame:
    """Filter out rows which do NOT contain a CURIE with a prefix in the filter_prefixes list.

    :param df: Pandas DataFrame of SSSOM Mapping
    :param filter_prefixes: List of prefixes
    :param features: List of dataframe column names dataframe to consider
    :param require_all_prefixes: If True, all prefixes must be present in a row to be filtered out
    :return: Pandas Dataframe
    """
    filter_prefix_set = set(filter_prefixes)
    rows = []
    selection = all if require_all_prefixes else any

    for _, row in df.iterrows():
        prefixes = {get_prefix_from_curie(curie) for curie in row[features] if curie is not None}
        if selection(prefix in filter_prefix_set for prefix in prefixes):
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=features)


@deprecation.deprecated(details="This is no longer used and will be removed from the public API.")
def guess_file_format(filename: Union[str, TextIO]) -> str:
    """Get file format.

    :param filename: filename
    :raises ValueError: Unrecognized file extension
    :return: File extension
    """
    extension = get_file_extension(filename)
    if extension in ["owl", "rdf"]:
        return SSSOM_DEFAULT_RDF_SERIALISATION
    elif extension in RDF_FORMATS:
        return extension
    else:
        raise ValueError(f"File extension {extension} does not correspond to a legal file format")


def prepare_context(
    prefix_map: Optional[PrefixMap] = None,
) -> Mapping[str, Any]:
    """Prepare a JSON-LD context from a prefix map."""
    context = get_jsonld_context()
    if prefix_map is None:
        prefix_map = get_default_metadata().prefix_map

    for k, v in prefix_map.items():
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
    return context


def prepare_context_str(prefix_map: Optional[PrefixMap] = None, **kwargs) -> str:
    """Prepare a JSON-LD context and dump to a string.

    :param prefix_map: Prefix map, defaults to None
    :param kwargs: Keyword arguments to pass through to :func:`json.dumps`
    :return: Context in str format
    """
    return json.dumps(prepare_context(prefix_map), **kwargs)


def raise_for_bad_prefix_map_mode(prefix_map_mode: str = None):
    """Raise exception if prefix map mode is invalid.

    :param prefix_map_mode: The prefix map mode
    :raises ValueError: Invalid prefix map mode
    """
    if prefix_map_mode not in PREFIX_MAP_MODES:
        raise ValueError(
            f"{prefix_map_mode} is not a valid prefix map mode, "
            f"must be one of {' '.join(PREFIX_MAP_MODES)}"
        )


def raise_for_bad_path(file_path: Union[str, Path]) -> None:
    """Raise exception if file path is invalid.

    :param file_path: File path
    :raises FileNotFoundError: Invalid file path
    """
    if isinstance(file_path, Path):
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} is not a valid file path or url.")
    elif not isinstance(file_path, str):
        logging.info("Path provided to raise_for_bad_path() is neither a Path nor str-like object.")
    elif not validators.url(file_path) and not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} is not a valid file path or url.")


def is_multivalued_slot(slot: str) -> bool:
    """Check whether the slot is multivalued according to the SSSOM specification.

    :param slot: Slot name
    :return: Slot is multivalued or no
    """
    # Ideally:
    # view = SchemaView('schema/sssom.yaml')
    # return view.get_slot(slot).multivalued
    return slot in _get_sssom_schema_object().multivalued_slots


def reconcile_prefix_and_data(
    msdf: MappingSetDataFrame, prefix_reconciliation: dict
) -> MappingSetDataFrame:
    """Reconciles prefix_map and translates CURIE switch in dataframe.

    :param msdf: Mapping Set DataFrame.
    :param prefix_reconciliation: Prefix reconcilation dictionary from a YAML file
    :return: Mapping Set DataFrame with reconciled prefix_map and data.
    """
    # Discussion about this found here:
    # https://github.com/mapping-commons/sssom-py/issues/216#issue-1171701052

    prefix_map = msdf.prefix_map
    df: pd.DataFrame = msdf.df
    data_switch_dict = dict()

    prefix_synonyms = prefix_reconciliation["prefix_synonyms"]
    prefix_expansion = prefix_reconciliation["prefix_expansion_reconciliation"]

    # The prefix exists but the expansion needs to be updated.
    expansion_replace = {
        k: v for k, v in prefix_expansion.items() if k in prefix_map.keys() and v != prefix_map[k]
    }

    # Updates expansions in prefix_map
    prefix_map.update(expansion_replace)

    # Prefixes that need to be replaced
    # IF condition:
    #   1. Key OR Value in prefix_synonyms are keys in prefix_map
    #       e.g.: ICD10: ICD10CM - either should be present within
    #           the prefix_map.
    #   AND
    #   2. Value in prefix_synonyms is NOT a value in expansion_replace.
    #      In other words, the existing expansion do not match the YAML.

    prefix_replace = [
        k
        for k, v in prefix_synonyms.items()
        if (k in prefix_map.keys() or v in prefix_map.keys()) and v not in expansion_replace.keys()
    ]

    if len(prefix_replace) > 0:
        for pr in prefix_replace:
            correct_prefix = prefix_synonyms[pr]
            correct_expansion = prefix_expansion[correct_prefix]
            prefix_map[correct_prefix] = correct_expansion
            logging.info(f"Adding prefix_map {correct_prefix}: {correct_expansion}")
            if pr in prefix_map.keys():
                prefix_map.pop(pr, None)
                data_switch_dict[pr] = correct_prefix

                logging.warning(f"Replacing prefix {pr} with {correct_prefix}")

    # Data editing
    if len(data_switch_dict) > 0:
        # Read schema file
        slots = _get_sssom_schema_object().dict["slots"]
        entity_reference_columns = [k for k, v in slots.items() if v["range"] == "EntityReference"]
        update_columns = [c for c in df.columns if c in entity_reference_columns]
        for k, v in data_switch_dict.items():
            df[update_columns] = df[update_columns].replace(k + ":", v + ":", regex=True)

    msdf.df = df
    msdf.prefix_map = prefix_map

    # TODO: When expansion of 2 prefixes in the prefix_map are the same.
    return msdf


def sort_df_rows_columns(
    df: pd.DataFrame, by_columns: bool = True, by_rows: bool = True
) -> pd.DataFrame:
    """
    Canonical sorting of DataFrame columns.

    :param df: Pandas DataFrame with random column sequence.
    :param by_columns: Boolean flag to sort columns canonically.
    :param by_rows: Boolean flag to sort rows by column #1 (ascending order).
    :return: Pandas DataFrame columns sorted canonically.
    """
    if by_columns and len(df.columns) > 0:
        column_sequence = [
            col for col in _get_sssom_schema_object().dict["slots"].keys() if col in df.columns
        ]
        df = df.reindex(column_sequence, axis=1)
    if by_rows and len(df) > 0:
        df = df.sort_values(by=column_sequence, ignore_index=True, na_position="last")
    return df


def get_all_prefixes(msdf: MappingSetDataFrame) -> list:
    """Fetch all prefixes in the MappingSetDataFrame.

    :param msdf: MappingSetDataFrame
    :raises ValidationError: If slot is wrong.
    :raises ValidationError: If slot is wrong.
    :return:  List of all prefixes.
    """
    prefix_list = []
    if msdf.metadata and not msdf.df.empty:  # type: ignore
        metadata_keys = list(msdf.metadata.keys())
        df_columns_list = msdf.df.columns.to_list()  # type: ignore
        all_keys = metadata_keys + df_columns_list
        ent_ref_slots = [
            s for s in all_keys if s in _get_sssom_schema_object().entity_reference_slots
        ]

        for slot in ent_ref_slots:
            if slot in metadata_keys:
                if type(msdf.metadata[slot]) == list:
                    for s in msdf.metadata[slot]:
                        if get_prefix_from_curie(s) == "":
                            # print(
                            #     f"Slot '{slot}' has an incorrect value: {msdf.metadata[s]}"
                            # )
                            raise ValidationError(
                                f"Slot '{slot}' has an incorrect value: {msdf.metadata[s]}"
                            )
                        prefix_list.append(get_prefix_from_curie(s))
                else:
                    if get_prefix_from_curie(msdf.metadata[slot]) == "":
                        # print(
                        #     f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}"
                        # )
                        logging.warning(
                            f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}"
                        )
                    prefix_list.append(get_prefix_from_curie(msdf.metadata[slot]))
            else:
                column_prefixes = list(
                    set(
                        [
                            get_prefix_from_curie(s)
                            for s in list(set(msdf.df[slot].to_list()))  # type: ignore
                            if get_prefix_from_curie(s) != ""
                        ]
                    )
                )
                prefix_list = prefix_list + column_prefixes

        prefix_list = list(set(prefix_list))

    return prefix_list


def augment_metadata(
    msdf: MappingSetDataFrame, meta: dict, replace_multivalued: bool = False
) -> MappingSetDataFrame:
    """Augment metadata with parameters passed.

    :param msdf: MappingSetDataFrame (MSDF) object.
    :param meta: Dictionary that needs to be added/updated to the metadata of the MSDF.
    :param replace_multivalued: Multivalued slots should be
        replaced or not, defaults to False.
    :raises ValueError: If type of slot is neither str nor list.
    :return: MSDF with updated metadata.
    """
    are_params_slots(meta)

    if msdf.metadata:
        for k, v in meta.items():
            # If slot is multivalued, add to list.
            if k in _get_sssom_schema_object().multivalued_slots and not replace_multivalued:
                tmp_value: list = []
                if isinstance(msdf.metadata[k], str):
                    tmp_value = [msdf.metadata[k]]
                elif isinstance(msdf.metadata[k], list):
                    tmp_value = msdf.metadata[k]
                else:
                    raise ValueError(
                        f"{k} is of type {type(msdf.metadata[k])} and \
                        as of now only slots of type 'str' or 'list' are handled."
                    )
                tmp_value.extend(v)
                msdf.metadata[k] = list(set(tmp_value))
            elif k in _get_sssom_schema_object().multivalued_slots and replace_multivalued:
                msdf.metadata[k] = list(v)
            else:
                msdf.metadata[k] = v[0]

    return msdf


def are_params_slots(params: dict) -> bool:
    """Check if parameters conform to the slots in MAPPING_SET_SLOTS.

    :param params: Dictionary of parameters.
    :raises ValueError: If params are not slots.
    :return: True/False
    """
    empty_params = {k: v for k, v in params.items() if v is None or v == ""}
    if len(empty_params) > 0:
        logging.info(f"Parameters: {empty_params.keys()} has(ve) no value.")

    legit_params = all(p in _get_sssom_schema_object().mapping_set_slots for p in params.keys())
    if not legit_params:
        invalids = [p for p in params if p not in _get_sssom_schema_object().mapping_set_slots]
        raise ValueError(
            f"The params are invalid: {invalids}. Should be any of the following: {_get_sssom_schema_object().mapping_set_slots}"
        )
    return True


def _get_sssom_schema_object() -> SSSOMSchemaView:
    sssom_sv_object = (
        SSSOMSchemaView.instance if hasattr(SSSOMSchemaView, "instance") else SSSOMSchemaView()
    )
    return sssom_sv_object


def invert_mappings(
    df: pd.DataFrame,
    subject_prefix: Optional[str] = None,
    merge_inverted: bool = True,
    predicate_invert_dictionary: dict = None,
) -> pd.DataFrame:
    """Switching subject and objects based on their prefixes and adjusting predicates accordingly.

    :param df: Pandas dataframe.
    :param subject_prefix: Prefix of subjects desired.
    :param merge_inverted: If True (default), add inverted dataframe to input else,
                          just return inverted data.
    :param predicate_invert_dictionary: YAML file providing the inverse mapping for predicates.
    :return: Pandas dataframe with all subject IDs having the same prefix.
    """
    if predicate_invert_dictionary:
        predicate_invert_map = predicate_invert_dictionary
    else:
        predicate_invert_map = PREDICATE_INVERT_DICTIONARY
    columns_invert_map = COLUMN_INVERT_DICTIONARY

    if PREDICATE_MODIFIER in df.columns:
        blank_predicate_modifier = df[PREDICATE_MODIFIER] == ""
        predicate_modified_df = pd.DataFrame(df[~blank_predicate_modifier])
        non_predicate_modified_df = pd.DataFrame(df[blank_predicate_modifier])
    else:
        predicate_modified_df = pd.DataFrame(columns=df.columns)
        non_predicate_modified_df = df

    if subject_prefix:
        subject_starts_with_prefix_condition = df[SUBJECT_ID].str.startswith(subject_prefix + ":")
        object_starts_with_prefix_condition = df[OBJECT_ID].str.startswith(subject_prefix + ":")
        prefixed_subjects_df = pd.DataFrame(
            non_predicate_modified_df[
                (subject_starts_with_prefix_condition & ~object_starts_with_prefix_condition)
            ]
        )
        non_prefix_subjects_df = pd.DataFrame(
            non_predicate_modified_df[
                (~subject_starts_with_prefix_condition & object_starts_with_prefix_condition)
            ]
        )
        df_to_invert = non_prefix_subjects_df.loc[
            non_prefix_subjects_df[PREDICATE_ID].isin(list(predicate_invert_map.keys()))
        ]
        non_inverted_df_by_predicate = pd.DataFrame(columns=non_prefix_subjects_df.columns)
    else:
        prefixed_subjects_df = pd.DataFrame()
        df_to_invert = non_predicate_modified_df.loc[
            non_predicate_modified_df[PREDICATE_ID].isin(list(predicate_invert_map.keys()))
        ]
        non_inverted_df_by_predicate = non_predicate_modified_df.loc[
            ~non_predicate_modified_df[PREDICATE_ID].isin(list(predicate_invert_map.keys()))
        ]
    list_of_subject_object_columns = [
        x for x in df_to_invert.columns if x.startswith(("subject", "object"))
    ]
    inverted_df = df_to_invert.rename(
        columns=_invert_column_names(list_of_subject_object_columns, columns_invert_map)
    )
    inverted_df = inverted_df[df.columns]
    inverted_df[PREDICATE_ID] = inverted_df[PREDICATE_ID].map(predicate_invert_map)
    inverted_df[MAPPING_JUSTIFICATION] = SEMAPV.MappingInversion.value
    if not prefixed_subjects_df.empty:
        return_df = pd.concat([prefixed_subjects_df, inverted_df]).drop_duplicates()
    else:
        return_df = pd.concat(
            [inverted_df, predicate_modified_df, non_inverted_df_by_predicate]
        ).drop_duplicates()
    if merge_inverted:
        return pd.concat([df, return_df]).drop_duplicates()
    else:
        return return_df


def _invert_column_names(column_names: list, columns_invert_map: dict) -> dict:
    """Return a dictionary for column renames in pandas DataFrame."""
    return {x: columns_invert_map[x] for x in column_names}
