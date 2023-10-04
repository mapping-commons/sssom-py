"""Utility functions."""

import itertools as itt
import json
import logging as _logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from string import punctuation
from typing import Any, DefaultDict, Dict, List, Optional, Set, TextIO, Tuple, Union

import curies
import numpy as np
import pandas as pd
import validators
import yaml
from curies import Converter
from jsonschema import ValidationError
from linkml_runtime.linkml_model.types import Uriorcurie
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
    RDFS_SUBCLASS_OF,
    SCHEMA_YAML,
    SEMAPV,
    SKOS_BROAD_MATCH,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_NARROW_MATCH,
    SKOS_RELATED_MATCH,
    SSSOM_SUPERCLASS_OF,
    SSSOM_URI_PREFIX,
    SUBJECT_CATEGORY,
    SUBJECT_ID,
    SUBJECT_LABEL,
    SUBJECT_SOURCE,
    UNKNOWN_IRI,
    SSSOMSchemaView,
)
from .context import SSSOM_BUILT_IN_PREFIXES, _get_built_in_prefix_map
from .sssom_document import MappingSetDocument
from .typehints import MetadataType, PrefixMap, get_default_metadata

logging = _logging.getLogger(__name__)

#: The key that's used in the YAML section of an SSSOM file
PREFIX_MAP_KEY = "curie_map"

SSSOM_DEFAULT_RDF_SERIALISATION = "turtle"

URI_SSSOM_MAPPINGS = f"{SSSOM_URI_PREFIX}mappings"

#: The 4 columns whose combination would be used as primary keys while merging/grouping
KEY_FEATURES = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID, PREDICATE_MODIFIER]
TRIPLES_IDS = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID]


@dataclass
class MappingSetDataFrame:
    """A collection of mappings represented as a DataFrame, together with additional metadata."""

    df: pd.DataFrame
    converter: Converter
    metadata: MetadataType = field(default_factory=get_default_metadata)

    @property
    def prefix_map(self):
        """Get a simple, bijective prefix map."""
        return self.converter.bimap

    @classmethod
    def with_converter(
        cls,
        converter: Converter,
        df: pd.DataFrame,
        metadata: Optional[MetadataType] = None,
    ) -> "MappingSetDataFrame":
        """Instantiate with a converter instead of a vanilla prefix map."""
        # TODO replace with regular instantiation
        return cls(
            df=df,
            converter=converter,
            metadata=metadata or get_default_metadata(),
        )

    def clean_context(self) -> None:
        """Clean up the context."""
        self.converter = curies.chain([_get_built_in_prefix_map(), self.converter])

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
            self.converter = msdf.converter
            self.metadata = msdf.metadata
            return self
        else:
            return msdf

    def __str__(self) -> str:  # noqa:D105
        description = "SSSOM data table \n"
        description += f"Number of extended prefix map records: {len(self.converter.records)} \n"
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
        prefixes_in_table = get_prefixes_used_in_table(self.df, converter=self.converter)
        if self.metadata:
            prefixes_in_table.update(get_prefixes_used_in_metadata(self.metadata))

        missing_prefixes = prefixes_in_table - self.converter.get_prefixes()
        if missing_prefixes and strict:
            raise ValueError(
                f"{missing_prefixes} are used in the SSSOM mapping set but it does not exist in the prefix map"
            )

        subconverter = self.converter.get_subconverter(prefixes_in_table)
        for prefix in missing_prefixes:
            subconverter.add_prefix(prefix, f"{UNKNOWN_IRI}{prefix.lower()}/")

        self.converter = subconverter

    def remove_mappings(self, msdf: "MappingSetDataFrame") -> None:
        """Remove mappings in right msdf from left msdf.

        :param msdf: MappingSetDataframe object to be removed from primary msdf object.
        """
        merge_on = KEY_FEATURES.copy()
        if self.df is not None and PREDICATE_MODIFIER not in self.df.columns:
            merge_on.remove(PREDICATE_MODIFIER)

        self.df = (
            pd.merge(
                self.df,
                msdf.df,
                on=merge_on,
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


def parse(filename: Union[str, Path]) -> pd.DataFrame:
    """Parse a TSV to a pandas frame."""
    logging.info(f"Parsing {filename}")
    return pd.read_csv(filename, sep="\t", comment="#")


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
    if df is None:
        ValueError("DataFrame cannot be empty to 'assign_default_confidence'.")
    new_df = df.copy()
    if CONFIDENCE not in new_df.columns:
        new_df[CONFIDENCE] = 0.0  # np.NaN
        nan_df = pd.DataFrame(columns=new_df.columns)
    else:
        new_df = df[~df[CONFIDENCE].isna()]
        nan_df = df[df[CONFIDENCE].isna()]
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


def add_default_confidence(df: pd.DataFrame, confidence: float = np.NAN) -> pd.DataFrame:
    """Add `confidence` column to DataFrame if absent and initializes to 0.95.

    If `confidence` column already exists, only fill in the None ones by 0.95.

    :param df: DataFrame whose `confidence` column needs to be filled.
    :return: DataFrame with a complete `confidence` column.
    """
    if CONFIDENCE in df.columns:
        df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: confidence * x if x is not None else x)
        df[CONFIDENCE].fillna(float(confidence), inplace=True)
    else:
        df[CONFIDENCE] = float(confidence)

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

    converter = curies.chain(msdf.converter for msdf in msdf_with_meta)
    merged_msdf = MappingSetDataFrame.with_converter(df=df_merged, converter=converter)
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
    msdf = MappingSetDataFrame.with_converter(df=df, converter=doc.converter, metadata=meta)
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


def get_prefixes_used_in_table(df: pd.DataFrame, converter: Converter) -> Set[str]:
    """Get a list of prefixes used in CURIEs in key feature columns in a dataframe."""
    prefixes = set(SSSOM_BUILT_IN_PREFIXES)
    if df.empty:
        return prefixes
    for col in _get_sssom_schema_object().entity_reference_slots:
        if col not in df.columns:
            continue
        prefixes.update(
            converter.parse_curie(row).prefix
            for row in df[col]
            if not is_iri(row) and is_curie(row)
        )
    return set(prefixes)


def get_prefixes_used_in_metadata(meta: MetadataType) -> Set[str]:
    """Get a set of prefixes used in CURIEs in the metadata."""
    prefixes = set(SSSOM_BUILT_IN_PREFIXES)
    if not meta:
        return prefixes
    for value in meta.values():
        if isinstance(value, list):
            prefixes.update(prefix for curie in value if (prefix := get_prefix_from_curie(curie)))
        else:
            if prefix := get_prefix_from_curie(str(value)):
                prefixes.add(prefix)
    return prefixes


def filter_out_prefixes(
    df: pd.DataFrame,
    filter_prefixes: List[str],
    features: Optional[list] = None,
    require_all_prefixes: bool = False,
) -> pd.DataFrame:
    """Filter out rows which contains a CURIE with a prefix in the filter_prefixes list.

    :param df: Pandas DataFrame of SSSOM Mapping
    :param filter_prefixes: List of prefixes
    :param features: List of dataframe column names dataframe to consider
    :param require_all_prefixes: If True, all prefixes must be present in a row to be filtered out
    :return: Pandas Dataframe
    """
    if features is None:
        features = KEY_FEATURES
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
    features: Optional[list] = None,
    require_all_prefixes: bool = True,
) -> pd.DataFrame:
    """Filter out rows which do NOT contain a CURIE with a prefix in the filter_prefixes list.

    :param df: Pandas DataFrame of SSSOM Mapping
    :param filter_prefixes: List of prefixes
    :param features: List of dataframe column names dataframe to consider
    :param require_all_prefixes: If True, all prefixes must be present in a row to be filtered out
    :return: Pandas Dataframe
    """
    if features is None:
        features = KEY_FEATURES
    filter_prefix_set = set(filter_prefixes)
    rows = []
    selection = all if require_all_prefixes else any

    for _, row in df.iterrows():
        prefixes = {get_prefix_from_curie(curie) for curie in row[features] if curie is not None}
        if selection(prefix in filter_prefix_set for prefix in prefixes):
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=features)


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

    This method is build on :func:`curies.remap_curie_prefixes` and
    :func:`curies.rewire`. Note that if you want to overwrite a CURIE prefix in the Bioregistry
    extended prefix map, you need to provide a place for the old one to go as in
    ``{"geo": "ncbi.geo", "geogeo": "geo"}``.
    Just doing ``{"geogeo": "geo"}`` would not work since `geo` already exists.
    """
    # Discussion about this found here:
    # https://github.com/mapping-commons/sssom-py/issues/216#issue-1171701052
    converter = msdf.converter
    converter = curies.remap_curie_prefixes(converter, prefix_reconciliation["prefix_synonyms"])
    converter = curies.rewire(converter, prefix_reconciliation["prefix_expansion_reconciliation"])

    # TODO make this standardization code directly part of msdf after
    #  switching to native converter
    def _upgrade(curie_or_iri: str) -> str:
        if not is_iri(curie_or_iri) and is_curie(curie_or_iri):
            return converter.standardize_curie(curie_or_iri) or curie_or_iri
        return curie_or_iri

    for column, values in _get_sssom_schema_object().dict["slots"].items():
        if values["range"] != "EntityReference":
            continue
        if column not in msdf.df.columns:
            continue
        msdf.df[column] = msdf.df[column].map(_upgrade)

    msdf.converter = converter
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


def get_all_prefixes(msdf: MappingSetDataFrame) -> Set[str]:
    """Fetch all prefixes in the MappingSetDataFrame.

    :param msdf: MappingSetDataFrame
    :raises ValidationError: If slot is wrong.
    :raises ValidationError: If slot is wrong.
    :return:  List of all prefixes.
    """
    if not msdf.metadata or msdf.df is None or msdf.df.empty:
        return set()

    prefixes: Set[str] = set()
    metadata_keys = set(msdf.metadata.keys())
    keys = {
        slot
        for slot in itt.chain(metadata_keys, msdf.df.columns.to_list())
        if slot in _get_sssom_schema_object().entity_reference_slots
    }
    for slot in keys:
        if slot not in metadata_keys:
            prefixes.update(
                prefix
                for curie in msdf.df[slot].unique()
                if (prefix := get_prefix_from_curie(curie))
            )
        elif isinstance(msdf.metadata[slot], list):
            for curie in msdf.metadata[slot]:
                prefix = get_prefix_from_curie(curie)
                if not prefix:
                    raise ValidationError(
                        f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}"
                    )
                prefixes.add(prefix)
        else:
            prefix = get_prefix_from_curie(msdf.metadata[slot])
            if not prefix:
                logging.warning(f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}")
                continue
            prefixes.add(prefix)

    return prefixes


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
    if not msdf.metadata:
        return msdf
    for k, v in meta.items():
        # If slot is multivalued, add to list.
        if k in _get_sssom_schema_object().multivalued_slots and not replace_multivalued:
            tmp_value: list = []
            if isinstance(msdf.metadata[k], str):
                tmp_value = [msdf.metadata[k]]
            elif isinstance(msdf.metadata[k], list):
                tmp_value = msdf.metadata[k]
            else:
                raise TypeError(
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


def safe_compress(uri: str, converter: Converter) -> str:
    """Parse a CURIE from an IRI.

    :param uri: The URI to parse. If this is already a CURIE, return directly.
    :param converter: Converter used for compression
    :return: A CURIE
    """
    if not is_curie(uri):
        return converter.compress_strict(uri)
    rv = converter.standardize_curie(uri)
    if rv is None:
        raise ValueError(
            f"CURIE appeared where there should be a URI, and could not be standardized: {uri}"
        )
    return rv
