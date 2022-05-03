"""Utilities for SSSOM."""

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

import numpy as np
import pandas as pd
import validators
import yaml
from linkml_runtime.linkml_model.types import Uriorcurie

from .constants import SCHEMA_DICT, SCHEMA_YAML
from .context import SSSOM_URI_PREFIX, get_default_metadata, get_jsonld_context
from .internal_context import multivalued_slots
from .sssom_datamodel import Mapping as SSSOM_Mapping
from .sssom_datamodel import slots
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
SSSOM_EXPORT_FORMATS = ["tsv", "rdf", "owl", "json"]

SSSOM_DEFAULT_RDF_SERIALISATION = "turtle"


# TODO: use sssom_datamodel (Mapping Class)
SUBJECT_ID = "subject_id"
SUBJECT_LABEL = "subject_label"
OBJECT_ID = "object_id"
OBJECT_LABEL = "object_label"
PREDICATE_ID = "predicate_id"
PREDICATE_MODIFIER = "predicate_modifier"
PREDICATE_MODIFIER_NOT = "Not"
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
MAPPING_SET_SOURCE = "mapping_set_source"

URI_SSSOM_MAPPINGS = f"{SSSOM_URI_PREFIX}mappings"

#: The 3 columns whose combination would be used as primary keys while merging/grouping
KEY_FEATURES = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID]


@dataclass
class MappingSetDataFrame:
    """A collection of mappings represented as a DataFrame, together with additional metadata."""

    df: Optional[pd.DataFrame] = None  # Mappings
    # maps CURIE prefixes to URI bases
    prefix_map: PrefixMap = field(default_factory=dict)
    metadata: Optional[MetadataType] = None  # header metadata excluding prefixes

    def merge(
        self, *msdfs: "MappingSetDataFrame", inplace: bool = True
    ) -> "MappingSetDataFrame":
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

    def clean_prefix_map(self) -> None:
        """Remove unused prefixes from the internal prefix map based on the internal dataframe."""
        prefixes_in_map = get_prefixes_used_in_table(self.df)
        new_prefixes: PrefixMap = dict()
        missing_prefixes = []
        for prefix in prefixes_in_map:
            if prefix in self.prefix_map:
                new_prefixes[prefix] = self.prefix_map[prefix]
            else:
                logging.warning(
                    f"{prefix} is used in the data frame but does not exist in prefix map"
                )
                missing_prefixes.append(prefix)
        if missing_prefixes:
            self.df = filter_out_prefixes(self.df, missing_prefixes)
        self.prefix_map = new_prefixes


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
    df2 = (
        df.groupby([SUBJECT_ID, PREDICATE_ID, OBJECT_ID])[CONFIDENCE]
        .apply(max)
        .reset_index()
    )
    return df2


def sort_sssom(df: pd.DataFrame) -> pd.DataFrame:
    """Sort SSSOM by columns.

    :param df: SSSOM DataFrame to be sorted.
    :return: Sorted SSSOM DataFrame
    """
    df.sort_values(by=sorted(df.columns), ascending=False, inplace=True)
    return df


def filter_redundant_rows(
    df: pd.DataFrame, ignore_predicate: bool = False
) -> pd.DataFrame:
    """Remove rows if there is another row with same S/O and higher confidence.

    :param df: Pandas DataFrame to filter
    :param ignore_predicate: If true, the predicate_id column is ignored, defaults to False
    :return: Filtered pandas DataFrame
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
    max_conf: Dict[Tuple[str, ...], float] = {}
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
    if return_df[CONFIDENCE].isnull().all():
        return_df = return_df.drop(columns=[CONFIDENCE], axis=1)
    return return_df


def assign_default_confidence(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign :data:`numpy.nan` to confidence that are blank.

    :param df: SSSOM DataFrame
    :return: A Tuple consisting of the original DataFrame and dataframe consisting of empty confidence values.
    """
    # Get rows having numpy.NaN as confidence
    if df is not None:
        if CONFIDENCE not in df.columns:
            df[CONFIDENCE] = np.NaN
            nan_df = pd.DataFrame(columns=df.columns)
        else:
            df = df[~df[CONFIDENCE].isna()]
            nan_df = df[df[CONFIDENCE].isna()]
    else:
        ValueError("DataFrame cannot be empty to 'assign_default_confidence'.")
    return df, nan_df


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
    d.combined_dataframe = pd.DataFrame(rows)
    return d


def dataframe_to_ptable(df: pd.DataFrame, *, inverse_factor: float = 0.5):
    """Export a KBOOM table.

    :param df: Pandas DataFrame
    :param inverse_factor: Multiplier to (1 - confidence), defaults to 0.5
    :raises ValueError: Predicate value error
    :raises ValueError: Predicate type value error
    :return: List of rows
    """
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
        # * Added by H2 ############################
        elif predicate == "oboInOwl:hasDbXref":
            predicate_type = PREDICATE_HAS_DBXREF
        elif predicate == "skos:relatedMatch":
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
    reconcile: bool = True,
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
        if PREDICATE_MODIFIER in merged_msdf.df.columns:
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
        MATCH_TYPE,
    ]
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
        reconciled_df_subset[PREDICATE_MODIFIER] = reconciled_df_subset[
            PREDICATE_MODIFIER
        ].fillna("")

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
                msdf.df[k] = v
    return msdf


def get_file_extension(file: Union[str, Path, TextIO]) -> str:
    """Get file extension.

    :param file: File path
    :raises Exception: Cannot determine extension exception
    :return: format of the file passed
    """
    if isinstance(file, str):
        filename = file
    elif isinstance(file, Path):
        return file.suffix
    else:
        filename = file.name
    parts = filename.split(".")
    if len(parts) > 0:
        f_format = parts[-1]
        return f_format
    else:
        raise Exception(f"Cannot guess format from {filename}")


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
    return pd.read_csv(StringIO(lines), sep=sep)


def read_metadata(filename: str) -> Metadata:
    """Read a metadata file (yaml) that is supplied separately from a TSV."""
    prefix_map = {}
    with open(filename) as file:
        metadata = yaml.safe_load(file)
    if PREFIX_MAP_KEY in metadata:
        prefix_map = metadata.pop(PREFIX_MAP_KEY)
    return Metadata(prefix_map=prefix_map, metadata=metadata)


def read_pandas(
    file: Union[str, Path, TextIO], sep: Optional[str] = None
) -> pd.DataFrame:
    """Read a tabular data file by wrapping func:`pd.read_csv` to handles comment lines correctly.

    :param file: The file to read. If no separator is given, this file should be named.
    :param sep: File separator for pandas
    :return: A pandas dataframe
    """
    if sep is None:
        extension = get_file_extension(file)
        if extension == "tsv":
            sep = "\t"
        elif extension == "csv":
            sep = ","
        else:
            sep = "\t"
            logging.warning("Cannot automatically determine table format, trying tsv.")
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
        for s in SCHEMA_DICT["slots"].keys()
        if SCHEMA_DICT["slots"][s]["range"] == "double"
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
    df = df.dropna(axis=1, how="all")  # remove columns with all row = 'None'-s.
    df.loc[:, ~df.columns.isin(slots_with_double_as_range)].replace(
        np.nan, "", inplace=True
    )
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
        for s in SCHEMA_DICT["slots"].keys()
        if SCHEMA_DICT["slots"][s]["range"] == "double"
    ]
    for property in map_obj:
        if map_obj[property] is not None:
            if isinstance(map_obj[property], list):
                # IF object is an enum
                if (
                    SCHEMA_DICT["slots"][property]["range"]
                    in SCHEMA_DICT["enums"].keys()
                ):
                    # IF object is a multivalued enum
                    if SCHEMA_DICT["slots"][property]["multivalued"]:
                        map_dict[property] = "|".join(
                            enum_value.code.text for enum_value in map_obj[property]
                        )
                    # If object is NOT multivalued BUT an enum.
                    else:
                        map_dict[property] = map_obj[property].code.text
                # IF object is NOT an enum but a list
                else:
                    map_dict[property] = "|".join(
                        enum_value for enum_value in map_obj[property]
                    )
            # IF object NOT a list
            else:
                # IF object is an enum
                if (
                    SCHEMA_DICT["slots"][property]["range"]
                    in SCHEMA_DICT["enums"].keys()
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


CURIE_RE = re.compile(r"[A-Za-z0-9_]+[:][A-Za-z0-9_]")


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
    prefixes = []
    if not df.empty:
        for col in KEY_FEATURES:
            for v in df[col].values:
                prefixes.append(get_prefix_from_curie(v))
    return list(set(prefixes))


def filter_out_prefixes(df: pd.DataFrame, filter_prefixes: List[str]) -> pd.DataFrame:
    """Filter any row where a CURIE in one of the key column uses one of the given prefixes.

    :param df: Pandas DataFrame
    :param filter_prefixes: List of prefixes
    :return: Pandas Dataframe
    """
    filter_prefix_set = set(filter_prefixes)
    rows = []

    for _, row in df.iterrows():
        # Get list of CURIEs from the 3 columns (KEY_FEATURES) for the row.
        prefixes = {get_prefix_from_curie(curie) for curie in row[KEY_FEATURES]}
        # Confirm if none of the 3 CURIEs in the list above appear in the filter_prefixes list.
        # If TRUE, append row.
        if not any(prefix in prefixes for prefix in filter_prefix_set):
            rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=KEY_FEATURES)


# TODO this is not used anywhere
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
        raise ValueError(
            f"File extension {extension} does not correspond to a legal file format"
        )


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


def raise_for_bad_path(file_path: Union[str, Path]) -> None:
    """Raise exception if file path is invalid.

    :param file_path: File path
    :raises FileNotFoundError: Invalid file path
    """
    if isinstance(file_path, Path):
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} is not a valid file path or url.")
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

    return slot in multivalued_slots


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
        k: v
        for k, v in prefix_expansion.items()
        if k in prefix_map.keys() and v != prefix_map[k]
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
        if (k in prefix_map.keys() or v in prefix_map.keys())
        and v not in expansion_replace.keys()
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
        slots = SCHEMA_DICT["slots"]
        entity_reference_columns = [
            k for k, v in slots.items() if v["range"] == "EntityReference"
        ]
        update_columns = [c for c in df.columns if c in entity_reference_columns]
        for k, v in data_switch_dict.items():
            df[update_columns] = df[update_columns].replace(
                k + ":", v + ":", regex=True
            )

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
            col for col in SCHEMA_DICT["slots"].keys() if col in df.columns
        ]
        df = df.reindex(column_sequence, axis=1)
    if by_rows and len(df) > 0:
        df = df.sort_values(by=df.columns[0], ignore_index=True)
    return df
