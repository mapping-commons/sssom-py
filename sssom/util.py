import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from io import FileIO, StringIO
from typing import (
    Any,
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

from .context import get_default_metadata, get_jsonld_context
from .sssom_datamodel import Entity, slots
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
    """A collection of mappings represented as a DataFrame, together with additional metadata."""

    df: Optional[pd.DataFrame] = None  # Mappings
    #: maps CURIE prefixes to URI bases
    prefix_map: PrefixMap = field(default_factory=dict)
    metadata: Optional[MetadataType] = None  # header metadata excluding prefixes

    def merge(
        self, msdf2: "MappingSetDataFrame", inplace: bool = True
    ) -> "MappingSetDataFrame":
        """Merge two MappingSetDataframes.

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
            self.prefix_map = msdf.prefix_map
            self.metadata = msdf.metadata
            # FIXME should return self if inplace
        return msdf

    def __str__(self) -> str:  # noqa:D105
        description = "SSSOM data table \n"
        description += f"Number of mappings: {len(self.df.index)} \n"
        description += f"Number of prefixes: {len(self.prefix_map)} \n"
        description += f"Metadata: {json.dumps(self.metadata)} \n"
        description += "\nFirst rows of data: \n"
        description += self.df.head().to_string() + "\n"
        description += "\nLast rows of data: \n"
        description += self.df.tail().to_string() + "\n"
        return description

    def clean_prefix_map(self) -> None:
        prefixes_in_map = get_prefixes_used_in_table(self.df)
        new_prefixes: PrefixMap = dict()
        missing_prefix = []
        for prefix in prefixes_in_map:
            if prefix in self.prefix_map:
                new_prefixes[prefix] = self.prefix_map[prefix]
            else:
                logging.warning(
                    f"{prefix} is used in the data frame but does not exist in prefix map"
                )
                missing_prefix.append(prefix)
        if missing_prefix:
            self.df = filter_out_prefixes(self.df, missing_prefix)
        self.prefix_map = new_prefixes


@dataclass
class EntityPair:
    """
    A tuple of entities.

    Note that (e1,e2) == (e2,e1)
    """

    subject_entity: Entity
    object_entity: Entity

    def __hash__(self) -> int:  # noqa:D105
        if self.subject_entity.id <= self.object_entity.id:
            t = self.subject_entity.id, self.object_entity.id
        else:
            t = self.object_entity.id, self.subject_entity.id
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


def sort_sssom_columns(columns: List[str]) -> List[str]:
    """Sort columns: Ideally, the order of the sssom column names is parsed strictly from sssom.yaml.

    :param columns: Column names
    :type columns: List[str]
    :return: Sorted column names
    :rtype: List[str]
    """
    logging.warning("SSSOM sort columns not implemented")
    columns.sort()
    return columns


def sort_sssom(df: pd.DataFrame) -> pd.DataFrame:
    """Sort SSSOM by columns.

    :param df: SSSOM DataFrame to be sorted.
    :type df: pd.DataFrame
    :return: Sorted SSSOM DataFrame
    :rtype: pd.DataFrame
    """
    df.sort_values(
        by=sort_sssom_columns(list(df.columns)), ascending=False, inplace=True
    )
    return df


def filter_redundant_rows(
    df: pd.DataFrame, ignore_predicate: bool = False
) -> pd.DataFrame:
    """Remove rows if there is another row with same S/O and higher confidence.

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
    return return_df


def assign_default_confidence(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign numpy.NaN values to confidence that are blank.

    :param df: SSSOM DataFrame
    :type df: pd.DataFrame
    :return: A Tuple consisting of the original DataFrame and dataframe consisting of empty confidence values.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # Get rows having numpy.NaN as confidence
    if df is not None and "confidence" not in df.columns:
        df["confidence"] = np.NaN

    nan_df = df[df["confidence"].isna()]
    if nan_df is None:
        nan_df = pd.DataFrame(columns=df.columns)
    return df, nan_df


def remove_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where no match is found.

    TODO: https://github.com/OBOFoundry/SSSOM/issues/28
    :param df:
    :return:
    """
    return df[df[PREDICATE_ID] != "noMatch"]


def create_entity(row, eid: str, mappings: Dict[str, Any]) -> Entity:
    """Create an Entity object.

    :param row: Row - [UNUSED]
    :type row: [type]
    :param eid: Entity Id
    :type eid: str
    :param mappings: Mapping dictionary
    :type mappings: Dict[str, Any]
    :return: An Entity object
    :rtype: Entity
    """
    logging.warning(f"create_entity() has row parameter ({row}), but not used.")
    e = Entity(id=eid)
    for k, v in mappings.items():
        if k in e:
            e[k] = v
    return e


def group_mappings(df: pd.DataFrame) -> Dict[EntityPair, List[pd.Series]]:
    """Group mappings by EntityPairs."""
    mappings: DefaultDict[EntityPair, List[pd.Series]] = defaultdict(list)
    for _, row in df.iterrows():
        subject_entity = create_entity(
            row,
            row[SUBJECT_ID],
            {
                "label": SUBJECT_LABEL,
                "category": SUBJECT_CATEGORY,
                "source": SUBJECT_SOURCE,
            },
        )
        object_entity = create_entity(
            row,
            row[OBJECT_ID],
            {
                "label": OBJECT_LABEL,
                "category": OBJECT_CATEGORY,
                "source": OBJECT_SOURCE,
            },
        )
        mappings[EntityPair(subject_entity, object_entity)].append(row)
    return dict(mappings)


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> MappingSetDiff:
    """Perform a diff between two SSSOM dataframes.

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


def sha256sum(filename: str) -> str:
    """Hashing function.

    :param filename: Filename
    :type filename: str
    :return: Hashed value
    :rtype: str
    """
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
    """Merge msdf2 into msdf1.

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
    merged_msdf.prefix_map = dict_merge(
        source=msdf2.prefix_map, target=msdf1.prefix_map, dict_name="prefix_map"
    )
    # After a Slack convo with @matentzn, commented out below.
    # merged_msdf.metadata = dict_merge(msdf2.metadata, msdf1.metadata, 'metadata')

    """if inplace:
            msdf1.prefix_map = merged_msdf.prefix_map
            msdf1.metadata = merged_msdf.metadata
            msdf1.df = merged_msdf.df"""

    if reconcile:
        merged_msdf.df = filter_redundant_rows(merged_msdf.df)
        merged_msdf.df = deal_with_negation(merged_msdf.df)  # deals with negation

    return merged_msdf


def deal_with_negation(df: pd.DataFrame) -> pd.DataFrame:
    """Combine negative and positive rows with matching [SUBJECT_ID, OBJECT_ID, CONFIDENCE] combination.

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
    """Merge two dictionaries with a certain structure."""
    if source is None:
        return target
    for k, v in source.items():
        if k not in target:
            if v not in list(target.values()):
                target[k] = v
            else:
                common_values = [i for i, val in target.items() if val == v]
                raise ValueError(
                    f"Value [{v}] is present in {dict_name} for multiple keys [{common_values}]."
                )
        elif target[k] != v:
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


def get_file_extension(file: Union[str, TextIO]) -> str:
    """Get file extension.

    :param file: Name of the file
    :type file: Union[str, TextIO]
    :raises Exception: Cannot determine extension exception
    :return: format of the file passed
    :rtype: str
    """
    if isinstance(file, str):
        filename = file
    else:
        filename = file.name
    parts = filename.split(".")
    if len(parts) > 0:
        f_format = parts[-1]
        return f_format
    else:
        raise Exception(f"Cannot guess format from {filename}")


def read_csv(filename: str, comment: str = "#", sep: str = ",") -> pd.DataFrame:
    """Read TSV file.

    :param filename: filename
    :type filename: str
    :param comment: character that indicates beginning of a comment, defaults to "#"
    :type comment: str, optional
    :param sep: separator, defaults to ","
    :type sep: str, optional
    :return: Pandas DataFrame
    :rtype: pd.DataFrame
    """
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


def read_metadata(filename: str) -> Metadata:
    """Read a metadata file (yaml) that is supplied separately from a TSV."""
    prefix_map = {}
    with open(filename) as file:
        metadata = yaml.safe_load(file)
    if PREFIX_MAP_KEY in metadata:
        prefix_map = metadata.pop(PREFIX_MAP_KEY)
    return Metadata(prefix_map=prefix_map, metadata=metadata)


def read_pandas(file: Union[str, TextIO], sep: Optional[str] = None) -> pd.DataFrame:
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
    return read_csv(file, comment="#", sep=sep).fillna("")


def extract_global_metadata(msdoc: MappingSetDocument) -> Dict[str, PrefixMap]:
    """Extract metadata.

    :param msdoc: MappingSetDocument object
    :type msdoc: MappingSetDocument
    :return: Dictionary containing metadata
    :rtype: Dict[str, PrefixMap]
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


def to_mapping_set_dataframe(doc: MappingSetDocument) -> MappingSetDataFrame:
    """Convert MappingSetDocument into MappingSetDataFrame.

    :param doc: MappingSetDocument object
    :type doc: MappingSetDocument
    :return: MappingSetDataFrame object
    :rtype: MappingSetDataFrame
    """
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
    meta.pop(PREFIX_MAP_KEY, None)
    msdf = MappingSetDataFrame(df=df, prefix_map=doc.prefix_map, metadata=meta)
    return msdf


# to_mapping_set_document is in parser.py in order to avoid circular import errors


class NoCURIEException(ValueError):
    pass


CURIE_RE = re.compile(r"[A-Za-z0-9_]+[:][A-Za-z0-9_]")


def is_curie(string: str) -> bool:
    """Check if string is CURIE or not.

    :param string: String
    :type string: str
    :return: Boolean indicating CURIE or not
    :rtype: bool
    """
    return bool(CURIE_RE.match(string))


def get_prefix_from_curie(curie: str) -> str:
    """Get prefix from CURIE.

    :param curie: CURIE
    :type curie: str
    :return: Prefix
    :rtype: str
    """
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
            return f"{prefix}:{remainder}"
    raise NoCURIEException(f"{uri} does not follow any known prefixes")


def get_prefixes_used_in_table(df: pd.DataFrame) -> List[str]:
    """Get prefixes used in table.

    :param df: Pandas DatafFrame
    :type df: pd.DataFrame
    :return: List of unique prefixes
    :rtype: List[str]
    """
    prefixes = []
    for col in KEY_FEATURES:
        for v in df[col].values:
            prefixes.append(get_prefix_from_curie(v))
    return list(set(prefixes))


def filter_out_prefixes(df: pd.DataFrame, filter_prefixes: List[str]) -> pd.DataFrame:
    """Get all prefixes in filter_prefixes from pandas DataFrame.

    :param df: Pandas DataFrame
    :type df: pd.DataFrame
    :param filter_prefixes: List of prefixes
    :type filter_prefixes: List[str]
    :return: Pandas Dataframe
    :rtype: pd.DataFrame
    """
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


def guess_file_format(filename: Union[str, TextIO]) -> str:
    """Get file format.

    :param filename: filename
    :type filename: Union[str, TextIO]
    :raises Exception: Unrecoginized file extension
    :return: File extension
    :rtype: str
    """
    extension = get_file_extension(filename)
    if extension in ["owl", "rdf"]:
        return SSSOM_DEFAULT_RDF_SERIALISATION
    elif extension in RDF_FORMATS:
        return extension
    else:
        raise Exception(
            f"File extension {extension} does not correspond to a legal file format"
        )


def prepare_context(prefix_map: Optional[PrefixMap] = None):
    """Prepare context.

    :param prefix_map: Prefix map, defaults to None
    :type prefix_map: Optional[PrefixMap], optional
    :return: Context
    :rtype: Any
    """
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
    """Prepare context string.

    :param prefix_map: Prefix map, defaults to None
    :type prefix_map: Optional[PrefixMap], optional
    :return: Context in str format
    :rtype: str
    """
    return json.dumps(prepare_context(prefix_map), **kwargs)


def raise_for_bad_path(file_path: str) -> None:
    """Raise exception if file path is invalid.

    :param file_path: File path
    :type file_path: str
    :raises Exception: Invalid file path
    """
    if not validators.url(file_path) and not os.path.exists(file_path):
        raise Exception(f"{file_path} is not a valid file path or url.")
