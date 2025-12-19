"""Utility functions."""

from __future__ import annotations

import itertools as itt
import json
import logging as _logging
import os
import re
from collections import ChainMap, defaultdict
from dataclasses import dataclass, field
from functools import partial, reduce
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import curies
import numpy as np
import pandas as pd
import validators
from curies import Converter, ReferenceTuple
from jsonschema import ValidationError
from linkml_runtime.linkml_model.types import Uriorcurie
from packaging.version import parse
from sssom_schema import Mapping as SSSOM_Mapping
from sssom_schema import MappingSet, slots
from typing_extensions import TypedDict

from .constants import (
    CARDINALITY_SCOPE,
    COLUMN_INVERT_DICTIONARY,
    COMMENT,
    CONFIDENCE,
    EXTENSION_DEFINITIONS,
    EXTENSION_SLOT_NAME,
    MAPPING_CARDINALITY,
    MAPPING_JUSTIFICATION,
    MAPPING_SET_ID,
    MAPPING_SOURCE,
    NO_TERM_FOUND,
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
    MetadataType,
    PathOrIO,
    SSSOMSchemaView,
    _get_sssom_schema_object,
    get_default_metadata,
    parse_sssom_version,
)
from .context import (
    SSSOM_BUILT_IN_PREFIXES,
    ConverterHint,
    _get_built_in_prefix_map,
    ensure_converter,
    get_converter,
)
from .sssom_document import MappingSetDocument

logging = _logging.getLogger(__name__)

X = TypeVar("X")
Y = TypeVar("Y")

SSSOM_DEFAULT_RDF_SERIALISATION = "turtle"

URI_SSSOM_MAPPINGS = f"{SSSOM_URI_PREFIX}mappings"

#: The 4 columns whose combination would be used as primary keys while merging/grouping
KEY_FEATURES = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID, PREDICATE_MODIFIER]
TRIPLES_IDS = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID]

# ! This will be unnecessary when pandas >= 3.0.0 is released
# ! https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.infer_objects.html#
# A value is trying to be set on a copy of a slice from a DataFrame
pd.options.mode.copy_on_write = True
# Get the version of pandas as a tuple of integers
pandas_version = parse(pd.__version__).release  # Returns (major, minor, micro)


@dataclass
class MappingSetDataFrame:
    """A collection of mappings represented as a DataFrame, together with additional metadata."""

    df: pd.DataFrame
    converter: Converter = field(default_factory=get_converter)
    metadata: MetadataType = field(default_factory=get_default_metadata)

    @property
    def prefix_map(self) -> Mapping[str, str]:
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

    @classmethod
    def from_mappings(
        cls,
        mappings: List[SSSOM_Mapping],
        *,
        converter: ConverterHint = None,
        metadata: Optional[MetadataType] = None,
    ) -> "MappingSetDataFrame":
        """Instantiate from a list of mappings, mapping set metadata, and an optional converter."""
        # This combines multiple pieces of metadata in the following priority order:
        #  1. The explicitly given metadata passed to from_mappings()
        #  2. The default metadata (which includes a dummy license and mapping set URI)
        chained_metadata = ChainMap(
            metadata or {},
            get_default_metadata(),
        )
        mapping_set = MappingSet(mappings=mappings, **chained_metadata)
        return cls.from_mapping_set(mapping_set=mapping_set, converter=converter)

    @classmethod
    def from_mapping_set(
        cls, mapping_set: MappingSet, *, converter: ConverterHint = None
    ) -> "MappingSetDataFrame":
        """Instantiate from a mapping set and an optional converter.

        :param mapping_set: A mapping set
        :param converter: A prefix map or pre-instantiated converter. If none given, uses a default
            prefix map derived from the Bioregistry.

        :returns: A mapping set dataframe
        """
        doc = MappingSetDocument(converter=ensure_converter(converter), mapping_set=mapping_set)
        return cls.from_mapping_set_document(doc)

    @classmethod
    def from_mapping_set_document(cls, doc: MappingSetDocument) -> "MappingSetDataFrame":
        """Instantiate from a mapping set document."""
        if doc.mapping_set.mappings is None:
            return cls(df=pd.DataFrame(), converter=doc.converter)

        df = pd.DataFrame(get_dict_from_mapping(mapping) for mapping in doc.mapping_set.mappings)
        meta = _extract_global_metadata(doc)

        # remove columns where all values are blank.
        df.replace("", np.nan, inplace=True)
        df = df.infer_objects()
        df.dropna(axis=1, how="all", inplace=True)  # remove columns with all row = 'None'-s.

        slots = _get_sssom_schema_object().dict["slots"]
        slots_with_double_as_range = {
            slot for slot, slot_metadata in slots.items() if slot_metadata["range"] == "double"
        }
        non_double_cols = df.loc[:, ~df.columns.isin(slots_with_double_as_range)]

        if pandas_version >= (2, 0, 0):
            # For pandas >= 2.0.0, use the 'copy' parameter
            non_double_cols = non_double_cols.infer_objects(copy=False)
        else:
            # For pandas < 2.0.0, call 'infer_objects()' without any parameters
            non_double_cols = non_double_cols.infer_objects()

        non_double_cols.replace(np.nan, "", inplace=True)
        df.update(non_double_cols)

        df = sort_df_rows_columns(df)
        return cls.with_converter(df=df, converter=doc.converter, metadata=meta)

    def to_mapping_set_document(self) -> "MappingSetDocument":
        """Get a mapping set document."""
        from .parsers import to_mapping_set_document

        return to_mapping_set_document(self)

    def to_mapping_set(self) -> MappingSet:
        """Get a mapping set."""
        return self.to_mapping_set_document().mapping_set

    def to_mappings(self) -> List[SSSOM_Mapping]:
        """Get a mapping set."""
        return cast(List[SSSOM_Mapping], self.to_mapping_set().mappings)

    def clean_context(self) -> None:
        """Clean up the context."""
        self.converter = curies.chain([_get_built_in_prefix_map(), self.converter])

    def standardize_references(self) -> None:
        """Standardize this MSDF's dataframe and metadata with respect to its converter."""
        self._standardize_metadata_references()
        self._standardize_df_references()

    def _standardize_df_references(self) -> None:
        """Standardize this MSDF's dataframe with respect to its converter."""
        func = partial(_standardize_curie_or_iri, converter=self.converter)
        for column, schema_data in _get_sssom_schema_object().dict["slots"].items():
            if schema_data["range"] != "EntityReference":
                continue
            if column not in self.df.columns:
                continue
            self.df[column] = self.df[column].map(func)

    def _standardize_metadata_references(self, *, raise_on_invalid: bool = False) -> None:
        """Standardize this MSDF's metadata with respect to its converter."""
        _standardize_metadata(
            converter=self.converter, metadata=self.metadata, raise_on_invalid=raise_on_invalid
        )

    def merge(self, *msdfs: "MappingSetDataFrame", inplace: bool = True) -> "MappingSetDataFrame":
        """Merge two MappingSetDataframes.

        :param msdfs: Multiple/Single MappingSetDataFrame(s) to merge with self
        :param inplace: If true, msdf2 is merged into the calling MappingSetDataFrame, if false, it
            simply return the merged data frame.

        :returns: Merged MappingSetDataFrame
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
        description += f"Metadata: {json.dumps(self.metadata)} \n"
        description += f"Number of mappings: {len(self.df.index)} \n"
        description += "\nFirst rows of data: \n"
        description += f"{self.df.head().to_string()}\n"
        description += "\nLast rows of data: \n"
        description += f"{self.df.tail().to_string()}\n"
        return description

    def clean_prefix_map(self, strict: bool = True) -> None:
        """Remove unused prefixes from the internal prefix map based on the internal dataframe.

        :param strict: Boolean if True, errors out if all prefixes in dataframe are not listed in
            the 'curie_map'.

        :raises ValueError: If prefixes absent in 'curie_map' and strict flag = True
        """
        prefixes_in_table = get_prefixes_used_in_table(self.df)
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
        if PREDICATE_MODIFIER not in self.df.columns:
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

    def propagate(self, fill_empty: bool = False) -> List[str]:
        """Propagate slot values from the set level down to individual records.

        Propagation, as defined by the SSSOM specification, is the process by which the values of
        so-called "propagatable slots" in the set metadata are moved to the corresponding slots in
        each individual mapping records.

        Propagation of a slot is only allowed iff no individual records already have a value for
        that slot.

        :param fill_empty: If True, propagation of a slot is allowed even if some individual records
            already have a value for that slot. The set-level value will be propagated to all the
            records for which the slot is empty. Note that (1) this is not spec-compliant behaviour,
            and (2) this makes the operation non-reversible by a subsequent condensation.

        :returns: The list of slots that were effectively propagated.
        """
        return propagate_condensed_slots(self.df, self.metadata, fill_empty)

    def condense(self) -> List[str]:
        """Condense record-level slot values to the set whenever possible.

        Condensation is the opposite of propagation. It is the process by which the values of
        so-called "propagatable" slots found in individual mapping records are moved to the
        corresponding slots in the set metadata.

        Condensation of a slot is only allowed iff (1) all records have the same value for that slot
        and (2) the slot does not already have a different value in the set metadata.

        :returns: The list of slots that were effectively condensed.
        """
        schema = SSSOMSchemaView()
        condensed = []

        for slot in schema.propagatable_slots:
            if slot not in self.df.columns:  # Nothing to condense
                continue
            values = self.df[slot].unique()
            if len(values) > 1:
                # Different values across the records, cannot condense
                continue

            if schema.view.get_slot(slot).multivalued:
                value = values[0].split("|")
            else:
                value = values[0]

            if slot in self.metadata:
                if self.metadata[slot] != value:
                    logging.warning(
                        f"Not condensing slot '{slot}' because it already has a different value in the set metadata."
                    )
                    continue
                # No need to set the condensed value in the set metadata as it
                # is already there, but we must still remove the column from
                # the dataframe
                condensed.append(slot)
            else:
                self.metadata[slot] = value
                condensed.append(slot)

        self.df.drop(columns=condensed, inplace=True)
        return condensed

    def infer_cardinality(self, scope: Optional[List[str]] = None) -> None:
        """Infer cardinality values in the set.

        This method will automatically fill the `mapping_cardinality` slot for all records in the
        set, overwriting any pre-existing values.

        See
        <https://mapping-commons.github.io/sssom/spec-model/#mapping-cardinality-and-cardinality-scope>
        for more information about cardinality computation,
        <https://mapping-commons.github.io/sssom/spec-model/#literal-mappings> for how to deal with
        literal mapping records, and
        <https://mapping-commons.github.io/sssom/spec-model/#representing-unmapped-entities> for how
        to deal with mapping records involving `sssom:NoTermFound`.

        :param scope: A list of slot names that defines the subset of the records in which
            cardinality will be computed. For example, with a scope of `['predicate_id']`, for any
            given record the cardinality will be computed relatively to the subset of records that
            have the same predicate. The default is an empty list, meaning that cardinality is
            computed relatively to the entire set of records.
        """
        if scope is None:
            scope = []

        #: Unique subjects for any given object
        subjects_by_object: defaultdict[str, set[str]] = defaultdict(set)
        #: Unique objects for any given subject
        objects_by_subject: defaultdict[str, set[str]] = defaultdict(set)

        schema = SSSOMSchemaView()
        unknown_slots = [slot for slot in scope if slot not in schema.mapping_slots]
        if len(unknown_slots) > 0:
            logging.warning(f"Ignoring invalid slot name(s): {unknown_slots}.")
            scope = list(set(scope) - set(unknown_slots))

        # Helper function to transform a row into a string that represents
        # a subject (or object) in a given scope; `side` is either `subject`
        # or `object`.
        def _to_string(row: pd.Series, side: str) -> str:
            # We prepend a one-letter code (`L` or `E`) to the actual subject
            # or object so that literal and non-literal mapping records are
            # always distinguishable and can be counted separately.
            if row.get(f"{side}_type") == "rdfs literal":
                s = "L\0" + (row.get(f"{side}_label") or "")
            else:
                s = "E\0" + (row.get(f"{side}_id") or "")
            for slot in scope:
                s += "\0" + (row.get(slot) or "")
            return s

        # We iterate over the records a first time to collect the different
        # objects mapped to each subject and vice versa
        for _, row in self.df.iterrows():
            if row.get(SUBJECT_ID) == NO_TERM_FOUND or row.get(OBJECT_ID) == NO_TERM_FOUND:
                # Mappings to sssom:NoTermFound are ignored for cardinality computations
                continue

            subj = _to_string(row, "subject")
            obj = _to_string(row, "object")

            subjects_by_object[obj].add(subj)
            objects_by_subject[subj].add(obj)

        # Second iteration to compute the actual cardinality values. Since we
        # must not modify a row while we are iterating over the dataframe, we
        # collect the values in a separate array.
        cards = []
        for _, row in self.df.iterrows():
            # Special cases involving sssom:NoTermFound on either side
            if row.get(SUBJECT_ID) == NO_TERM_FOUND:
                if row.get(OBJECT_ID) == NO_TERM_FOUND:
                    cards.append("0:0")
                else:
                    cards.append("0:1")
            elif row.get(OBJECT_ID) == NO_TERM_FOUND:
                cards.append("1:0")
            else:
                # General case
                n_subjects = len(subjects_by_object[_to_string(row, "object")])
                n_objects = len(objects_by_subject[_to_string(row, "subject")])

                if n_subjects == 1:
                    if n_objects == 1:
                        cards.append("1:1")
                    else:
                        cards.append("1:n")
                else:
                    if n_objects == 1:
                        cards.append("n:1")
                    else:
                        cards.append("n:n")

        # Add the computed values to the dataframe
        self.df[MAPPING_CARDINALITY] = cards
        if len(scope) > 0:
            self.df[CARDINALITY_SCOPE] = "|".join(scope)
        else:
            # No scope, so remove any pre-existing "cardinality_scope" column
            self.df.drop(columns=CARDINALITY_SCOPE, inplace=True, errors="ignore")

    def get_compatible_version(self) -> str:
        """Get the minimum version of SSSOM this set is compatible with."""
        schema = SSSOMSchemaView()
        versions: Set[Tuple[int, int]] = set()

        # First get the minimum versions required by the slots present
        # in the set; this is entirely provided by the SSSOM model.
        for slot in self.metadata.keys():
            version = schema.get_minimum_version(slot, "mapping set")
            if version is not None:
                versions.add(version)
        for slot in self.df.columns:
            version = schema.get_minimum_version(slot, "mapping")
            if version is not None:
                versions.add(version)

        # Then take care of enum values
        for new_enum_value in schema.get_new_enum_values():
            for slot in new_enum_value.slots:
                if self.metadata.get(slot) == new_enum_value.value or (
                    slot in self.df.columns and new_enum_value.value in self.df[slot].values
                ):
                    versions.add(new_enum_value.added_in)

        # Get the highest of the accumulated versions.
        return ".".join([str(i) for i in max(versions)])

    def enforce_version(
        self, version: str, strict: bool = False, inplace: bool = False
    ) -> "MappingSetDataFrame":
        """Ensure the set is compliant with a given version of the SSSOM specification.

        This method will forcefully remove any slot or enum value that
        is not defined in the specified version of the specification.

        :param version: The targeted version of the specification, as a
                        string of the form `X.Y`.
        :param strict: If `True`, unknown slots will be removed as well,
                       unless they are properly declared as extensions.
        :param inplace: if `True`, the method will modify and return the
                        set it has been called upon. The default is to
                        leave that set untouched and to return a
                        modified copy.
        :return: A set that is compliant with the requested version of
                 the SSSOM specification.
        """
        if inplace:
            msdf = self
        else:
            msdf = MappingSetDataFrame(df=self.df.copy(), metadata=self.metadata.copy())

        schema = SSSOMSchemaView()
        target_version = parse_sssom_version(version)
        defined_extensions = [
            ext.get(EXTENSION_SLOT_NAME) for ext in msdf.metadata.get(EXTENSION_DEFINITIONS, [])
        ]

        # Helper method to decide whether to keep or discard a slot
        def _keep(name: str, version: Optional[Tuple[int, int]]) -> bool:
            if version is not None:
                # This is a known slot, keep if compatible with target version
                return version <= target_version
            elif strict:
                # Unknown slot in strict mode, keep only if declared as an extension
                return name in defined_extensions
            else:
                # Unknown slot in non-strict mode, always keep
                return True

        # First the mapping set slots
        to_remove = [
            name
            for name in msdf.metadata.keys()
            if not _keep(name, schema.get_minimum_version(name, "mapping set"))
        ]
        for new_enum_value in schema.get_new_enum_values(after=target_version):
            for slot in new_enum_value.slots:
                if msdf.metadata.get(slot) == new_enum_value.value:
                    to_remove.append(slot)
        for slot in to_remove:
            msdf.metadata.pop(slot)

        # Then the individual mapping record slots
        to_remove = [
            name
            for name in msdf.df.columns
            if not _keep(name, schema.get_minimum_version(name, "mapping"))
        ]
        msdf.df.drop(columns=to_remove, inplace=True)
        for new_enum_value in schema.get_new_enum_values(after=target_version):
            for slot in new_enum_value.slots:
                if slot in msdf.df.columns:
                    msdf.df.loc[msdf.df[slot] == new_enum_value.value, slot] = ""

        return msdf


def _standardize_curie_or_iri(curie_or_iri: str, *, converter: Converter) -> str:
    """Standardize a CURIE or IRI, returning the original if not possible.

    :param curie_or_iri: Either a string representing a CURIE or an IRI

    :returns: A standardized CURIE or IRI

        - If the string represents an IRI, tries to standardize it. If not possible, returns the
          original value
        - If the string represents a CURIE, tries to standardize it. If not possible, returns the
          original value
        - Otherwise, return the original value
    """
    return converter.compress_or_standardize(curie_or_iri, passthrough=True)


def _standardize_metadata(
    converter: Converter, metadata: Dict[str, Any], *, raise_on_invalid: bool = False
) -> None:
    schema_object = _get_sssom_schema_object()
    slots_dict = schema_object.dict["slots"]

    # remove all falsy values. This has to be
    # done this way and not by making a new object
    # since we work in place
    for k, v in list(metadata.items()):
        if not k or not v:
            del metadata[k]

    for key, value in metadata.items():
        slot_metadata = slots_dict.get(key)
        if slot_metadata is None:
            text = f"invalid metadata key {key}"
            if raise_on_invalid:
                raise ValueError(text)
            logging.warning(text)
            continue
        if slot_metadata["range"] != "EntityReference":
            continue
        if is_multivalued_slot(key):
            if isinstance(value, str):
                metadata[key] = [
                    _standardize_curie_or_iri(v.strip(), converter=converter)
                    for v in value.split("|")
                ]
            elif isinstance(value, list):
                metadata[key] = [_standardize_curie_or_iri(v, converter=converter) for v in value]
            else:
                raise TypeError(f"{key} requires either a string or a list, got: {value}")
        elif isinstance(value, list):
            print("here")
            if len(value) > 1:
                raise TypeError(
                    f"value for {key} should have been a single value, but got a list: {value}"
                )
            print("also here")
            # note that the scenario len(value) == 0 is already
            # taken care of by the "if not value:" line above
            metadata[key] = _standardize_curie_or_iri(value[0], converter=converter)
        else:
            metadata[key] = _standardize_curie_or_iri(value, converter=converter)


@dataclass
class EntityPair:
    """A tuple of entities.

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
    """Represents a difference between two mapping sets.

    Currently this is limited to diffs at the level of entity-pairs. For example, if file1 has A
    owl:equivalentClass B, and file2 has A skos:closeMatch B, this is considered a mapping in
    common.
    """

    unique_tuples1: Set[EntityPair]
    unique_tuples2: Set[EntityPair]
    common_tuples: Set[EntityPair]

    combined_dataframe: pd.DataFrame
    """Dataframe that combines with left and right dataframes with information injected into the comment column"""


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows with same S/P/O and combines confidence."""
    df2 = df.groupby([SUBJECT_ID, PREDICATE_ID, OBJECT_ID])[CONFIDENCE].apply(max).reset_index()
    return df2


def sort_sssom(df: pd.DataFrame) -> pd.DataFrame:
    """Sort SSSOM by columns.

    :param df: SSSOM DataFrame to be sorted.

    :returns: Sorted SSSOM DataFrame
    """
    df.sort_values(by=sorted(df.columns), ascending=False, inplace=True)
    return df


def filter_redundant_rows(df: pd.DataFrame, ignore_predicate: bool = False) -> pd.DataFrame:
    """Remove rows if there is another row with same S/O and higher confidence.

    :param df: Pandas DataFrame to filter
    :param ignore_predicate: If true, the predicate_id column is ignored, defaults to False

    :returns: Filtered pandas DataFrame
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
    if not df.empty:
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
                    lambda x: x[CONFIDENCE]
                    >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID], x[PREDICATE_ID])],
                    axis=1,
                )
            ]
    # We are preserving confidence = NaN rows without making assumptions.
    # This means that there are potential duplicate mappings

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


def get_row_based_on_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Get row based on hierarchy of predicates.

    The hierarchy is as follows:

    1. owl:equivalentClass
    2. owl:equivalentProperty
    3. rdfs:subClassOf
    4. rdfs:subPropertyOf
    5. owl:sameAs
    6. skos:exactMatch
    7. skos:closeMatch
    8. skos:broadMatch
    9. skos:narrowMatch
    10. oboInOwl:hasDbXref
    11. skos:relatedMatch
    12. rdfs:seeAlso

    :param df: Dataframe containing multiple predicates for same subject and object.

    :returns: Dataframe with a single row which ranks higher in the hierarchy.

    :raises KeyError: if no rows are available
    """
    for pred in PREDICATE_LIST:
        hierarchical_df = df[df[PREDICATE_ID] == pred]
        if not hierarchical_df.empty:
            return hierarchical_df
    raise KeyError


def assign_default_confidence(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign :data:`numpy.nan` to confidence that are blank.

    :param df: SSSOM DataFrame

    :returns: A Tuple consisting of the original DataFrame and dataframe consisting of empty
        confidence values.
    """
    # Get rows having numpy.NaN as confidence
    if df is None:
        ValueError("DataFrame cannot be empty to 'assign_default_confidence'.")
    new_df = df.copy()
    if CONFIDENCE not in new_df.columns:
        new_df[CONFIDENCE] = 0.0  # np.nan
        nan_df = pd.DataFrame(columns=new_df.columns)
    else:
        new_df = df[~df[CONFIDENCE].isna()]
        nan_df = df[df[CONFIDENCE].isna()]
    return new_df, nan_df


def remove_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where no match is found.

    :param df: Pandas DataFrame

    :returns: Pandas DataFrame with 'PREDICATE_ID' not 'noMatch'.

    .. todo:: https://github.com/OBOFoundry/SSSOM/issues/28
    """
    return df[df[PREDICATE_ID] != "noMatch"]


def create_entity(identifier: str, mappings: Dict[str, Any]) -> Uriorcurie:
    """Create an Entity object.

    :param identifier: Entity Id
    :param mappings: Mapping dictionary

    :returns: An Entity object
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

    .. warning::

        currently does not discriminate between mappings with different predicates
    """
    mappings1 = group_mappings(df1.copy())
    mappings2 = group_mappings(df2.copy())
    tuples1 = set(mappings1.keys())
    tuples2 = set(mappings2.keys())
    unique_tuples1 = tuples1.difference(tuples2)
    unique_tuples2 = tuples2.difference(tuples1)
    common_tuples = tuples1.intersection(tuples2)
    all_tuples = tuples1.union(tuples2)
    all_ids = set()
    for t in all_tuples:
        all_ids.update({t.subject_entity, t.object_entity})
    rows = []
    for t in unique_tuples1:
        for r in mappings1[t]:
            r[COMMENT] = "UNIQUE_1"
        rows += mappings1[t]
    for t in unique_tuples2:
        for r in mappings2[t]:
            r[COMMENT] = "UNIQUE_2"
        rows += mappings2[t]
    for t in common_tuples:
        new_rows = mappings1[t] + mappings2[t]
        for r in new_rows:
            r[COMMENT] = "COMMON_TO_BOTH"
        rows += new_rows
    # for r in rows:
    #    r['other'] = 'synthesized sssom file'
    d = MappingSetDiff(
        unique_tuples1=unique_tuples1,
        unique_tuples2=unique_tuples2,
        common_tuples=common_tuples,
        combined_dataframe=pd.DataFrame(rows).drop_duplicates(),
    )
    return d


def add_default_confidence(df: pd.DataFrame, confidence: float = np.nan) -> pd.DataFrame:
    """Add `confidence` column to DataFrame if absent and initializes to 0.95.

    If `confidence` column already exists, only fill in the None ones by 0.95.

    :param df: DataFrame whose `confidence` column needs to be filled.

    :returns: DataFrame with a complete `confidence` column.
    """
    if CONFIDENCE in df.columns:
        df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: confidence * x if x is not None else x)
        df[CONFIDENCE].fillna(float(confidence), inplace=True)
    else:
        df[CONFIDENCE] = float(confidence)

    return df


def dataframe_to_ptable(
    df: pd.DataFrame,
    *,
    inverse_factor: Optional[float] = None,
    default_confidence: Optional[float] = None,
) -> list[tuple[str, str, float, float, float, float]]:
    """Export a KBOOM table.

    :param df: Pandas DataFrame
    :param inverse_factor: Multiplier to (1 - confidence), defaults to 0.5
    :param default_confidence: Default confidence to be assigned if absent.

    :returns: List of rows

    :raises ValueError: Predicate value error
    :raises ValueError: Predicate type value error
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
        confidence: float = row[CONFIDENCE]
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
        rows.append((subject_id, object_id, *ps))
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
    :param reconcile: If reconcile=True, then dedupe(remove redundant lower confidence mappings) and
        reconcile (if msdf contains a higher confidence _negative_ mapping, then remove lower
        confidence positive one. If confidence is the same, prefer HumanCurated. If both
        HumanCurated, prefer negative mapping). Defaults to False.

    :returns: Merged MappingSetDataFrame.
    """
    # Propagate slots, inject source if possible
    source_injected = 0
    for msdf in msdfs:
        msdf.propagate()
        if MAPPING_SET_ID in msdf.metadata and MAPPING_SOURCE not in msdf.df.columns:
            msdf.df[MAPPING_SOURCE] = msdf.metadata[MAPPING_SET_ID]
            source_injected += 1

    columns = set([c for msdf in msdfs for c in msdf.df.columns])
    if source_injected > 1:
        # If we injected a mapping_source slot into each individual
        # record for at least two of the input sets, then we must ignore
        # that slot when attempting to remove duplicates below, because
        # the mere presence of that slot would cause two identical
        # records to appear different just because they come from
        # different sources (which they would not do if we had not
        # injected the mapping_source above).
        columns.remove(MAPPING_SOURCE)

    df_merged = reduce(
        lambda left, right: pd.concat([left, right], axis=0, ignore_index=True),
        [msdf.df for msdf in msdfs],
    ).drop_duplicates(ignore_index=True, subset=columns)

    converter = curies.chain([msdf.converter for msdf in msdfs])
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

    :returns: Pandas DataFrame with negations addressed

    :raises ValueError: If the dataframe is none after assigning default confidence
    """
    # 1. Mappings in mapping1 trump mappings in mapping2 (if mapping2 contains a conflicting mapping in mapping1,
    #     the one in mapping1 is preserved).
    # 2. Reconciling means two things
    #     [i] if the same s,p,o (subject_id, object_id, predicate_id) is present multiple times,
    #         only preserve the highest confidence one. If confidence is same, rule 1 (above) applies.
    #     [ii] If s,!p,o and s,p,o , then prefer higher confidence and remove the other.
    #             If same confidence prefer "HumanCurated" .If same again prefer negative.
    # 3. Prefixes:
    #     [i] if there is the same prefix in mapping1 as in mapping2, and the prefix URL is different,
    #     throw an error and fail hard
    #         else just merge the two prefix maps
    # 4. Metadata: same as rule 1.
    #
    # #1; #2(i) #3 and $4 are taken care of by 'filtered_merged_df' Only #2(ii) should be performed here.

    # Handle DataFrames with no 'confidence' column (basically adding a np.nan to all non-numeric confidences)
    confidence_in_original = CONFIDENCE in df.columns
    df, nan_df = assign_default_confidence(df)

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
    max_confidence_df = combined_normalized_subset.groupby(
        TRIPLES_IDS, as_index=False
    )[  # type:ignore
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

        reconciled_df_subset.loc[match_condition_2[match_condition_2].index, PREDICATE_MODIFIER] = (
            row_2[PREDICATE_MODIFIER]
        )

    if PREDICATE_MODIFIER in reconciled_df_subset.columns:
        reconciled_df_subset[PREDICATE_MODIFIER] = reconciled_df_subset[PREDICATE_MODIFIER].fillna(
            ""
        )

    # .fillna(df) towards the end fills an empty value
    # with a corresponding value from df.
    # This needs to happen because the columns in df
    # not in reconciled_df_subset will be NaN otherwise
    # which is incorrect.
    reconciled_df: pd.DataFrame = df.merge(
        reconciled_df_subset, how="right", on=list(reconciled_df_subset.columns)
    ).fillna(df)

    if nan_df.empty:
        return_df = reconciled_df
    else:
        return_df = reconciled_df.append(nan_df).drop_duplicates()  # type:ignore

    if not confidence_in_original:
        return_df = return_df.drop(columns=[CONFIDENCE], axis=1)

    return return_df


def propagate_condensed_slots(
    df: pd.DataFrame, metadata: MetadataType, fill_empty: bool = False
) -> List[str]:
    """Propagate slot values from the set level down to individual records.

    This function performs the same operation as the `MappingSetDataFrame#propagate()`
    method. It is intended to allow propagating a mapping set before an instance of
    the `MappingSetDataFrame` class can be obtained.

    :param df: The DataFrame into which values should be propagated.
    :param metadata: The dictionary of set-level metadata.
    :param fill_empty: If True, propagation of a slot is allowed even if some individual records
        already have a value for that slot. The set-level value will be propagated to all the
        records for which the slot is empty. Note that (1) this is not spec-compliant behaviour,
        and (2) this makes the operation non-reversible by a subsequent condensation.

    :returns: The list of slots that were effectively propagated.
    """
    schema = SSSOMSchemaView()
    propagated = []

    for slot in schema.propagatable_slots:
        if slot not in metadata:  # Nothing to propagate
            continue
        is_present = slot in df.columns
        if is_present and not fill_empty:
            logging.warning(
                f"Not propagating value for '{slot}' because the slot is already set on individual records."
            )
            continue

        if schema.view.get_slot(slot).multivalued:
            value = "|".join(metadata.pop(slot))
        else:
            value = metadata.pop(slot)

        if is_present:
            df.loc[df[slot].eq("") | df[slot].isna(), slot] = value
        else:
            df[slot] = value
        propagated.append(slot)

    return propagated


ExtensionLiteral = Literal["tsv", "csv"]


def get_file_extension(file: PathOrIO) -> Optional[ExtensionLiteral]:
    """Get file extension.

    :param file: File path

    :returns: format of the file passed, default tsv
    """
    if not isinstance(file, (str, Path)):
        if not hasattr(file, "name"):
            logging.debug("cannot guess format for object without name: %s", file)
            return None
        file = Path(file.name)
    elif isinstance(file, str):
        file = Path(file)

    filename = file.name.removesuffix(".gz")
    if filename.endswith(".tsv"):
        return "tsv"
    elif filename.endswith(".csv"):
        return "csv"
    else:
        logging.debug("cannot guess format for %s", filename)
        return None


def _extract_global_metadata(msdoc: MappingSetDocument) -> MetadataType:
    """Extract metadata.

    :param msdoc: MappingSetDocument object

    :returns: Dictionary containing metadata
    """
    meta = {}
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

    :returns: MappingSetDataFrame object
    """
    return MappingSetDataFrame.from_mapping_set_document(doc)


def get_dict_from_mapping(map_obj: Union[Any, Dict[str, Any], SSSOM_Mapping]) -> dict[str, Any]:
    """Get information for linkml objects (MatchTypeEnum, PredicateModifierEnum) from the Mapping object and return the dictionary form of the object.

    :param map_obj: Mapping object

    :returns: Dictionary
    """
    map_dict = {}
    sssom_schema_object = _get_sssom_schema_object()
    for property in map_obj:
        mapping_property = map_obj[property]
        if mapping_property is None:
            map_dict[property] = np.nan if property in sssom_schema_object.double_slots else ""
            continue

        slot_of_interest = sssom_schema_object.slots[property]
        is_enum = slot_of_interest["range"] in sssom_schema_object.mapping_enum_keys  # type:ignore

        # Check if the mapping_property is a list
        if isinstance(mapping_property, list):
            # If the property is an enumeration and it allows multiple values
            if is_enum and slot_of_interest["multivalued"]:  # type:ignore
                # Join all the enum values into a string separated by '|'
                map_dict[property] = "|".join(
                    enum_value.code.text for enum_value in mapping_property
                )
            else:
                # If the property is not an enumeration or doesn't allow multiple values,
                # join all the values into a string separated by '|'
                map_dict[property] = "|".join(enum_value for enum_value in mapping_property)
        elif is_enum:
            # Assign the text of the enumeration code to the property in the dictionary
            map_dict[property] = mapping_property.code.text
        else:
            # If the mapping_property is neither a list nor an enumeration,
            # assign the value directly to the property in the dictionary
            map_dict[property] = mapping_property

    return map_dict


CURIE_PATTERN = r"[A-Za-z0-9_.]+[:][A-Za-z0-9_]"
CURIE_RE = re.compile(CURIE_PATTERN)


def _is_curie(string: str) -> bool:
    """Check if the string is a CURIE."""
    if string and isinstance(string, str):
        return bool(CURIE_RE.match(string))
    else:
        return False


def _is_iri(string: str) -> bool:
    """Check if the string is an IRI."""
    if string and isinstance(string, str):
        return bool(validators.url(string))
    else:
        return False


def get_prefix_from_curie(curie: str) -> str:
    """Get the prefix from a CURIE."""
    if _is_curie(curie):
        return ReferenceTuple.from_curie(curie).prefix
    else:
        return ""


def get_prefixes_used_in_table(df: pd.DataFrame) -> Set[str]:
    """Get a list of prefixes used in CURIEs in key feature columns in a dataframe."""
    prefixes = set(SSSOM_BUILT_IN_PREFIXES)
    if df.empty:
        return prefixes
    sssom_schema_object = _get_sssom_schema_object()
    entity_reference_slots = sssom_schema_object.entity_reference_slots & set(df.columns)
    new_prefixes = {
        ReferenceTuple.from_curie(row).prefix
        for col in entity_reference_slots
        for row in df[col]
        if not _is_iri(row) and _is_curie(row)
        # we don't use the converter here since get_prefixes_used_in_table
        # is often used to identify prefixes that are not properly registered
        # in the converter
    }

    prefixes.update(new_prefixes)

    return prefixes


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
    features: Optional[list[str]] = None,
    require_all_prefixes: bool = False,
) -> pd.DataFrame:
    """Filter out rows which contains a CURIE with a prefix in the filter_prefixes list.

    :param df: Pandas DataFrame of SSSOM Mapping
    :param filter_prefixes: List of prefixes
    :param features: List of dataframe column names dataframe to consider
    :param require_all_prefixes: If True, all prefixes must be present in a row to be filtered out

    :returns: Pandas Dataframe
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
    features: Optional[list[str]] = None,
    require_all_prefixes: bool = True,
) -> pd.DataFrame:
    """Filter out rows which do NOT contain a CURIE with a prefix in the filter_prefixes list.

    :param df: Pandas DataFrame of SSSOM Mapping
    :param filter_prefixes: List of prefixes
    :param features: List of dataframe column names dataframe to consider
    :param require_all_prefixes: If True, all prefixes must be present in a row to be filtered out

    :returns: Pandas Dataframe
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

    :returns: Slot is multivalued or no
    """
    return slot in _get_sssom_schema_object().multivalued_slots


class PrefixReconciliation(TypedDict):
    """Reconciliation dictionaries."""

    prefix_synonyms: dict[str, str]
    prefix_expansion_reconciliation: dict[str, str]


def reconcile_prefix_and_data(
    msdf: MappingSetDataFrame, prefix_reconciliation: PrefixReconciliation
) -> MappingSetDataFrame:
    """Reconciles prefix_map and translates CURIE switch in dataframe.

    :param msdf: Mapping Set DataFrame.
    :param prefix_reconciliation: Prefix reconcilation dictionary from a YAML file

    :returns: Mapping Set DataFrame with reconciled prefix_map and data.

    This method is build on :func:`curies.remap_curie_prefixes` and :func:`curies.rewire`. Note that
    if you want to overwrite a CURIE prefix in the Bioregistry extended prefix map, you need to
    provide a place for the old one to go as in ``{"geo": "ncbi.geo", "geogeo": "geo"}``. Just doing
    ``{"geogeo": "geo"}`` would not work since `geo` already exists.
    """
    # Discussion about this found here:
    # https://github.com/mapping-commons/sssom-py/issues/216#issue-1171701052
    converter = msdf.converter
    converter = curies.remap_curie_prefixes(converter, prefix_reconciliation["prefix_synonyms"])
    converter = curies.rewire(converter, prefix_reconciliation["prefix_expansion_reconciliation"])
    msdf.converter = converter
    msdf.standardize_references()
    return msdf


def sort_df_rows_columns(
    df: pd.DataFrame, by_columns: bool = True, by_rows: bool = True
) -> pd.DataFrame:
    """Canonical sorting of DataFrame columns.

    :param df: Pandas DataFrame with random column sequence.
    :param by_columns: Boolean flag to sort columns canonically.
    :param by_rows: Boolean flag to sort rows by column #1 (ascending order).

    :returns: Pandas DataFrame columns sorted canonically.
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

    :returns: List of all prefixes.

    :raises ValidationError: If slot is wrong.
    :raises ValidationError: If slot is wrong.
    """
    # FIXME investigate the logic for this function -
    #  some of the falsy checks don't make sense
    if not msdf.metadata or msdf.df.empty:
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
    msdf: MappingSetDataFrame, meta: MetadataType, replace_multivalued: bool = False
) -> MappingSetDataFrame:
    """Augment metadata with parameters passed.

    :param msdf: MappingSetDataFrame (MSDF) object.
    :param meta: Dictionary that needs to be added/updated to the metadata of the MSDF.
    :param replace_multivalued: Multivalued slots should be replaced or not, defaults to False.

    :returns: MSDF with updated metadata.

    :raises ValueError: If type of slot is neither str nor list.
    """
    # TODO this now partially redundant of the MSDF built-in standardize functionality
    are_params_slots(meta)
    if not msdf.metadata:
        return msdf
    for k, v in meta.items():
        # If slot is multivalued, add to list.
        if k in _get_sssom_schema_object().multivalued_slots and not replace_multivalued:
            tmp_value: list[str]
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


def are_params_slots(params: dict[str, Any]) -> bool:
    """Check if parameters conform to the slots in MAPPING_SET_SLOTS.

    :param params: Dictionary of parameters.

    :returns: True/False

    :raises ValueError: If params are not slots.
    """
    empty_params = {k for k, v in params.items() if v is None or v == ""}
    if len(empty_params) > 0:
        logging.info(f"Parameters: {empty_params} has(ve) no value.")

    legit_params = all(p in _get_sssom_schema_object().mapping_set_slots for p in params.keys())
    if not legit_params:
        invalids = [p for p in params if p not in _get_sssom_schema_object().mapping_set_slots]
        raise ValueError(
            f"The params are invalid: {invalids}. Should be any of the following: {_get_sssom_schema_object().mapping_set_slots}"
        )
    return True


def invert_mappings(
    df: pd.DataFrame,
    subject_prefix: Optional[str] = None,
    merge_inverted: bool = True,
    update_justification: bool = True,
    predicate_invert_dictionary: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Switching subject and objects based on their prefixes and adjusting predicates accordingly.

    :param df: Pandas dataframe.
    :param subject_prefix: Prefix of subjects desired.
    :param merge_inverted: If True (default), add inverted dataframe to input else, just return
        inverted data.
    :param update_justification: If True (default), the justification is updated to
        "sempav:MappingInversion", else it is left as it is.
    :param predicate_invert_dictionary: YAML file providing the inverse mapping for predicates.

    :returns: Pandas dataframe with all subject IDs having the same prefix.
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
        # Filter rows where 'SUBJECT_ID' starts with the prefix but 'OBJECT_ID' does not
        prefixed_subjects_df = pd.DataFrame(
            non_predicate_modified_df[
                (
                    non_predicate_modified_df[SUBJECT_ID].str.startswith(subject_prefix + ":")
                    & ~non_predicate_modified_df[OBJECT_ID].str.startswith(subject_prefix + ":")
                )
            ]
        )

        # Filter rows where 'SUBJECT_ID' does not start with the prefix but 'OBJECT_ID' does
        non_prefix_subjects_df = pd.DataFrame(
            non_predicate_modified_df[
                (
                    ~non_predicate_modified_df[SUBJECT_ID].str.startswith(subject_prefix + ":")
                    & non_predicate_modified_df[OBJECT_ID].str.startswith(subject_prefix + ":")
                )
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
    inverted_df = sort_df_rows_columns(inverted_df, by_rows=False)
    inverted_df[PREDICATE_ID] = inverted_df[PREDICATE_ID].map(predicate_invert_map)
    if update_justification:
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


def _invert_column_names(column_names: list[X], columns_invert_map: Mapping[X, Y]) -> dict[X, Y]:
    """Return a dictionary for column renames in pandas DataFrame."""
    return {x: columns_invert_map[x] for x in column_names}


def safe_compress(uri: str, converter: Converter) -> str:
    """Parse a CURIE from an IRI.

    :param uri: The URI to parse. If this is already a CURIE, return directly.
    :param converter: Converter used for compression

    :returns: A CURIE
    """
    return converter.compress_or_standardize(uri, strict=True)


def pandas_set_no_silent_downcasting(no_silent_downcasting: bool = True) -> None:
    """Set pandas future.no_silent_downcasting option. Context https://github.com/pandas-dev/pandas/issues/57734."""
    try:
        pd.set_option("future.no_silent_downcasting", no_silent_downcasting)
    except KeyError:
        # Option does not exist in this version of pandas
        pass
