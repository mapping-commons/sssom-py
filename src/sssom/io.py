"""I/O utilities for SSSOM."""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Union

import pandas as pd
from curies import Converter
from pansql import sqldf

from sssom.validators import validate

from .constants import (
    PREFIX_MAP_MODE_MERGED,
    PREFIX_MAP_MODE_METADATA_ONLY,
    PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY,
    SchemaValidationType,
)
from .context import (
    add_built_in_prefixes_to_prefix_map,
    get_default_metadata,
    set_default_license,
    set_default_mapping_set_id,
)
from .parsers import get_parsing_function, parse_sssom_table, split_dataframe
from .typehints import Metadata
from .util import (
    MappingSetDataFrame,
    are_params_slots,
    augment_metadata,
    is_curie,
    is_iri,
    raise_for_bad_path,
    raise_for_bad_prefix_map_mode,
    read_metadata,
)
from .writers import get_writer_function, write_table, write_tables


def convert_file(
    input_path: str,
    output: TextIO,
    output_format: Optional[str] = None,
) -> None:
    """Convert a file from one format to another.

    :param input_path: The path to the input SSSOM tsv file
    :param output: The path to the output file. If none is given, will default to using stdout.
    :param output_format: The format to which the SSSOM TSV should be converted.
    """
    raise_for_bad_path(input_path)
    doc = parse_sssom_table(input_path)
    write_func, fileformat = get_writer_function(output_format=output_format, output=output)
    # TODO cthoyt figure out how to use protocols for this
    write_func(doc, output, serialisation=fileformat)  # type:ignore


def parse_file(
    input_path: str,
    output: TextIO,
    input_format: Optional[str] = None,
    metadata_path: Optional[str] = None,
    prefix_map_mode: Optional[str] = None,
    clean_prefixes: bool = True,
    strict_clean_prefixes: bool = True,
    embedded_mode: bool = True,
    mapping_predicate_filter: tuple = None,
) -> None:
    """Parse an SSSOM metadata file and write to a table.

    :param input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
    :param output: The path to the output file.
    :param input_format: The string denoting the input format.
    :param metadata_path: The path to a file containing the sssom metadata (including prefix_map)
        to be used during parse.
    :param prefix_map_mode: Defines whether the prefix map in the metadata should be extended or replaced with
        the SSSOM default prefix map. Must be one of metadata_only, sssom_default_only, merged
    :param clean_prefixes: If True (default), records with unknown prefixes are removed from the SSSOM file.
    :param strict_clean_prefixes: If True (default), clean_prefixes() will be in strict mode.
    :param embedded_mode:If True (default), the dataframe and metadata are exported in one file (tsv), else two separate files (tsv and yaml).
    :param mapping_predicate_filter: Optional list of mapping predicates or filepath containing the same.
    """
    raise_for_bad_path(input_path)
    metadata = get_metadata_and_prefix_map(
        metadata_path=metadata_path, prefix_map_mode=prefix_map_mode
    )
    parse_func = get_parsing_function(input_format, input_path)
    mapping_predicates = None
    # Get list of predicates of interest.
    if mapping_predicate_filter:
        mapping_predicates = get_list_of_predicate_iri(
            mapping_predicate_filter, metadata.prefix_map
        )

    # if mapping_predicates:
    doc = parse_func(
        input_path,
        prefix_map=metadata.prefix_map,
        meta=metadata.metadata,
        mapping_predicates=mapping_predicates,
    )
    # else:
    #     doc = parse_func(
    #         input_path,
    #         prefix_map=metadata.prefix_map,
    #         meta=metadata.metadata,
    #     )
    if clean_prefixes:
        # We do this because we got a lot of prefixes from the default SSSOM prefixes!
        doc.clean_prefix_map(strict=strict_clean_prefixes)
    write_table(doc, output, embedded_mode)


def validate_file(input_path: str, validation_types: List[SchemaValidationType]) -> None:
    """Validate the incoming SSSOM TSV according to the SSSOM specification.

    :param input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
    :param validation_types: A list of validation types to run.
    """
    # Two things to check:
    # 1. All prefixes in the DataFrame are define in prefix_map
    # 2. All columns in the DataFrame abide by sssom-schema.
    msdf = parse_sssom_table(file_path=input_path)
    validate(msdf=msdf, validation_types=validation_types)


def split_file(input_path: str, output_directory: Union[str, Path]) -> None:
    """Split an SSSOM TSV by prefixes and relations.

    :param  input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
    :param output_directory: The directory to which the split file should be exported.
    """
    raise_for_bad_path(input_path)
    msdf = parse_sssom_table(input_path)
    splitted = split_dataframe(msdf)
    write_tables(splitted, output_directory)


def _get_prefix_map(metadata: Metadata, prefix_map_mode: str = None):
    if prefix_map_mode is None:
        prefix_map_mode = PREFIX_MAP_MODE_METADATA_ONLY

    raise_for_bad_prefix_map_mode(prefix_map_mode=prefix_map_mode)

    prefix_map = metadata.prefix_map

    if prefix_map_mode != PREFIX_MAP_MODE_METADATA_ONLY:
        default_metadata: Metadata = get_default_metadata()
        if prefix_map_mode == PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY:
            prefix_map = default_metadata.prefix_map
        elif prefix_map_mode == PREFIX_MAP_MODE_MERGED:
            for prefix, uri_prefix in default_metadata.prefix_map.items():
                if prefix not in prefix_map:
                    prefix_map[prefix] = uri_prefix
    return prefix_map


def get_metadata_and_prefix_map(
    metadata_path: Optional[str] = None, prefix_map_mode: Optional[str] = None
) -> Metadata:
    """
    Load SSSOM metadata from a file, and then augments it with default prefixes.

    :param metadata_path: The metadata file in YAML format
    :param prefix_map_mode: one of metadata_only, sssom_default_only, merged
    :return: a prefix map dictionary and a metadata object dictionary
    """
    if metadata_path is None:
        return get_default_metadata()

    metadata = read_metadata(metadata_path)
    prefix_map = _get_prefix_map(metadata=metadata, prefix_map_mode=prefix_map_mode)

    m = Metadata(prefix_map=prefix_map, metadata=metadata.metadata)
    m = set_default_mapping_set_id(m)
    m = set_default_license(m)
    return m


def get_list_of_predicate_iri(predicate_filter: tuple, prefix_map: dict) -> list:
    """Return a list of IRIs for predicate CURIEs passed.

    :param predicate_filter: CURIE OR list of CURIEs OR file path containing the same.
    :param prefix_map: Prefix map of mapping set (possibly) containing custom prefix:IRI combination.
    :return: A list of IRIs.
    """
    pred_filter_list = list(predicate_filter)
    iri_list = []
    for p in pred_filter_list:
        p_iri = extract_iri(p, prefix_map)
        if p_iri:
            iri_list.extend(p_iri)
    return list(set(iri_list))


def extract_iri(input: str, prefix_map: Dict[str, str]) -> List[str]:
    """
    Recursively extracts a list of IRIs from a string or file.

    :param input: CURIE OR list of CURIEs OR file path containing the same.
    :param prefix_map: Prefix map of mapping set (possibly) containing custom prefix:IRI combination.
    :return: A list of IRIs.
    :rtype: list
    """
    if is_iri(input):
        return [input]
    elif is_curie(input):
        converter = Converter.from_prefix_map(prefix_map)
        p_iri = converter.expand(input)
        if p_iri:
            return [p_iri]

    elif os.path.isfile(input):
        pred_list = Path(input).read_text().splitlines()
        iri_list: List[str] = []
        for p in pred_list:
            p_iri = extract_iri(p, prefix_map)
            if p_iri:
                iri_list.extend(p_iri)
        return iri_list

    else:
        logging.warning(
            f"{input} is neither a valid curie, nor an IRI, nor a local file path, "
            f"skipped from processing."
        )
    return []


# def filter_file(input: str, prefix: tuple, predicate: tuple) -> MappingSetDataFrame:
#     """Filter mapping file based on prefix and predicates provided.

#     :param input: Input mapping file (tsv)
#     :param prefix: Prefixes to be retained.
#     :param predicate: Predicates to be retained.
#     :return: Filtered MappingSetDataFrame.
#     """
#     msdf: MappingSetDataFrame = parse_sssom_table(input)
#     prefix_map = msdf.prefix_map
#     df: pd.DataFrame = msdf.df
#     # Filter prefix_map
#     filtered_prefix_map = {
#         k: v for k, v in prefix_map.items() if k in prefix_map.keys() and k in prefix
#     }

#     filtered_predicates = {
#         k: v
#         for k, v in prefix_map.items()
#         if len([x for x in predicate if str(x).startswith(k)])
#         > 0  # use re.find instead.
#     }
#     filtered_prefix_map.update(filtered_predicates)
#     filtered_prefix_map = add_built_in_prefixes_to_prefix_map(filtered_prefix_map)

#     # Filter df based on predicates
#     predicate_filtered_df: pd.DataFrame = df.loc[
#         df[PREDICATE_ID].apply(lambda x: x in predicate)
#     ]

#     # Filter df based on prefix_map
#     prefix_keys = tuple(filtered_prefix_map.keys())
#     condition_subj = predicate_filtered_df[SUBJECT_ID].apply(
#         lambda x: str(x).startswith(prefix_keys)
#     )
#     condition_obj = predicate_filtered_df[OBJECT_ID].apply(
#         lambda x: str(x).startswith(prefix_keys)
#     )
#     filtered_df = predicate_filtered_df.loc[condition_subj & condition_obj]

#     new_msdf: MappingSetDataFrame = MappingSetDataFrame(
#         df=filtered_df, prefix_map=filtered_prefix_map, metadata=msdf.metadata
#     )
#     return new_msdf


def run_sql_query(query: str, inputs: List[str], output: TextIO) -> MappingSetDataFrame:
    """Run a SQL query over one or more SSSOM files.

    Each of the N inputs is assigned a table name df1, df2, ..., dfN

    Alternatively, the filenames can be used as table names - these are first stemmed
    E.g. ~/dir/my.sssom.tsv becomes a table called 'my'

    Example:
        sssom dosql -Q "SELECT * FROM df1 WHERE confidence>0.5 ORDER BY confidence" my.sssom.tsv

    Example:
        `sssom dosql -Q "SELECT file1.*,file2.object_id AS ext_object_id, file2.object_label AS ext_object_label \
        FROM file1 INNER JOIN file2 WHERE file1.object_id = file2.subject_id" FROM file1.sssom.tsv file2.sssom.tsv`

    :param query: Query to be executed over a pandas DataFrame (msdf.df).
    :param inputs: Input files that form the source tables for query.
    :param output: Output.
    :return: Filtered MappingSetDataFrame object.
    """
    n = 1
    new_msdf = MappingSetDataFrame()
    while len(inputs) >= n:
        fn = inputs[n - 1]
        msdf = parse_sssom_table(fn)
        df = msdf.df
        # df = parse(fn)
        globals()[f"df{n}"] = df
        tn = re.sub("[.].*", "", Path(fn).stem).lower()
        globals()[tn] = df
        n += 1

    new_df = sqldf(query)
    new_msdf.df = new_df
    new_msdf.prefix_map = add_built_in_prefixes_to_prefix_map(msdf.prefix_map)
    new_msdf.metadata = msdf.metadata
    write_table(new_msdf, output)
    return new_msdf


def filter_file(input: str, output: TextIO, **kwargs) -> MappingSetDataFrame:
    """Filter a dataframe by dynamically generating queries based on user input.

    e.g. sssom filter --subject_id x:% --subject_id y:% --object_id y:% --object_id z:% tests/data/basic.tsv

    yields the query:

    "SELECT * FROM df WHERE (subject_id LIKE 'x:%'  OR subject_id LIKE 'y:%')
     AND (object_id LIKE 'y:%'  OR object_id LIKE 'z:%') " and displays the output.

    :param input: DataFrame to be queried over.
    :param output: Output location.
    :param **kwargs: Filter options provided by user which generate queries (e.g.: --subject_id x:%).
    :raises ValueError: If parameter provided is invalid.
    :return: Filtered MappingSetDataFrame object.
    """
    params = {k: v for k, v in kwargs.items() if v}
    query = "SELECT * FROM df WHERE ("
    multiple_params = True if len(params) > 1 else False

    # Check if all params are legit
    input_df: pd.DataFrame = parse_sssom_table(input).df
    if not input_df.empty and len(input_df.columns) > 0:
        column_list = list(input_df.columns)
    else:
        raise ValueError(f"{input} is either not a SSSOM TSV file or an empty one.")
    legit_params = all(p in column_list for p in params)
    if not legit_params:
        invalids = [p for p in params if p not in column_list]
        raise ValueError(f"The params are invalid: {invalids}")

    for idx, (k, v) in enumerate(params.items(), start=1):
        query += k + " LIKE '" + v[0] + "' "
        if len(v) > 1:
            for idx2, exp in enumerate(v[1:]):
                query += " OR "
                query += k + " LIKE '" + exp + "'"
                if idx2 + 1 == len(v) - 1:
                    query += ") "
        else:
            query += ") "
        if multiple_params and idx != len(params):
            query += " AND ("
    return run_sql_query(query=query, inputs=[input], output=output)


def annotate_file(
    input: str, output: TextIO, replace_multivalued: bool = False, **kwargs
) -> MappingSetDataFrame:
    """Annotate a file i.e. add custom metadata to the mapping set.

    :param input: SSSOM tsv file to be queried over.
    :param output: Output location.
    :param replace_multivalued: Multivalued slots should be
        replaced or not, defaults to False
    :param **kwargs: Options provided by user
        which are added to the metadata (e.g.: --mapping_set_id http://example.org/abcd)
    :return: Annotated MappingSetDataFrame object.
    """
    params = {k: v for k, v in kwargs.items() if v}
    are_params_slots(params)
    input_msdf = parse_sssom_table(input)
    msdf = augment_metadata(input_msdf, params, replace_multivalued)
    write_table(msdf, output)
    return msdf
