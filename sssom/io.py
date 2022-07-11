"""I/O utilities for SSSOM."""

import logging
import os
from pathlib import Path
from typing import List, Optional, TextIO, Union

import pandas as pd
from bioregistry import get_iri

from sssom.validators import validate

from .constants import (
    OBJECT_ID,
    PREDICATE_ID,
    PREFIX_MAP_MODE_MERGED,
    PREFIX_MAP_MODE_METADATA_ONLY,
    PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY,
    SUBJECT_ID,
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
    :param output_format: The format to which the the SSSOM TSV should be converted.
    """
    raise_for_bad_path(input_path)
    doc = parse_sssom_table(input_path)
    write_func, fileformat = get_writer_function(
        output_format=output_format, output=output
    )
    # TODO cthoyt figure out how to use protocols for this
    write_func(doc, output, serialisation=fileformat)  # type:ignore


def parse_file(
    input_path: str,
    output: TextIO,
    input_format: Optional[str] = None,
    metadata_path: Optional[str] = None,
    prefix_map_mode: Optional[str] = None,
    clean_prefixes: bool = True,
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
        doc.clean_prefix_map()
    write_table(doc, output, embedded_mode)


def validate_file(
    input_path: str, validation_types: List[SchemaValidationType]
) -> None:
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


def extract_iri(input, prefix_map) -> list:
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
        p_iri = get_iri(input, prefix_map=prefix_map, use_bioregistry_io=False)
        if not p_iri:
            p_iri = get_iri(input)
        if p_iri:
            return [p_iri]
        else:
            logging.warning(
                f"{input} is a curie but could not be resolved to an IRI, "
                f"neither with the provided prefix map nor with bioregistry."
            )
    elif os.path.isfile(input):
        pred_list = Path(input).read_text().splitlines()
        iri_list = []
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


def filter_file(input: str, prefix: tuple, predicate: tuple) -> MappingSetDataFrame:
    """Filter mapping file based on prefix and predicates provided.

    :param input: Input mapping file (tsv)
    :param prefix: Prefixes to be retained.
    :param predicate: Predicates to be retained.
    :return: Filtered MappingSetDataFrame.
    """
    msdf: MappingSetDataFrame = parse_sssom_table(input)
    prefix_map = msdf.prefix_map
    df: pd.DataFrame = msdf.df
    # Filter prefix_map
    filtered_prefix_map = {
        k: v for k, v in prefix_map.items() if k in prefix_map.keys() and k in prefix
    }

    filtered_predicates = {
        k: v
        for k, v in prefix_map.items()
        if len([x for x in predicate if str(x).startswith(k)]) > 0 # use re.find instead.
    }
    filtered_prefix_map.update(filtered_predicates)
    filtered_prefix_map = add_built_in_prefixes_to_prefix_map(filtered_prefix_map)

    # Filter df based on predicates
    predicate_filtered_df: pd.DataFrame = df.loc[
        df[PREDICATE_ID].apply(lambda x: x in predicate)
    ]

    # Filter df based on prefix_map
    prefix_keys = tuple(filtered_prefix_map.keys())
    condition_subj = predicate_filtered_df[SUBJECT_ID].apply(
        lambda x: str(x).startswith(prefix_keys)
    )
    condition_obj = predicate_filtered_df[OBJECT_ID].apply(
        lambda x: str(x).startswith(prefix_keys)
    )
    filtered_df = predicate_filtered_df.loc[condition_subj & condition_obj]

    new_msdf: MappingSetDataFrame = MappingSetDataFrame(
        df=filtered_df, prefix_map=filtered_prefix_map, metadata=msdf.metadata
    )
    return new_msdf
