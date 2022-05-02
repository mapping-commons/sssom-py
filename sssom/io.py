"""I/O utilities for SSSOM."""

import itertools
import logging
import os
from pathlib import Path
from typing import Optional, TextIO, Union

from bioregistry import get_iri

from .context import (
    get_default_metadata,
    set_default_license,
    set_default_mapping_set_id,
)
from .parsers import get_parsing_function, parse_sssom_table, split_dataframe
from .typehints import Metadata
from .util import is_curie, raise_for_bad_path, read_metadata
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
    :param mapping_predicate_filter: Optional list of mapping predicates or filepath containing the same.
    """
    raise_for_bad_path(input_path)
    metadata = get_metadata_and_prefix_map(
        metadata_path=metadata_path, prefix_map_mode=prefix_map_mode
    )
    metadata = set_default_mapping_set_id(metadata)
    metadata = set_default_license(metadata)
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
    write_table(doc, output)


def validate_file(input_path: str) -> bool:
    """Validate the incoming SSSOM TSV according to the SSSOM specification.

    :param input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
    :returns: True if valid SSSOM, false otherwise.
    """
    try:
        parse_sssom_table(file_path=input_path)
        return True
    except Exception as e:
        logging.exception("The file is invalid", e)
        return False


def split_file(input_path: str, output_directory: Union[str, Path]) -> None:
    """Split an SSSOM TSV by prefixes and relations.

    :param  input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
    :param output_directory: The directory to which the split file should be exported.
    """
    raise_for_bad_path(input_path)
    msdf = parse_sssom_table(input_path)
    splitted = split_dataframe(msdf)
    write_tables(splitted, output_directory)


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
    if prefix_map_mode is None:
        prefix_map_mode = "metadata_only"
    prefix_map, metadata = read_metadata(metadata_path)
    # TODO reduce complexity by flipping conditionals
    #  and returning eagerly (it's fine if there are multiple returns)
    if prefix_map_mode != "metadata_only":
        default_metadata: Metadata = get_default_metadata()
        if prefix_map_mode == "sssom_default_only":
            prefix_map = default_metadata.prefix_map
        elif prefix_map_mode == "merged":
            for prefix, uri_prefix in default_metadata.prefix_map.items():
                if prefix not in prefix_map:
                    prefix_map[prefix] = uri_prefix
    return Metadata(prefix_map=prefix_map, metadata=metadata)


def get_list_of_predicate_iri(predicate_filter: tuple, prefix_map: dict) -> list:
    """Return a list of IRIs for predicate CURIEs passed.

    :param predicate_filter: CURIE OR list of CURIEs OR file path containing the same.
    :param prefix_map: Prefix map of mapping set (possibly) containing custom prefix:IRI combination.
    :return: A list of IRIs.
    """
    pred_filter_list = list(predicate_filter)
    preds = [p for p in pred_filter_list if is_curie(p)]
    preds_iri = [get_iri(p) for p in preds]
    non_bioreg_preds = [p for p in preds if get_iri(p) is None]
    non_bioreg_pred_iri = [
        get_iri(p, prefix_map=prefix_map, use_bioregistry_io=False)
        for p in non_bioreg_preds
    ]
    preds_iri.extend(non_bioreg_pred_iri)

    if len(preds) != len(pred_filter_list) and len(preds) > 0:
        # The user passed file paths too.
        pred_fps = [p for p in pred_filter_list if p not in preds]
        if all(os.path.isfile(p) for p in pred_fps):
            pred_list = list(
                itertools.chain(*[Path(f).read_text().splitlines() for f in pred_fps])
            )
            preds_iri.extend([get_iri(p) for p in pred_list])

        else:
            logging.warn(f"{pred_fps} does not contain a valid file path.")
    return list(set(preds_iri))
