"""I/O utilities for SSSOM."""

import logging
from pathlib import Path
from typing import Optional, TextIO, Union

from .context import get_default_metadata
from .parsers import get_parsing_function, read_sssom_table, split_dataframe
from .typehints import Metadata
from .util import raise_for_bad_path, read_metadata
from .writers import get_writer_function, write_table, write_tables


def convert_file(
    input_path: str,
    output: TextIO,
    output_format: Optional[str] = None,
) -> None:
    """Convert a file.

    :param input_path: The path to the input SSSOM tsv file
    :param output: The path to the output file. If none is given, will default to using stdout.
    :param output_format: The format to which the the SSSOM TSV should be converted.
    """
    raise_for_bad_path(input_path)
    doc = read_sssom_table(input_path)
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
    """
    raise_for_bad_path(input_path)
    metadata = get_metadata_and_prefix_map(
        metadata_path=metadata_path, prefix_map_mode=prefix_map_mode
    )
    parse_func = get_parsing_function(input_format, input_path)
    doc = parse_func(
        input_path, prefix_map=metadata.prefix_map, meta=metadata.prefix_map
    )
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
        read_sssom_table(file_path=input_path)
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
    msdf = read_sssom_table(input_path)
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
        meta_sssom, prefix_map_sssom = get_default_metadata()
        if prefix_map_mode == "sssom_default_only":
            prefix_map = prefix_map_sssom
        elif prefix_map_mode == "merged":
            for prefix, uri_prefix in prefix_map_sssom.items():
                if prefix not in prefix_map:
                    prefix_map[prefix] = uri_prefix
    return Metadata(prefix_map=prefix_map, metadata=metadata)
