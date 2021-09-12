import logging
from typing import Optional

from .context import get_default_metadata
from .parsers import get_parsing_function, read_sssom_table, split_dataframe
from .util import raise_for_bad_path, read_metadata
from .writers import get_writer_function, write_table, write_tables


def convert_file(
    input_path: str,
    output_path: str,
    output_format: Optional[str] = None,
) -> None:
    """Convert a file.

    Args:
        input_path: The path to the input SSSOM tsv file
        output_path: The path to the output file.
        output_format: The format to which the the SSSOM TSV should be converted.
    """
    raise_for_bad_path(input_path)
    doc = read_sssom_table(input_path)
    write_func, fileformat = get_writer_function(output_format, output_path)
    write_func(doc, output_path, serialisation=fileformat)


def parse_file(
    input_path: str,
    output_path: Optional[str] = None,
    input_format: Optional[str] = None,
    metadata_path: Optional[str] = None,
    curie_map_mode: Optional[str] = None,
    clean_prefixes: bool = True,
) -> None:
    """

    Args:
        input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
        output_path: The path to the output file.
        input_format: The string denoting the input format.
        metadata_path: The path to a file containing the sssom metadata (including curie_map)
            to be used during parse.
        curie_map_mode: Defines whether the curie map in the metadata should be extended or replaced with
            the SSSOM default curie map. Must be one of metadata_only, sssom_default_only, merged
        clean_prefixes: If True (default), records with unknown prefixes are removed from the SSSOM file.
    """
    raise_for_bad_path(input_path)
    meta, curie_map = get_metadata_and_curie_map(
        metadata_path=metadata_path, curie_map_mode=curie_map_mode
    )
    parse_func = get_parsing_function(input_format, input_path)
    doc = parse_func(input_path, curie_map=curie_map, meta=meta)
    if clean_prefixes:
        # We do this because we got a lot of prefixes from the default SSSOM prefixes!
        doc.clean_prefix_map()
    write_table(doc, output_path)


def validate_file(input_path: str) -> bool:
    """
    Validate the incoming SSSOM TSV according to the SSSOM specification.

    Args:
        input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml

    Returns:
        Boolean. True if valid SSSOM, false otherwise.
    """
    try:
        read_sssom_table(file_path=input_path)
        return True
    except Exception as e:
        logging.exception("The file is invalid", e)
        return False


def split_file(input_path: str, output_directory: str) -> None:
    """
    Split an SSSOM TSV by prefixes and relations.

    Args:
        input_path: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
        output_directory: The directory to which the split file should be exported.
    """
    raise_for_bad_path(input_path)
    msdf = read_sssom_table(input_path)
    splitted = split_dataframe(msdf)
    write_tables(splitted, output_directory)


def get_metadata_and_curie_map(
    metadata_path: Optional[str] = None, curie_map_mode: str = "metadata_only"
):
    """
    Load SSSOM metadata from a file, and then augments it with default prefixes.

    :param metadata_path: The metadata file in YAML format
    :param curie_map_mode: one of metadata_only, sssom_default_only, merged
    :return: a curie map dictionary and a metadata object dictionary
    """
    if metadata_path is None:
        return get_default_metadata()

    meta, curie_map = read_metadata(metadata_path)
    if curie_map_mode != "metadata_only":
        meta_sssom, curie_map_sssom = get_default_metadata()
        if curie_map_mode == "sssom_default_only":
            curie_map = curie_map_sssom
        elif curie_map_mode == "merged":
            for prefix, uri_prefix in curie_map_sssom.items():
                if prefix not in curie_map:
                    curie_map[prefix] = uri_prefix

    return meta, curie_map
