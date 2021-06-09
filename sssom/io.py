import os
from sssom.datamodel_util import MappingSetDataFrame, read_metadata
from sssom.sssom_datamodel import MappingSet

from .parsers import get_parsing_function, from_tsv, split_dataframe
from .writers import get_writer_function, write_tsv, write_tsvs
from .context import get_default_metadata
import validators
import yaml
import logging

cwd = os.path.abspath(os.path.dirname(__file__))

def write_sssom(msdf: MappingSetDataFrame, output: str = None) -> None:
    if msdf.metadata is not None:
        obj = {k:v for k,v in msdf.metadata.items()}
    else:
        obj = {}
    if msdf.prefixmap is not None:
        obj['curie_map'] = msdf.prefixmap
    lines = yaml.safe_dump(obj).split("\n")
    lines = [f'# {line}' for line in lines if line != '']
    s = msdf.df.to_csv(sep="\t", index=False)
    lines = lines + [s]
    if output is None:
        for line in lines:
            print(line)
    else:
        with open(output, 'w') as stream:
            for line in lines:
                stream.write(line)

def convert_file(input_path: str, output: str = None, input_format: str = None, output_format: str = None, context_path=None,
                 read_func=None, write_func=None):
    """
    converts from one format to another
    :param input:
    :param output:
    :param input_format:
    :param output_format:
    :param context_path:
    :param read_func
    :param write_func
    :return:
    """
    if validators.url(input_path) or os.path.exists(input_path):
        curie_map, meta = get_metadata_and_curie_map(metadata_path=context_path, curie_map_mode="metadata_only")

        if read_func is None:
            read_func = get_parsing_function(input_format, input_path)
        doc = read_func(input_path,curie_map=curie_map)

        if write_func is None:
            write_func, fileformat = get_writer_function(output_format, output)
        write_func(doc, output, fileformat=fileformat, context_path=context_path)
        
    else:
        raise Exception(f"{input_path} is not a valid file path or url.")
    


def parse_file(input_path: str, output_path: str = None, input_format: str = None, metadata_path: str = None, curie_map_mode: str = None):
    """
    converts from one format to another
    :param input_path:
    :param output_path:
    :param input_format:
    :param metadata_path:
    :param curie_map_mode:
    :return:
    """

    if validators.url(input_path) or os.path.exists(input_path):
        curie_map, meta = get_metadata_and_curie_map(metadata_path=metadata_path, curie_map_mode=curie_map_mode)
        parse_func = get_parsing_function(input_format, input_path)
        doc = parse_func(input_path, curie_map=curie_map, meta=meta)
        write_tsv(doc, output_path)
    else:
        raise Exception(f"{input_path} is not a valid file path or url.")

def validate_file(input_path: str):
    """
    converts from one format to another
    :param input_path: SSSOM file
    :param output_path: Validation Report
    :return:
    """
    try:
        from_tsv(file_path=input_path)
        return True
    except:
        logging.exception("The file is invalid")
        return False


def split_file(input_path: str, output_directory: str):
    """
    Splits an SSSOM TSV by prefixes and relations.
    :param input_path: the SSSOM file
    :param output_directory: the directory in which to export the split files
    :return:
    """
    if validators.url(input_path) or os.path.exists(input_path):
        msdf = from_tsv(input_path)
        splitted = split_dataframe(msdf)
        write_tsvs(splitted, output_directory)
    else:
        raise Exception(f"{input_path} is not a valid file path or url.")


def get_metadata_and_curie_map(metadata_path, curie_map_mode: str = "metadata_only"):
    """
    Loads sssom metadata from a file, and then augments it with default prefixes.
    :param metadata_path: The metadata file in YAML format
    :param curie_map_mode:
    :return: a curie map dictionary and a metadata object dictionary
    """
    if metadata_path:
        meta, curie_map = read_metadata(metadata_path)
        if curie_map_mode != "metadata_only":
            meta_sssom, curie_map_sssom = get_default_metadata()
            if curie_map_mode == "sssom_default_only":
                curie_map = curie_map_sssom
            elif curie_map_mode == "merged":
                for prefix, uri_prefix in curie_map_sssom.items():
                    if prefix not in curie_map:
                        curie_map[prefix] = uri_prefix
    else:
        meta, curie_map = get_default_metadata()
    return curie_map, meta
