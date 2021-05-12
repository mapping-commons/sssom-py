import os
from sssom.datamodel_util import MappingSetDataFrame
from sssom.sssom_datamodel import MappingSet

from .parsers import get_parsing_function
from .writers import get_writer_function, write_tsv
from .context import get_jsonld_context
import json
import yaml
from .datamodel_util import MappingSetDataFrame
import logging

cwd = os.path.abspath(os.path.dirname(__file__))

def write_sssom(mset: MappingSetDataFrame, output: str = None) -> None:
    if mset.metadata is not None:
        obj = {k:v for k,v in mset.metadata.items()}
    else:
        obj = {}
    if mset.prefixmap is not None:
        obj['curie_map'] = mset.prefixmap
    lines = yaml.safe_dump(obj).split("\n")
    lines = [f'# {line}' for line in lines if line != '']
    s = mset.df.to_csv(sep="\t", index=False)
    lines = lines + [s]
    if output is None:
        for line in lines:
            print(line)
    else:
        with open(output, 'w') as stream:
            for line in lines:
                stream.write(line)

def convert_file(input: str, output: str = None, input_format: str = None, output_format: str = None, context_path=None,
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
    curie_map={}
    contxt = get_jsonld_context()

    if context_path:
        if os.path.isfile(context_path):
            with open(context_path) as json_file:
                contxt = json.load(json_file)



    for key in contxt["@context"]:
        v = contxt["@context"][key]
        if isinstance(v,str):
            curie_map[key]=v
    if read_func is None:
        read_func = get_parsing_function(input_format, input)
    doc = read_func(input,curie_map=curie_map)

    if write_func is None:
        write_func, fileformat = get_writer_function(output_format, output)
    write_func(doc, output, fileformat=fileformat, context_path=context_path)


def parse_file(input_path: str, output_path: str = None, input_format: str = None, metadata_path=None):
    """
    converts from one format to another
    :param input_path:
    :param output_path:
    :param input_format:
    :param metadata_path:
    :return:
    """
    curie_map={}
    contxt = get_jsonld_context()

    '''if context_path:
        if os.path.isfile(context_path):
            with open(context_path) as json_file:
                contxt = json.load(json_file)'''

     

    for key in contxt["@context"]:
        v = contxt["@context"][key]
        if isinstance(v,str):
            curie_map[key]=v
    
    read_func = get_parsing_function(input_format, input)
    doc = read_func(input,curie_map=curie_map)
    write_tsv(doc, output_path)
    
