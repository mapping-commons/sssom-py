from sssom.datamodel_util import to_mapping_set_dataframe, SSSOM_READ_FORMATS
import click
import yaml
import re
from pathlib import Path
from sssom import slots
from .util import parse, collapse, dataframe_to_ptable, filter_redundant_rows, remove_unmatched, compare_dataframes
from .cliques import split_into_cliques, summarize_cliques
from .io import convert_file, write_sssom, parse_file, split_file, validate_file
from .parsers import from_tsv
from .writers import write_tsv
from typing import Tuple, List, Dict
import pandas as pd
from scipy.stats import chi2_contingency
import logging
from pandasql import sqldf

# Click 'help' variabes
help_input = 'Input file. For e.g.: SSSOM tsv file'
help_input_list = 'List of input files.'
help_input_format = 'Input file format.'
help_output = 'Output TSV/SSSOM file'
help_output_format = 'Output file format.'
help_output_directory = 'Output directory path.'
help_format = 'Desired output format.'
help_context = 'Context.'
help_metadata = 'Metadata.'
help_curie_map_mode = 'Curie map mode.'
help_inverse_factor = 'Inverse factor.'
help_query = 'SQL query. Use "df" as table name'

@click.group()
@click.option('-v', '--verbose', count=True)
@click.option('-q', '--quiet')
def main(verbose:int, quiet:bool):
    """Main

    Args:

        verbose (int): Verbose.

        quiet (bool): Quiet.
    """    
    if verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    if quiet:
        logging.basicConfig(level=logging.ERROR)


@main.command('convert_file')
@click.option('-i', '--input', help=help_input)
@click.option('-f', '--format', help=help_input_format)
@click.option('-o', '--output', help= help_output)
@click.option('-t', '--to-format', help=help_format)
@click.option('-c', '--context', help=help_context)
def convert(input: str, output: str, format: str, to_format: str, context: str):
    """Convert file (currently only supports conversion to RDF)

    Example:

    A B C

    D E F

    G H I
    """   

    convert_file(input=input, output=output, input_format=format, output_format=to_format, context_path=context)

## Input and metadata would be files (file paths). Check if exists.
@main.command('parse_file')
@click.option('-i', '--input', required=True, type=click.Path(), help=help_input)
@click.option('-I', '--input-format', required=False,
              type=click.Choice(SSSOM_READ_FORMATS, case_sensitive=False), help=help_input_format)
@click.option('-m', '--metadata', required=False, type=click.Path(), help= help_metadata)
@click.option('-c', '--curie-map-mode', default='metadata_only', show_default=True, required=True,
              type=click.Choice(['metadata_only', 'sssom_default_only', 'merged'], case_sensitive=False), help=help_curie_map_mode)
@click.option('-o', '--output', type=click.Path(), help= help_output)
def parse(input: str, input_format: str, metadata:str, curie_map_mode: str, output: str):
    """parse file (currently only supports conversion to RDF)

    Example:

    A B C

    D E F
    
    G H I
    """    

    parse_file(input_path=input, output_path=output, input_format=input_format, metadata_path=metadata, curie_map_mode=curie_map_mode)

@main.command('validate_file')
@click.option('-i', '--input', required=True, type=click.Path(), help=help_input)
@click.option('-I', '--input-format', required=False,
              type=click.Choice(SSSOM_READ_FORMATS, case_sensitive=False), help=help_input_format)
@click.option('-m', '--metadata', required=False, type=click.Path(), help= help_metadata)
@click.option('-c', '--curie-map-mode', default='metadata_only', show_default=True, required=True,
              type=click.Choice(['metadata_only', 'sssom_default_only', 'merged'], case_sensitive=False), help=help_curie_map_mode)
@click.option('-o', '--output', type=click.Path(), help= help_output)
def validate(input: str, input_format: str, metadata:str, curie_map_mode: str, output: str):
    """parse file (currently only supports conversion to RDF)

    Example:

    A B C

    D E F
    
    G H I
    """    

    validate_file(input_path=input, output_path=output, input_format=input_format, metadata_path=metadata, curie_map_mode=curie_map_mode)

@main.command('split_file')
@click.option('-i', '--input', required=True, type=click.Path(), help=help_input)
@click.option('-d', '--output-directory', type=click.Path(), help=help_output_directory)
def split(input: str, output_directory: str):
    """Parse file (currently only supports conversion to RDF)

    Example:

    A B C

    D E F
    
    G H I

    """    

    split_file(input_path=input, output_directory=output_directory)


@main.command()
@click.option('-W', '--inverse-factor', help=help_inverse_factor)
@click.argument('input')
def ptable(input:str, inverse_factor):
    """Write ptable (kboom/boomer input)
    should maybe move to boomer (but for now it can live here, so cjm can tweak

    Example:

    A B C

    D E F
    
    G H I

    """    
    df = parse(input)
    df = collapse(df)
    # , priors=list(priors)
    rows = dataframe_to_ptable(df)
    for row in rows:
        logging.info("\t".join(row))


@main.command()
@click.option('-i', '--input', help=help_input)
@click.option('-o', '--output', help=help_output)
def dedupe(input: str, output: str):
    """Remove lower confidence duplicate lines.

    Example:

    A B C

    D E F
    
    G H I

    """    
    df = parse(input)
    df = filter_redundant_rows(df)
    df.to_csv(output, sep="\t", index=False)

@main.command()
@click.option('-q', '--query',
            help= help_query)
@click.option('-o', '--output',
              help= help_output)
@click.argument('inputs', nargs=-1)
def dosql(query:str, inputs: List[str], output: str):
    """
    Run a SQL query over one or more sssom files.

    Each of the N inputs is assigned a table name df1, df2, ..., dfN

    Alternatively, the filenames can be used as table names - these are first stemmed
    E.g. ~/dir/my.sssom.tsv becomes a table called 'my'

    Example:
        sssom dosql -q "SELECT * FROM df1 WHERE confidence>0.5 ORDER BY confidence" my.sssom.tsv

    Example:
        `sssom dosql -q "SELECT file1.*,file2.object_id AS ext_object_id, file2.object_label AS ext_object_label \
        FROM file1 INNER JOIN file2 WHERE file1.object_id = file2.subject_id" FROM file1.sssom.tsv file2.sssom.tsv`

    """   
    
    n = 1
    while len(inputs) >= n:
        fn = inputs[n-1]
        df = parse(fn)
        globals()[f'df{n}'] = df
        tn = re.sub('\..*','',Path(fn).stem).lower()
        globals()[tn] = df
        n += 1
    df = sqldf(query)
    if output is None:
        print(df.to_csv(sep="\t", index=False))
    else:
        df.to_csv(output, sep="\t", index=False)

from sssom.sparql_util import EndpointConfig, query_mappings
@main.command('write_sssom')
@click.option('-c', '--config', type=click.File('rb'))
@click.option('-e', '--url')
@click.option('-g', '--graph')
@click.option('--object-labels/--no-object-labels', default=None, help='if set, includes object labels')
@click.option('-l', '--limit', type=int)
@click.option('-P', '--prefix', type=click.Tuple([str, str]), multiple=True)
@click.option('-o', '--output')
def sparql(url: str = None, config = None, graph: str = None, limit: int = None,
           object_labels: bool = None,
           prefix:List[Dict[str,str]] = None,
           output: str = None):
    """Run a SPARQL query.

    Example:

    A B C

    D E F
    
    G H I

    """           

    endpoint = EndpointConfig()
    if config is not None:
        for k,v in yaml.safe_load(config).items():
            setattr(endpoint, k, v)
    if url is not None:
        endpoint.url = url
    if graph is not None:
        endpoint.graph = graph
    if limit is not None:
        endpoint.limit = limit
    if object_labels is not None:
        endpoint.include_object_labels = object_labels
    if prefix is not None:
        if endpoint.curie_map is None:
            endpoint.curie_map = {}
        for k,v in prefix:
            endpoint.curie_map[k] = v
    msdf = query_mappings(endpoint)
    write_sssom(msdf, output)

@main.command()
@click.option('-o', '--output')
@click.argument('inputs', nargs=2)
def diff(inputs: Tuple[str,str], output:str):
    """Compare two SSSOM files.
    The output is a new SSSOM file with the union of all mappings, and
    injected comments indicating uniqueness to set1 or set2

    Example:

    A B C

    D E F
    
    G H I

    """    

    (input1, input2) = inputs
    df1 = parse(input1)
    df2 = parse(input2)
    d = compare_dataframes(df1, df2)
    logging.info(f'COMMON: {len(d.common_tuples)} UNIQUE_1: {len(d.unique_tuples1)} UNIQUE_2: {len(d.unique_tuples2)}')
    d.combined_dataframe.to_csv(output, sep="\t", index=False)

@main.command()
@click.option('-d', '--outdir')
@click.argument('inputs', nargs=-1)
def partition(inputs: List[str], outdir: str):
    """Partitions an SSSOM file into multiple files, where each
    file is a strongly connected component

    Example:

    A B C

    D E F
    
    G H I

    """    

    docs = [from_tsv(input) for input in inputs]
    doc = docs.pop()
    for d2 in docs:
        doc.mapping_set.mappings += d2.mapping_set.mappings
    cliquedocs = split_into_cliques(doc)
    n = 0
    for cdoc in cliquedocs:
        n += 1
        ofn = f'{outdir}/clique_{n}.sssom.tsv'
        logging.info(f'Writing to {ofn}. Size={len(cdoc.mapping_set.mappings)}')
        logging.info(f'Example: {cdoc.mapping_set.mappings[0].subject_id}')
        msdf = to_mapping_set_dataframe(cdoc)
        write_tsv(msdf, ofn)


@main.command()
@click.option('-i', '--input')
@click.option('-o', '--output')
@click.option('-m', '--metadata')
@click.option('-s', '--statsfile')
def cliquesummary(input: str, output: str, metadata: str, statsfile: str):
    """Partitions an SSSOM file into multiple files, where each
    file is a strongly connected component.

    The data dictionary for the output is in cliquesummary.yaml

    Example:

    A B C

    D E F
    
    G H I

    """    
    import yaml
    if metadata is None:
        doc = from_tsv(input)
    else:
        meta_obj = yaml.safe_load(open(metadata))
        doc = from_tsv(input, meta=meta_obj)
    df = summarize_cliques(doc)
    df.to_csv(output, sep="\t")
    if statsfile is None:
        logging.info(df.describe)
    else:
        df.describe().transpose().to_csv(statsfile, sep="\t")


@main.command()
@click.option('-o', '--output')
@click.option('-t', '--transpose/--no-transpose', default=False)
@click.option('-F', '--fields', nargs=2, default=(slots.subject_category.name, slots.object_category.name))
@click.argument('input')
def crosstab(input:str, output:str, transpose:bool, fields):
    """Write sssom summary cross-tabulated by categories.

    Example:

    A B C

    D E F
    
    G H I

    """    

    df = remove_unmatched(parse(input))
    # df = parse(input)
    logging.info(f'#CROSSTAB ON {fields}')
    (f1, f2) = fields
    ct = pd.crosstab(df[f1], df[f2])
    if transpose:
        ct = ct.transpose()
    if output is not None:
        ct.to_csv(output, sep="\t")
    else:
        logging.info(ct)


@main.command()
@click.option('-o', '--output')
@click.option('-t', '--transpose/--no-transpose', default=False)
@click.option('-v', '--verbose/--no-verbose', default=False)
@click.option('-F', '--fields', nargs=2, default=(slots.subject_category.name, slots.object_category.name))
@click.argument('input')
def correlations(input:str, output:str, transpose:bool, verbose:bool, fields):
    """

    Example:

    A B C

    D E F
    
    G H I
    
    """    

    df = remove_unmatched(parse(input))
    if len(df) == 0:
        msg = f"No matched entities in this dataset!"
        logging.error(msg)
        exit(1)

    logging.info(f'#CROSSTAB ON {fields}')
    (f1, f2) = fields
    if verbose:
        logging.info(f'F1 {f1} UNIQUE: {df[f1].unique()}')
        logging.info(f'F2 {f2} UNIQUE: {df[f2].unique()}')

    ct = pd.crosstab(df[f1], df[f2])
    if transpose:
        ct = ct.transpose()

    chi2 = chi2_contingency(ct)
    if verbose:
        logging.info(chi2)
    _, _, _, ndarray = chi2
    corr = pd.DataFrame(ndarray, index=ct.index, columns=ct.columns)
    if output:
        corr.to_csv(output, sep="\t")
    else:
        logging.info(corr)

    if verbose:
        tups = []
        for i, row in corr.iterrows():
            for j, v in row.iteritems():
                logging.info(f'{i} x {j} = {v}')
                tups.append((v, i, j))
        tups = sorted(tups, key=lambda t: t[0])
        for t in tups:
            logging.info(f'{t[0]}\t{t[1]}\t{t[2]}')


if __name__ == "__main__":
    main()
