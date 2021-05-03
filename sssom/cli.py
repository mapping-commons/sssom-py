import click
import yaml
import re
from pathlib import Path
from sssom import slots
from .util import parse, collapse, dataframe_to_ptable, filter_redundant_rows, remove_unmatched, compare_dataframes
from .cliques import split_into_cliques, summarize_cliques
from .io import convert_file, write_sssom
from .parsers import from_tsv
from .writers import write_tsv
from .datamodel_util import MappingSetDataFrame
import statistics
from typing import Tuple, List, Dict
import pandas as pd
from scipy.stats import chi2_contingency
import logging
from pandasql import sqldf


@click.group()
@click.option('-v', '--verbose', count=True)
@click.option('-q', '--quiet')
def main(verbose, quiet):
    if verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    if quiet:
        logging.basicConfig(level=logging.ERROR)


@main.command()
@click.option('-i', '--input')
@click.option('-f', '--format')
@click.option('-o', '--output')
@click.option('-t', '--to-format')
@click.option('-c', '--context')
def convert(input: str, output: str, format: str, to_format: str, context: str):
    """
    convert file (currently only supports conversion to RDF)
    """
    convert_file(input=input, output=output, input_format=format, output_format=to_format, context_path=context)


@main.command()
@click.option('-W', '--inverse-factor')
@click.argument('input')
def ptable(input, inverse_factor):
    """
    write ptable (kboom/boomer input)
    should maybe move to boomer (but for now it can live here, so cjm can tweak
    """
    df = parse(input)
    df = collapse(df)
    # , priors=list(priors)
    rows = dataframe_to_ptable(df)
    for row in rows:
        logging.info("\t".join(row))


@main.command()
@click.option('-i', '--input')
@click.option('-o', '--output')
def dedupe(input: str, output: str):
    """
    remove lower confidence duplicate lines
    """
    df = parse(input)
    df = filter_redundant_rows(df)
    df.to_csv(output, sep="\t", index=False)

@main.command()
@click.option('-q', '--query',
              help='SQL query. Use "df" as table name')
@click.option('-o', '--output',
              help='output TSV/SSSOM file')
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
@main.command()
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
    """
    Run a SPARQL query.
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
def diff(inputs: Tuple[str,str], output):
    """
    compare two SSSOM files.
    The output is a new SSSOM file with the union of all mappings, and
    injected comments indicating uniqueness to set1 or set2
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
    """
    partitions an SSSOM file into multiple files, where each
    file is a strongly connected component
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
        write_tsv(cdoc, ofn)


@main.command()
@click.option('-i', '--input')
@click.option('-o', '--output')
@click.option('-m', '--metadata')
@click.option('-s', '--statsfile')
def cliquesummary(input: str, output: str, metadata: str, statsfile: str):
    """
    partitions an SSSOM file into multiple files, where each
    file is a strongly connected component.

    The data dictionary for the output is in cliquesummary.yaml
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
def crosstab(input, output, transpose, fields):
    """
    write sssom summary cross-tabulated by categories
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
def correlations(input, output, transpose, verbose, fields):
    """
    write sssom summary cross-tabulated by categories
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
