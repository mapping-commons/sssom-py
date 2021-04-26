import click
from sssom import slots
from .util import parse, collapse, dataframe_to_ptable, filter_redundant_rows, remove_unmatched, compare_dataframes
from .cliques import split_into_cliques, summarize_cliques
from .io import convert_file
from .parsers import from_tsv
from .writers import write_tsv
import statistics
from typing import Tuple
import pandas as pd
from scipy.stats import chi2_contingency
import logging
from pandasql import sqldf


@click.group()
@click.option('-v', '--verbose', count=True)
def main(verbose):
    if verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)


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
        print("\t".join(row))


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
@click.option('-q', '--query', help='SQL query. Use "df" as table name')
@click.option('-o', '--output')
@click.argument('input')
def dosql(query:str, input: str, output: str):
    """
    Run a SQL query.

    Example:
        sssom dosql -q "SELECT * FROM df WHERE confidence>0.5 ORDER BY confience" my.sssom.tsv
    """
    df = parse(input)
    df = sqldf(query)
    df.to_csv(output, sep="\t", index=False)

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
    print(f'COMMON: {len(d.common_tuples)} UNIQUE_1: {len(d.unique_tuples1)} UNIQUE_2: {len(d.unique_tuples2)}')
    d.combined_dataframe.to_csv(output, sep="\t", index=False)

@main.command()
@click.option('-i', '--input')
@click.option('-d', '--outdir')
def partition(input: str, outdir: str):
    """
    partitions an SSSOM file into multiple files, where each
    file is a strongly connected component
    """
    doc = from_tsv(input)
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
        print(df.describe)
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
        print(ct)


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
        print(f'F1 {f1} UNIQUE: {df[f1].unique()}')
        print(f'F2 {f2} UNIQUE: {df[f2].unique()}')

    ct = pd.crosstab(df[f1], df[f2])
    if transpose:
        ct = ct.transpose()

    chi2 = chi2_contingency(ct)
    if verbose:
        print(chi2)
    _, _, _, ndarray = chi2
    corr = pd.DataFrame(ndarray, index=ct.index, columns=ct.columns)
    if output:
        corr.to_csv(output, sep="\t")
    else:
        print(corr)

    if verbose:
        tups = []
        for i, row in corr.iterrows():
            for j, v in row.iteritems():
                print(f'{i} x {j} = {v}')
                tups.append((v, i, j))
        tups = sorted(tups, key=lambda t: t[0])
        for t in tups:
            print(f'{t[0]}\t{t[1]}\t{t[2]}')


if __name__ == "__main__":
    main()
