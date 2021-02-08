import click
from sssom import slots
from .util import parse, collapse, export_ptable, filter_redundant_rows, remove_unmatched
from .cliques import split_into_cliques, cliquesummary
from .io import convert_file
from .parsers import from_tsv
from .writers import write_tsv
import statistics
import pandas as pd
from scipy.stats import chi2_contingency
import logging

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
    #, priors=list(priors)
    export_ptable(df)

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
def cliquesummary(input: str, output: str):
    """
    partitions an SSSOM file into multiple files, where each
    file is a strongly connected component
    """
    doc = from_tsv(input)
    df = cliquesummary(doc)
    df.to_csv(output, sep="\t")
    print(df.describe)



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
    #df = parse(input)
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
    _,_,_,ndarray = chi2
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
                tups.append( (v, i, j) )
        tups = sorted(tups, key=lambda t: t[0])
        for t in tups:
            print(f'{t[0]}\t{t[1]}\t{t[2]}')


if __name__ == "__main__":
    main()
