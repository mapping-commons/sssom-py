import click
from sssom import slots
from .util import parse, collapse, export_ptable, filter_redundant_rows, remove_unmatched
from .io import convert_file
import pandas as pd
from scipy.stats import chi2_contingency
import logging

@click.group()
def main():
    pass

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
    export_ptable(df, priors=list(priors))

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
@click.option('-o', '--output')
@click.option('-t', '--transpose/--no-transpose', default=False)
@click.option('-F', '--fields', nargs=2, default=(slots.subject_category.name, slots.object_category.name))
@click.option('-o', '--output')
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
@click.option('-F', '--fields', nargs=2, default=(slots.subject_category.name, slots.object_category.name))
@click.option('-o', '--output')
@click.argument('input')
def correlations(input, output, transpose, fields):
    """
    write sssom summary cross-tabulated by categories
    """
    df = remove_unmatched(parse(input))
    logging.info(f'#CROSSTAB ON {fields}')
    (f1, f2) = fields
    print(f'F1 {f1} UNIQUE: {df[f1].unique()}')
    print(f'F2 {f2} UNIQUE: {df[f2].unique()}')
    ct = pd.crosstab(df[f1], df[f2])
    if transpose:
        ct = ct.transpose()
    chi2 = chi2_contingency(ct)
    print(chi2)
    _,_,_,ndarray = chi2
    corr = pd.DataFrame(ndarray, index=ct.index, columns=ct.columns)
    if output:
        corr.to_csv(output, sep="\t")
    else:
        print(corr)

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
