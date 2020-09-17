import click
from .util import parse, collapse, export_ptable
from .io import convert_file
import pandas as pd
from scipy.stats import chi2_contingency

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
    convert file
    """
    convert_file(input=input, output=output, input_format=format, output_format=to_format, context_path=context)

@main.command()
@click.option('-p', '--priors', nargs=4, default=(0.02, 0.02, 0.02, 0.02))
@click.argument('input')
def ptable(input, priors):
    """
    write ptable (kboom/boomer input)
    """
    df = parse(input)
    df = collapse(df)
    export_ptable(df, priors=list(priors))

@main.command()
@click.option('-s', '--summary_file')
@click.option('-o', '--output')
@click.option('-t', '--transpose/--no-transpose', default=False)
@click.argument('input')
def crosstab(input, summary_file, output, transpose):
    """
    write ptable (kboom/boomer input)
    """
    df = parse(input)
    ct = pd.crosstab(df.subject_category, df.object_category)
    if transpose:
        ct = ct.transpose()
    if summary_file is not None:
        ct.to_csv(summary_file, sep="\t")
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
