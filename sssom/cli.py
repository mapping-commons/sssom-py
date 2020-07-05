import click
from .util import parse, collapse, export_ptable
from .io import convert_file

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
    write ptable
    """
    df = parse(input)
    df = collapse(df)
    export_ptable(df, priors=list(priors))

if __name__ == "__main__":
    main()
