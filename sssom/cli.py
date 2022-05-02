"""Command line interface for SSSOM.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m sssom`` python will execute``__main__.py`` as a script. That means there won't be any
  ``sssom.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``sssom.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/8.0.x/setuptools/
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import click
import pandas as pd
import yaml
from pandasql import sqldf
from rdflib import Graph
from scipy.stats import chi2_contingency

from .cliques import split_into_cliques, summarize_cliques
from .io import convert_file, parse_file, split_file, validate_file
from .parsers import parse_sssom_table
from .rdf_util import rewire_graph
from .sparql_util import EndpointConfig, query_mappings
from .util import (
    SSSOM_EXPORT_FORMATS,
    SSSOM_READ_FORMATS,
    MappingSetDataFrame,
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    merge_msdf,
    reconcile_prefix_and_data,
    remove_unmatched,
    sort_df_rows_columns,
    to_mapping_set_dataframe,
)
from .writers import write_table

# Click input options common across commands
input_argument = click.argument("input", required=True, type=click.Path())

input_format_option = click.option(
    "-I",
    "--input-format",
    help=f'The string denoting the input format, e.g. {",".join(SSSOM_READ_FORMATS)}',
)
output_option = click.option(
    "-o",
    "--output",
    help="Output file, e.g. a SSSOM tsv file.",
    type=click.File(mode="w"),
    default=sys.stdout,
)
output_format_option = click.option(
    "-O",
    "--output-format",
    help=f'Desired output format, e.g. {",".join(SSSOM_EXPORT_FORMATS)}',
)
output_directory_option = click.option(
    "-d",
    "--output-directory",
    type=click.Path(),
    help="Output directory path.",
    default=os.getcwd(),
)
metadata_option = click.option(
    "-m",
    "--metadata",
    required=False,
    type=click.Path(),
    help="The path to a file containing the sssom metadata (including prefix_map) to be used.",
)
transpose_option = click.option("-t", "--transpose", default=False)
fields_option = click.option(
    "-f",
    "--fields",
    nargs=2,
    default=("subject_category", "object_category"),
    help="Fields.",
)

predicate_filter_option = click.option(
    "-F",
    "--mapping-predicate-filter",
    multiple=True,
    help="A list of predicates or a file path containing the list of predicates to be considered.",
)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
def main(verbose: int, quiet: bool):
    """Run the SSSOM CLI."""
    if verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    if quiet:
        logging.basicConfig(level=logging.ERROR)


@main.command()
@input_argument
@output_option
@output_format_option
def convert(input: str, output: TextIO, output_format: str):
    """Convert a file.

    .. warning:: currently only supports conversion to RDF)

    Example:
        sssom convert my.sssom.tsv --output-format rdfxml --output my.sssom.owl
    """  # noqa: DAR101
    convert_file(input_path=input, output=output, output_format=output_format)


# Input and metadata would be files (file paths). Check if exists.
@main.command()
@input_argument
@input_format_option
@metadata_option
@click.option(
    "-C",
    "--prefix-map-mode",
    default="metadata_only",
    show_default=True,
    required=True,
    type=click.Choice(
        ["metadata_only", "sssom_default_only", "merged"], case_sensitive=False
    ),
    help="Defines whether the prefix map in the metadata should be extended or replaced with "
    "the SSSOM default prefix map. Must be one of metadata_only, sssom_default_only, merged",
)
@click.option(
    "-p",
    "--clean-prefixes",
    default=True,
    is_flag=True,
    required=True,
    help="If True (default), records with unknown prefixes are removed from the SSSOM file.",
)
@predicate_filter_option
@output_option
def parse(
    input: str,
    input_format: str,
    metadata: str,
    prefix_map_mode: str,
    clean_prefixes: bool,
    output: TextIO,
    mapping_predicate_filter: Optional[tuple],
):
    """Parse a file in one of the supported formats (such as obographs) into an SSSOM TSV file."""
    parse_file(
        input_path=input,
        output=output,
        input_format=input_format,
        metadata_path=metadata,
        prefix_map_mode=prefix_map_mode,
        clean_prefixes=clean_prefixes,
        mapping_predicate_filter=mapping_predicate_filter,
    )


@main.command()
@input_argument
def validate(input: str):
    """Produce an error report for an SSSOM file."""
    validate_file(input_path=input)


@main.command()
@input_argument
@output_directory_option
def split(input: str, output_directory: str):
    """Split input file into multiple output broken down by prefixes."""
    split_file(input_path=input, output_directory=output_directory)


@main.command()
@input_argument
@output_option
@click.option("-W", "--inverse-factor", help="Inverse factor.")
def ptable(input, output: TextIO, inverse_factor):
    """Convert an SSSOM file to a ptable for kboom/boomer."""
    # TODO should maybe move to boomer (but for now it can live here, so cjm can tweak
    logging.warning(
        f"inverse_factor ({inverse_factor}) ignored by this method, not implemented yet."
    )
    msdf = parse_sssom_table(input)
    # df = parse(input)
    df = collapse(msdf.df)
    # , priors=list(priors)
    rows = dataframe_to_ptable(df)
    for row in rows:
        print(*row, sep="\t", file=output)


@main.command()
@input_argument
@output_option
def dedupe(input: str, output: TextIO):
    """Remove lower confidence duplicate lines from an SSSOM file."""
    # df = parse(input)
    msdf = parse_sssom_table(input)
    df = filter_redundant_rows(msdf.df)
    msdf_out = MappingSetDataFrame(
        df=df, prefix_map=msdf.prefix_map, metadata=msdf.metadata
    )
    # df.to_csv(output, sep="\t", index=False)
    write_table(msdf_out, output)


@main.command()
@click.option("-Q", "--query", help='SQL query. Use "df" as table name.')
@click.argument("inputs", nargs=-1)
@output_option
def dosql(query: str, inputs: List[str], output: TextIO):
    """Run a SQL query over one or more SSSOM files.

    Each of the N inputs is assigned a table name df1, df2, ..., dfN

    Alternatively, the filenames can be used as table names - these are first stemmed
    E.g. ~/dir/my.sssom.tsv becomes a table called 'my'

    Example:
        sssom dosql -q "SELECT * FROM df1 WHERE confidence>0.5 ORDER BY confidence" my.sssom.tsv

    Example:
        `sssom dosql -q "SELECT file1.*,file2.object_id AS ext_object_id, file2.object_label AS ext_object_label \
        FROM file1 INNER JOIN file2 WHERE file1.object_id = file2.subject_id" FROM file1.sssom.tsv file2.sssom.tsv`
    """  # noqa: DAR101
    # should start with from_tsv and MOST should return write_sssom
    n = 1
    new_msdf = MappingSetDataFrame()
    while len(inputs) >= n:
        fn = inputs[n - 1]
        msdf = parse_sssom_table(fn)
        df = msdf.df
        # df = parse(fn)
        globals()[f"df{n}"] = df
        tn = re.sub("[.].*", "", Path(fn).stem).lower()
        globals()[tn] = df
        n += 1

    new_df = sqldf(query)
    new_msdf.df = new_df
    new_msdf.prefix_map = msdf.prefix_map
    new_msdf.metadata = msdf.metadata
    write_table(new_msdf, output)


@main.command()
@click.option("-c", "--config", type=click.File("rb"))
@click.option("-e", "--url")
@click.option("-g", "--graph")
@click.option(
    "--object-labels/--no-object-labels",
    default=None,
    help="if set, includes object labels",
)
@click.option("-l", "--limit", type=int)
@click.option("-P", "--prefix", type=click.Tuple([str, str]), multiple=True)
@output_option
def sparql(
    url: str,
    config,
    graph: str,
    limit: int,
    object_labels: bool,
    prefix: List[Dict[str, str]],
    output: TextIO,
):
    """Run a SPARQL query."""
    # FIXME this usage needs _serious_ refactoring
    endpoint = EndpointConfig()  # type: ignore
    if config is not None:
        for k, v in yaml.safe_load(config).items():
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
        if endpoint.prefix_map is None:
            endpoint.prefix_map = {}
        for k, v in prefix:
            endpoint.prefix_map[k] = v
    msdf = query_mappings(endpoint)
    write_table(msdf, output)


@main.command()
@output_option
@click.argument("inputs", nargs=2)
def diff(inputs: Tuple[str, str], output: TextIO):
    """Compare two SSSOM files.

    The output is a new SSSOM file with the union of all mappings, and
    injected comments indicating uniqueness to set1 or set2.
    """  # noqa: DAR101,DAR401
    input1, input2 = inputs
    msdf1 = parse_sssom_table(input1)
    msdf2 = parse_sssom_table(input2)
    d = compare_dataframes(msdf1.df, msdf2.df)
    if d.combined_dataframe is None:
        raise RuntimeError
    if (
        d.common_tuples is not None
        and d.unique_tuples1 is not None
        and d.unique_tuples2 is not None
    ):
        logging.info(
            f"COMMON: {len(d.common_tuples)} UNIQUE_1: {len(d.unique_tuples1)} UNIQUE_2: {len(d.unique_tuples2)}"
        )
    d.combined_dataframe.to_csv(output, sep="\t", index=False)


@main.command()
@output_directory_option
@click.argument("inputs", nargs=-1)
def partition(inputs: List[str], output_directory: str):
    """Partition an SSSOM into one file for each strongly connected component."""
    docs = [parse_sssom_table(input) for input in inputs]
    doc = docs.pop()
    """for d2 in docs:
        doc.mapping_set.mappings += d2.mapping_set.mappings"""
    cliquedocs = split_into_cliques(doc)
    for n, cdoc in enumerate(cliquedocs, start=1):
        ofn = f"{output_directory}/clique_{n}.sssom.tsv"
        # logging.info(f'Writing to {ofn}. Size={len(cdoc.mapping_set.mappings)}')
        # logging.info(f'Example: {cdoc.mapping_set.mappings[0].subject_id}')
        # logging.info(f'Writing to {ofn}. Size={len(cdoc)}')
        msdf = to_mapping_set_dataframe(cdoc)
        with open(ofn, "w") as file:
            write_table(msdf, file)
        # write_tsv(msdf, ofn)


@main.command()
@input_argument
@output_option
@metadata_option
@click.option("-s", "--statsfile")
def cliquesummary(input: str, output: TextIO, metadata: str, statsfile: str):
    """Calculate summaries for each clique in a SSSOM file."""
    import yaml

    if metadata is None:
        doc = parse_sssom_table(input)
    else:
        meta_obj = yaml.safe_load(open(metadata))
        doc = parse_sssom_table(input, meta=meta_obj)
    df = summarize_cliques(doc)
    df.to_csv(output, sep="\t")
    if statsfile is None:
        logging.info(df.describe)
    else:
        df.describe().transpose().to_csv(statsfile, sep="\t")


@main.command()
@input_argument
@output_option
@transpose_option
@fields_option
def crosstab(input: str, output: TextIO, transpose: bool, fields: Tuple):
    """Write sssom summary cross-tabulated by categories."""
    df = remove_unmatched(parse_sssom_table(input).df)
    # df = parse(input)
    logging.info(f"#CROSSTAB ON {fields}")
    (f1, f2) = fields
    ct = pd.crosstab(df[f1], df[f2])
    if transpose:
        ct = ct.transpose()
    ct.to_csv(output, sep="\t")


@main.command()
@output_option
@transpose_option
@fields_option
@input_argument
def correlations(input: str, output: TextIO, transpose: bool, fields: Tuple):
    """Calculate correlations."""
    msdf = parse_sssom_table(input)
    df = remove_unmatched(msdf.df)
    # df = remove_unmatched(parse(input))
    if len(df) == 0:
        msg = "No matched entities in this dataset!"
        logging.error(msg)
        exit(1)

    logging.info(f"#CROSSTAB ON {fields}")
    (f1, f2) = fields

    logging.info(f"F1 {f1} UNIQUE: {df[f1].unique()}")
    logging.info(f"F2 {f2} UNIQUE: {df[f2].unique()}")

    ct = pd.crosstab(df[f1], df[f2])
    if transpose:
        ct = ct.transpose()

    chi2 = chi2_contingency(ct)

    logging.info(chi2)
    _, _, _, ndarray = chi2
    corr = pd.DataFrame(ndarray, index=ct.index, columns=ct.columns)
    corr.to_csv(output, sep="\t")

    tups = []
    for i, row in corr.iterrows():
        for j, v in row.items():
            logging.info(f"{i} x {j} = {v}")
            tups.append((v, i, j))
    tups = sorted(tups, key=lambda tx: tx[0])
    for t in tups:
        print(f"{t[0]}\t{t[1]}\t{t[2]}")


@main.command()
@click.argument("inputs", nargs=-1)
@click.option(
    "-R",
    "--reconcile",
    default=True,
    help="Boolean indicating the need for reconciliation of the SSSOM tsv file.",
)
@output_option
def merge(inputs: str, output: TextIO, reconcile: bool = True):
    """Merge multiple MappingSetDataFrames into one .

    if reconcile=True, then dedupe(remove redundant lower confidence mappings) and
    reconcile (if msdf contains a higher confidence _negative_ mapping,
    then remove lower confidence positive one. If confidence is the same,
    prefer HumanCurated. If both HumanCurated, prefer negative mapping).
    """  # noqa: DAR101
    msdfs = [parse_sssom_table(i) for i in inputs]
    merged_msdf = merge_msdf(*msdfs, reconcile=reconcile)
    write_table(merged_msdf, output)


@main.command()
@input_argument
@click.option("-m", "--mapping-file", help="Path to SSSOM file.")
@click.option("-I", "--input-format", default="turtle", help="Ontology input format.")
@click.option("-O", "--output-format", default="turtle", help="Ontology output format.")
@click.option(
    "--precedence",
    multiple=True,
    help="List of prefixes in order of precedence.",
)
@output_option
def rewire(
    input,
    mapping_file,
    precedence,
    output: TextIO,
    input_format,
    output_format,
):
    """Rewire an ontology using equivalent classes/properties from a mapping file.

    Example:
        sssom rewire -I xml  -i tests/data/cob.owl -m tests/data/cob-to-external.tsv --precedence PR

    # noqa: DAR101
    """
    msdf = parse_sssom_table(mapping_file)
    g = Graph()
    g.parse(input, format=input_format)
    rewire_graph(g, msdf, precedence=precedence)
    rdfstr = g.serialize(format=output_format)
    print(rdfstr, file=output)


@main.command()
@input_argument
@click.option(
    "-p",
    "--reconcile-prefix-file",
    help="Provide YAML file with prefix reconciliation information.",
)
@output_option
def reconcile_prefixes(input: str, reconcile_prefix_file: Path, output: TextIO):
    """
    Reconcile prefix_map based on provided YAML file.

    :param input: MappingSetDataFrame filename
    :param reconcile_prefix_file: YAML file containing the prefix reconcilation rules.
    :param output: Target file path.
    """
    msdf = parse_sssom_table(input)
    with open(reconcile_prefix_file, "rb") as rp_file:
        rp_dict = yaml.safe_load(rp_file)
    recon_msdf = reconcile_prefix_and_data(msdf, rp_dict)
    write_table(recon_msdf, output)


@main.command()
@input_argument
@output_option
@click.option(
    "-k",
    "--by-columns",
    default=True,
    help="Sort columns of DataFrame canonically.",
)
@click.option(
    "-r",
    "--by-rows",
    default=True,
    help="Sort rows by DataFrame column #1 (ascending).",
)
def sort(input: str, output: TextIO, by_columns: bool, by_rows: bool):
    """
    Sort DataFrame columns canonically.

    :param input: SSSOM TSV file.
    :param by_columns: Boolean flag to sort columns canonically.
    :param by_rows: Boolean flag to sort rows by column #1 (ascending order).
    :param output: SSSOM TSV file with columns sorted.
    """
    msdf = parse_sssom_table(input)
    msdf.df = sort_df_rows_columns(msdf.df, by_columns, by_rows)
    write_table(msdf, output)


if __name__ == "__main__":
    main()
