import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, TextIO, Tuple

import click
import pandas as pd
import yaml
from pandasql import sqldf
from rdflib import Graph
from scipy.stats import chi2_contingency

from .cliques import split_into_cliques, summarize_cliques
from .io import convert_file, parse_file, split_file, validate_file
from .parsers import read_sssom_table
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
    remove_unmatched,
    smart_open,
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
    "-o", "--output", help="Output file, e.g. a SSSOM tsv file."
)
improved_output_option = click.option(
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
    "-d", "--output-directory", type=click.Path(), help="Output directory path."
)
metadata_option = click.option(
    "-m",
    "--metadata",
    required=False,
    type=click.Path(),
    help="The path to a file containing the sssom metadata (including curie_map) to be used.",
)
transpose_option = click.option("-t", "--transpose/--no-transpose", default=False)
fields_option = click.option(
    "-F",
    "--fields",
    nargs=2,
    default=("subject_category", "object_category"),
    help="Fields.",
)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
def main(verbose: int, quiet: bool):
    """Main."""
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
@improved_output_option
@output_format_option
def convert(input: str, output: TextIO, output_format: str):
    """Convert file (currently only supports conversion to RDF)

    Example:
        sssom covert --input my.sssom.tsv --output-format rdfxml --output my.sssom.owl
    """
    convert_file(input_path=input, output=output, output_format=output_format)


# Input and metadata would be files (file paths). Check if exists.
@main.command()
@input_argument
@input_format_option
@metadata_option
@click.option(
    "-C",
    "--curie-map-mode",
    default="metadata_only",
    show_default=True,
    required=True,
    type=click.Choice(
        ["metadata_only", "sssom_default_only", "merged"], case_sensitive=False
    ),
    help="Defines wether the curie map in the metadata should be extended or replaced with "
    "the SSSOM default curie map. Must be one of metadata_only, sssom_default_only, merged",
)
@click.option(
    "-p",
    "--clean-prefixes",
    default=True,
    is_flag=True,
    required=True,
    help="If True (default), records with unknown prefixes are removed from the SSSOM file.",
)
@output_option
def parse(
    input: str,
    input_format: str,
    metadata: str,
    curie_map_mode: str,
    clean_prefixes: bool,
    output: str,
):
    """Parses a file in one of the supported formats (such as obographs) into an SSSOM TSV file.

    Args:

        input (str): The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
        input_format (str): The string denoting the input format.
        metadata (str): The path to a file containing the sssom metadata (including curie_map) to be used during parse.
        curie_map_mode (str): Curie map mode.
        clean_prefixes (bool): If True (default), records with unknown prefixes are removed from the SSSOM file.
        output (str): The path to the SSSOM TSV output file.

    Returns:

        None.
    """

    parse_file(
        input_path=input,
        output_path=output,
        input_format=input_format,
        metadata_path=metadata,
        curie_map_mode=curie_map_mode,
        clean_prefixes=clean_prefixes,
    )


@main.command()
@input_argument
def validate(input: str):
    """Takes 1 sssom file as input and produce an error report

    Args:

        input (str): Input file. For e.g.: SSSOM tsv file

    Returns:

        None.
    """

    validate_file(input_path=input)


@main.command()
@input_argument
@output_directory_option
def split(input: str, output_directory: str):
    """Split input file into multiple output broken down by prefixes

    Args:

        input (str): Input file. For e.g.: SSSOM tsv file.
        output_directory (str): Output directory path.

    Returns:

        None.
    """

    split_file(input_path=input, output_directory=output_directory)


@main.command()
@input_argument
@output_option
@click.option("-W", "--inverse-factor", help="Inverse factor.")
def ptable(input=None, output=None, inverse_factor=None):
    """Write ptable (kboom/boomer input) should maybe move to boomer (but for now it can live here, so cjm can tweak

    Args:

        input (str): Input file. For e.g.: SSSOM tsv file.
        output (str): the Output file
        inverse_factor (str): Inverse factor.

    Returns:

        None

    """
    logging.warning(
        f"inverse_factor ({inverse_factor}) ignored by this method, not implemented yet."
    )
    msdf = read_sssom_table(input)
    # df = parse(input)
    df = collapse(msdf.df)
    # , priors=list(priors)
    rows = dataframe_to_ptable(df)

    with smart_open(output) as fh:
        for row in rows:
            print("\t".join(row), file=fh)


@main.command()
@input_argument
@improved_output_option
def dedupe(input: str, output: TextIO):
    """Remove lower confidence duplicate lines.

    Args:

        input (str): Input file. For e.g.: SSSOM tsv file.
        output (str): Output TSV/SSSOM file.

    Returns:

        None.
    """
    # df = parse(input)
    msdf = read_sssom_table(input)
    df = filter_redundant_rows(msdf.df)
    msdf_out = MappingSetDataFrame(
        df=df, prefixmap=msdf.prefixmap, metadata=msdf.metadata
    )
    # df.to_csv(output, sep="\t", index=False)
    write_table(msdf_out, output)


@main.command()
@click.option("-q", "--query", help='SQL query. Use "df" as table name.')
@click.argument("inputs", nargs=-1)
@output_option
def dosql(query: str, inputs: List[str], output: str):
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

    Args:
        query (str): SQL query. Use "df" as table name.
        inputs (List): List of input files.
        output (str): Output TSV/SSSOM file.

    Returns:
        None.

    """
    # should start with from_tsv and MOST should return write_sssom
    n = 1
    while len(inputs) >= n:
        fn = inputs[n - 1]
        df = read_sssom_table(fn).df
        # df = parse(fn)
        globals()[f"df{n}"] = df
        tn = re.sub("[.].*", "", Path(fn).stem).lower()
        globals()[tn] = df
        n += 1
    df = sqldf(query)
    if output is None:
        print(df.to_csv(sep="\t", index=False))
    else:
        df.to_csv(output, sep="\t", index=False)


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
@improved_output_option
def sparql(
    url: str,
    config,
    graph: str,
    limit: int,
    object_labels: bool,
    prefix: List[Dict[str, str]],
    output: TextIO,
):
    """Run a SPARQL query.

    Args:

        url (str):
        config (str):
        graph (str):
        limit (int):
        object_labels (bool):
        prefix (List):
        output (str): Output TSV/SSSOM file.


    Returns:

        None.
    """

    endpoint = EndpointConfig()
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
        if endpoint.curie_map is None:
            endpoint.curie_map = {}
        for k, v in prefix:
            endpoint.curie_map[k] = v
    msdf = query_mappings(endpoint)
    write_table(msdf, output)


@main.command()
@improved_output_option
@click.argument("inputs", nargs=2)
def diff(inputs: Tuple[str, str], output: TextIO):
    """
    Compare two SSSOM files.
    The output is a new SSSOM file with the union of all mappings, and
    injected comments indicating uniqueness to set1 or set2.

    Args:

        inputs (tuple): A tuple of input filenames
        output (str): Output TSV/SSSOM file.

    Returns:

        None.
    """

    (input1, input2) = inputs
    # df1 = parse(input1)
    # df2 = parse(input2)
    msdf1 = read_sssom_table(input1)
    msdf2 = read_sssom_table(input2)
    d = compare_dataframes(msdf1.df, msdf2.df)
    logging.info(
        f"COMMON: {len(d.common_tuples)} UNIQUE_1: {len(d.unique_tuples1)} UNIQUE_2: {len(d.unique_tuples2)}"
    )
    d.combined_dataframe.to_csv(output, sep="\t", index=False)


@main.command()
@output_directory_option
@click.argument("inputs", nargs=-1)
def partition(inputs: List[str], output_directory: str):
    """Partitions an SSSOM file into multiple files, where each
    file is a strongly connected component.

    Args:

        inputs (List): List of input files.
        output_directory (str): Output directory path.

    Returns:

        None.
    """

    docs = [read_sssom_table(input) for input in inputs]
    doc = docs.pop()
    """for d2 in docs:
        doc.mapping_set.mappings += d2.mapping_set.mappings"""
    cliquedocs = split_into_cliques(doc)
    n = 0
    for cdoc in cliquedocs:
        n += 1
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
def cliquesummary(input: str, output: str, metadata: str, statsfile: str):
    """Partitions an SSSOM file into multiple files, where each
    file is a strongly connected component.

    The data dictionary for the output is in cliquesummary.yaml

    Args:

        input (str): Input file. For e.g.: SSSOM tsv file
        output (str): Output TSV/SSSOM file
        metadata (str): Metadata.
        statsfile (str): Stats File.

    Returns:

        None.
    """
    import yaml

    if metadata is None:
        doc = read_sssom_table(input)
    else:
        meta_obj = yaml.safe_load(open(metadata))
        doc = read_sssom_table(input, meta=meta_obj)
    df = summarize_cliques(doc)
    df.to_csv(output, sep="\t")
    if statsfile is None:
        logging.info(df.describe)
    else:
        df.describe().transpose().to_csv(statsfile, sep="\t")


@main.command()
@input_argument
@improved_output_option
@transpose_option
@fields_option
def crosstab(input: str, output: TextIO, transpose: bool, fields: Tuple):
    """
    Write sssom summary cross-tabulated by categories.

    Args:

        input (str): Input file. For e.g.: SSSOM tsv file
        output (str): Output TSV/SSSOM file
        transpose (bool): Yes/No
        fields (Type):

    Returns:

        None.
    """

    df = remove_unmatched(read_sssom_table(input).df)
    # df = parse(input)
    logging.info(f"#CROSSTAB ON {fields}")
    (f1, f2) = fields
    ct = pd.crosstab(df[f1], df[f2])
    if transpose:
        ct = ct.transpose()
    ct.to_csv(output, sep="\t")


@main.command()
@improved_output_option
@transpose_option
@fields_option
@input_argument
def correlations(input: str, output: TextIO, transpose: bool, fields: Tuple):
    """Correlations

    Args:

        input (str): Input file. For e.g.: SSSOM tsv file
        output (str): Output TSV/SSSOM file
        transpose (bool): Yes/No
        fields (Type): Fields.

    Returns:

        None.
    """
    msdf = read_sssom_table(input)
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
    "-r",
    "--reconcile",
    default=True,
    help="Boolean indicating the need for reconciliation of the SSSOM tsv file.",
)
@improved_output_option
def merge(inputs: Tuple[str, str], output: TextIO, reconcile: bool = True):
    """
    Merging msdf2 into msdf1,
        if reconcile=True, then dedupe(remove redundant lower confidence mappings) and
            reconcile (if msdf contains a higher confidence _negative_ mapping,
            then remove lower confidence positive one. If confidence is the same,
            prefer HumanCurated. If both HumanCurated, prefer negative mapping).

        Args:
            inputs: All MappingSetDataFrames that need to be merged
            output: SSSOM file containing the merged output
            reconcile (bool, optional): [description]. Defaults to True.

        Returns:

    """
    (input1, input2) = inputs[:2]
    msdf1 = read_sssom_table(input1)
    msdf2 = read_sssom_table(input2)
    merged_msdf = merge_msdf(msdf1, msdf2, reconcile)

    # If > 2 input files, iterate through each one
    # and merge them into the merged file above
    if len(inputs) > 2:
        for input_file in inputs[2:]:
            msdf1 = merged_msdf
            msdf2 = read_sssom_table(input_file)
            merged_msdf = merge_msdf(msdf1, msdf2, reconcile)

    # Export MappingSetDataFrame into a TSV
    write_table(merged_msdf, output)


@main.command()
@input_argument
@click.option("-m", "--mapping-file", help="Path to SSSOM file.")
@click.option("-I", "--input-format", default="turtle", help="Ontology input format.")
@click.option("-O", "--output-format", default="turtle", help="Ontology output format.")
@click.option(
    "--precedence", multiple=True, help="List of prefixes in order of precedence."
)
@improved_output_option
def rewire(
    input,
    mapping_file,
    precedence,
    output: TextIO,
    input_format,
    output_format,
):
    """Rewire an ontology using equivalence predicates from a mapping file

    Example:

        sssom rewire -I xml  -i tests/data/cob.owl -m tests/data/cob-to-external.tsv --precedence PR
    """
    msdf = read_sssom_table(mapping_file)
    g = Graph()
    g.parse(input, format=input_format)
    rewire_graph(g, msdf, precedence=precedence)
    rdfstr = g.serialize(format=output_format).decode()
    print(rdfstr, file=output)


if __name__ == "__main__":
    main()
