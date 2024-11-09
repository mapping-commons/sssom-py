"""Command line interface for SSSOM.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m sssom`` python will execute``__main__.py`` as a script. That means there won't be any
  ``sssom.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``sssom.__main__`` in ``sys.modules`` .

.. seealso:: https://click.palletsprojects.com/en/8.0.x/setuptools/
"""

import logging as _logging
import os
import sys
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, List, Optional, TextIO, Tuple, get_args

import click
import curies
import pandas as pd
import yaml
from curies import Converter
from rdflib import Graph
from scipy.stats import chi2_contingency

from sssom.constants import (
    DEFAULT_VALIDATION_TYPES,
    MergeMode,
    SchemaValidationType,
    _get_sssom_schema_object,
)

from . import __version__
from .cliques import split_into_cliques, summarize_cliques
from .io import (
    annotate_file,
    convert_file,
    filter_file,
    parse_file,
    run_sql_query,
    split_file,
    validate_file,
)
from .parsers import PARSING_FUNCTIONS, parse_sssom_table
from .rdf_util import rewire_graph
from .sparql_util import EndpointConfig, query_mappings
from .util import (
    MappingSetDataFrame,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    invert_mappings,
    merge_msdf,
    pandas_set_no_silent_downcasting,
    reconcile_prefix_and_data,
    remove_unmatched,
    sort_df_rows_columns,
    to_mapping_set_dataframe,
)
from .writers import WRITER_FUNCTIONS, write_table

logging = _logging.getLogger(__name__)

SSSOM_SV_OBJECT = _get_sssom_schema_object()


# Click input options common across commands
input_argument = click.argument("input", required=True, type=click.Path())

input_format_option = click.option(
    "-I",
    "--input-format",
    help="The string denoting the input format.",
    type=click.Choice(PARSING_FUNCTIONS),
)
output_option = click.option(
    "-o",
    "--output",
    help="Path of SSSOM output file.",
    type=click.File(mode="w"),
    default=sys.stdout,
)
output_format_option = click.option(
    "-O",
    "--output-format",
    help="Desired output format.",
    type=click.Choice(WRITER_FUNCTIONS),
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
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """Run the SSSOM CLI."""
    logger = _logging.getLogger()

    pandas_set_no_silent_downcasting()

    if verbose >= 2:
        logger.setLevel(level=_logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=_logging.INFO)
    else:
        logger.setLevel(level=_logging.WARNING)
    if quiet:
        logger.setLevel(level=_logging.ERROR)


@main.command()
@click.argument("subcommand")
@click.pass_context
def help(ctx, subcommand):
    """Echoes help for subcommands."""
    subcommand_obj = main.get_command(ctx, subcommand)
    if subcommand_obj is None:
        click.echo("The command you seek help with does not exist.")
    else:
        click.echo(subcommand_obj.get_help(ctx))


@main.command()
@input_argument
@output_option
@output_format_option
def convert(input: str, output: TextIO, output_format: str):
    """Convert a file.

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
    type=click.Choice(get_args(MergeMode), case_sensitive=False),
    help="Defines whether the prefix map in the metadata should be extended or replaced with "
    "the SSSOM default prefix map.",
)
@click.option(
    "-p",
    "--clean-prefixes / --no-clean-prefixes",
    default=True,
    is_flag=True,
    required=True,
    help="If True (default), records with unknown prefixes are removed from the SSSOM file.",
)
@click.option(
    "--strict-clean-prefixes / --no-strict-clean-prefixes",
    default=True,
    is_flag=True,
    help="If True (default), `clean_prefixes(strict = True)`.",
)
@click.option(
    "-E",
    "--embedded-mode / --non-embedded-mode",
    default=True,
    is_flag=True,
    help="If False, the resultant SSSOM file will be saved\
        in the 'filename'.tsv provided by -o/--output option\
        AND the metadata gets saved in the 'filename'.yml.",
)
@predicate_filter_option
@output_option
def parse(
    input: str,
    input_format: str,
    metadata: str,
    prefix_map_mode: MergeMode,
    clean_prefixes: bool,
    strict_clean_prefixes: bool,
    output: TextIO,
    embedded_mode: bool,
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
        strict_clean_prefixes=strict_clean_prefixes,
        embedded_mode=embedded_mode,
        mapping_predicate_filter=mapping_predicate_filter,
    )


@main.command()
@input_argument
@click.option(
    "--validation-types",
    "-V",
    type=click.Choice(SchemaValidationType),
    multiple=True,
    default=DEFAULT_VALIDATION_TYPES,
)
def validate(input: str, validation_types: List[SchemaValidationType]):
    """Produce an error report for an SSSOM file."""
    validation_type_list = [t for t in validation_types]
    validate_file(input_path=input, validation_types=validation_type_list)


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
@click.option(
    "--default-confidence",
    type=click.FloatRange(0, 1),
    help="Default confidence to be assigned if absent.",
)
def ptable(input, output: TextIO, inverse_factor: float, default_confidence: float):
    """Convert an SSSOM file to a ptable for kboom/`boomer <https://github.com/INCATools/boomer>`_."""
    # TODO should maybe move to boomer (but for now it can live here, so cjm can tweak
    msdf = parse_sssom_table(input)
    rows = dataframe_to_ptable(
        msdf.df, inverse_factor=inverse_factor, default_confidence=default_confidence
    )
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
    msdf_out = MappingSetDataFrame.with_converter(
        df=df, converter=msdf.converter, metadata=msdf.metadata
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
        sssom dosql -Q "SELECT * FROM df1 WHERE confidence>0.5 ORDER BY confidence" my.sssom.tsv

    Example:
        `sssom dosql -Q "SELECT file1.*,file2.object_id AS ext_object_id, file2.object_label AS ext_object_label \
        FROM file1 INNER JOIN file2 WHERE file1.object_id = file2.subject_id" FROM file1.sssom.tsv file2.sssom.tsv`
    """  # noqa: DAR101
    # should start with from_tsv and MOST should return write_sssom
    run_sql_query(query=query, inputs=inputs, output=output)
    # n = 1
    # new_msdf = MappingSetDataFrame()
    # while len(inputs) >= n:
    #     fn = inputs[n - 1]
    #     msdf = parse_sssom_table(fn)
    #     df = msdf.df
    #     # df = parse(fn)
    #     globals()[f"df{n}"] = df
    #     tn = re.sub("[.].*", "", Path(fn).stem).lower()
    #     globals()[tn] = df
    #     n += 1

    # new_df = sqldf(query)
    # new_msdf.df = new_df
    # new_msdf.prefix_map = msdf.prefix_map
    # new_msdf.metadata = msdf.metadata
    # write_table(new_msdf, output)


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
    prefix: List[Tuple[str, str]],
    output: TextIO,
):
    """Run a SPARQL query."""
    # FIXME this usage needs _serious_ refactoring
    endpoint = EndpointConfig(converter=Converter.from_prefix_map(dict(prefix)))  # type: ignore
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

    prefix_map_list = [msdf1, msdf2]
    converter = curies.chain(m.converter for m in prefix_map_list)
    msdf = MappingSetDataFrame.with_converter(
        df=d.combined_dataframe.drop_duplicates(), converter=converter
    )
    msdf.metadata[  # type:ignore
        "comment"
    ] = (
        f"Diff between {input1} and {input2}. See comment column for information."
    )
    write_table(msdf, output)


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
def crosstab(input: str, output: TextIO, transpose: bool, fields: Tuple[str, str]):
    """Write sssom summary cross-tabulated by categories."""
    df = remove_unmatched(parse_sssom_table(input).df)
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
def correlations(input: str, output: TextIO, transpose: bool, fields: Tuple[str, str]):
    """Calculate correlations."""
    msdf = parse_sssom_table(input)
    df = remove_unmatched(msdf.df)
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
    expected_frequencies_df = pd.DataFrame(chi2[3], index=ct.index, columns=ct.columns)
    expected_frequencies_df.to_csv(output, sep="\t")

    rows = []
    for i, row in expected_frequencies_df.iterrows():
        for j, v in row.items():
            logging.info(f"{i} x {j} = {v}")
            rows.append((v, i, j))
    for row in sorted(rows, key=itemgetter(0)):
        print(*row, sep="\t")


@main.command()
@click.argument("inputs", nargs=-1)
@click.option(
    "-R",
    "--reconcile",
    default=False,
    help="Boolean indicating the need for reconciliation of the SSSOM tsv file.",
)
@output_option
def merge(inputs: str, output: TextIO, reconcile: bool = False):
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


# @main.command()
# @input_argument
# @click.option(
#     "-P",
#     "--prefix",
#     multiple=True,
#     help="Prefixes that need to be filtered.",
# )
# @click.option(
#     "-D",
#     "--predicate",
#     multiple=True,
#     help="Predicates that need to be filtered.",
# )
# @output_option
# def filter(input: str, output: TextIO, prefix: tuple, predicate: tuple):
#     """Filter mapping file based on prefix and predicates provided.

#     :param input: Input mapping file (tsv)
#     :param output: SSSOM TSV file.
#     :param prefix: Prefixes to be retained.
#     :param predicate: Predicates to be retained.
#     """
#     filtered_msdf = filter_file(input=input, prefix=prefix, predicate=predicate)
#     write_table(msdf=filtered_msdf, file=output)


def dynamically_generate_sssom_options(options) -> Callable[[Any], Any]:
    """Dynamically generate click options.

    :param options: List of all possible options.
    :return: Click options deduced from user input into parameters.
    """

    def _decorator(f):
        for sssom_slot in reversed(options):
            click.option("--" + sssom_slot, multiple=True)(f)
        return f

    return _decorator


@main.command()
@input_argument
@output_option
@dynamically_generate_sssom_options(SSSOM_SV_OBJECT.mapping_slots)
def filter(input: str, output: TextIO, **kwargs):
    """Filter a dataframe by dynamically generating queries based on user input.

    e.g. sssom filter --subject_id x:% --subject_id y:% --object_id y:% --object_id z:% tests/data/basic.tsv

    yields the query:

    "SELECT * FROM df WHERE (subject_id LIKE 'x:%'  OR subject_id LIKE 'y:%')
     AND (object_id LIKE 'y:%'  OR object_id LIKE 'z:%') " and displays the output.

    :param input: DataFrame to be queried over.
    :param output: Output location.
    :param kwargs: Filter options provided by user which generate queries (e.g.: --subject_id x:%).
    """
    filter_file(input=input, output=output, **kwargs)


@main.command()
@input_argument
@output_option
# TODO Revist the option below.
#  If a multivalued slot needs to be partially preserved,
#  the users will need to type the ones they need and
#  set --replace-multivalued to True.
@click.option(
    "--replace-multivalued",
    default=False,
    type=bool,
    help="Multivalued slots should be replaced or not. [default: False]",
)
@dynamically_generate_sssom_options(SSSOM_SV_OBJECT.mapping_set_slots)
def annotate(input: str, output: TextIO, replace_multivalued: bool, **kwargs):
    """Annotate metadata of a mapping set.

    :param input: Input path of the SSSOM tsv file.
    :param output: Output location.
    :param replace_multivalued: Multivalued slots should be
        replaced or not, defaults to False
    :param kwargs: Options provided by user
        which are added to the metadata (e.g.: --mapping_set_id http://example.org/abcd)
    """
    annotate_file(input=input, output=output, replace_multivalued=replace_multivalued, **kwargs)


@main.command()
@input_argument
@click.option(
    "--remove-map",
    type=click.Path(),
    help="Mapping file path that needs to be removed from input.",
)
@output_option
def remove(input: str, output: TextIO, remove_map: str):
    """Remove mappings from an input mapping.

    :param input: Input SSSOM tsv file.
    :param output: Output path.
    :param remove_map: Mapping to be removed.
    """
    input_msdf = parse_sssom_table(input)
    remove_msdf = parse_sssom_table(remove_map)
    input_msdf.remove_mappings(remove_msdf)
    write_table(input_msdf, output)


@main.command()
@input_argument
@output_option
@click.option(
    "-P",
    "--subject-prefix",
    required=False,
    help="Invert subject_id and object_id such that all subject_ids have the same prefix.",
)
@click.option(
    "--merge-inverted/--no-merge-inverted",
    default=True,
    is_flag=True,
    help="If True (default), add inverted mappings to the input mapping set, else, just return inverted mappings as a separate mapping set.",
)
@click.option(
    "--update-justification/--no-update-justification",
    default=True,
    is_flag=True,
    help="If True (default), the justification is updated to 'sempav:MappingInversion', else it is left as it is.",
)
@click.option("--inverse-map", help="Path to file that contains the inverse predicate dictionary.")
def invert(
    input: str,
    output: TextIO,
    subject_prefix: Optional[str],
    merge_inverted: bool,
    update_justification: bool,
    inverse_map: TextIO,
):
    """
    Invert subject and object IDs such that all subjects have the prefix provided.

    :param input: SSSOM TSV file.
    :param subject_prefix: Prefix of all subject_ids.
    :param merge_inverted: If True (default), add inverted dataframe to input else,
                          just return inverted data.
    :param update_justification: If True (default), the justification is updated to "sempav:MappingInversion",
                          else it is left as it is.
    :param inverse_map: YAML file providing the inverse mapping for predicates.
    :param output: SSSOM TSV file with columns sorted.
    """
    msdf = parse_sssom_table(input)
    if inverse_map:
        with open(inverse_map, "r") as im:  # type: ignore
            inverse_dictionary = yaml.safe_load(im)
        inverse_predicate_map = inverse_dictionary["inverse_predicate_map"]
    else:
        inverse_predicate_map = None

    msdf.df = invert_mappings(
        df=msdf.df,
        subject_prefix=subject_prefix,
        merge_inverted=merge_inverted,
        update_justification=update_justification,
        predicate_invert_dictionary=inverse_predicate_map,
    )
    write_table(msdf, output)


if __name__ == "__main__":
    main()
