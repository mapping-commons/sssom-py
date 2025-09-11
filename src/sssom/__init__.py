"""sssom-py package."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover

from sssom_schema import Mapping, MappingSet, slots

from sssom.io import get_metadata_and_prefix_map
from sssom.sssom_document import MappingSetDocument
from sssom.util import (
    MappingSetDataFrame,
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    reconcile_prefix_and_data,
)

from .constants import generate_mapping_set_id, get_default_metadata
from .parsers import parse_csv, parse_sssom_table, parse_tsv
from .writers import write_json, write_owl, write_rdf, write_tsv

__all__ = [
    "write_json",
    "write_owl",
    "write_rdf",
    "write_tsv",
    "parse_csv",
    "generate_mapping_set_id",
    "get_default_metadata",
    "parse_sssom_table",
    "parse_tsv",
    "Mapping",
    "MappingSet",
    "MappingSetDocument",
    "MappingSetDataFrame",
    "slots",
    "get_metadata_and_prefix_map",
    "collapse",
    "compare_dataframes",
    "dataframe_to_ptable",
    "filter_redundant_rows",
    "group_mappings",
    "reconcile_prefix_and_data",
]
