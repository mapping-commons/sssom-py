"""sssom-py package."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover

from sssom_schema import Mapping, MappingSet, slots  # noqa:401

from sssom.io import get_metadata_and_prefix_map  # noqa:401
from sssom.sssom_document import MappingSetDocument  # noqa:401
from sssom.util import (  # noqa:401
    MappingSetDataFrame,
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    parse,
    reconcile_prefix_and_data,
)

from .constants import generate_mapping_set_id, get_default_metadata  # noqa:401
