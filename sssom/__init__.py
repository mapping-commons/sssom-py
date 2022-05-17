"""Initializing imports and constants."""

import importlib.metadata

from .sssom_datamodel import slots  # noqa:401
from .sssom_datamodel import Mapping, MappingSet  # noqa:401
from .util import (  # noqa:401
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    parse,
    reconcile_prefix_and_data,
)

__version__ = importlib.metadata.version(__name__)
