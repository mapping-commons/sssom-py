"""Initializing imports and constants."""

try:
    from importlib import metadata  # Python >=3.8

    __version__ = metadata.version(__name__)
except ImportError:  # for Python<3.8
    import importlib_metadata

    __version__ = importlib_metadata.version(__name__)

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
