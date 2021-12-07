"""Initializing imports and constants."""
import os

from .sssom_datamodel import slots  # noqa:401
from .sssom_datamodel import Mapping, MappingSet  # noqa:401
from .util import (  # noqa:401
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    parse,
)

SCHEMA_PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCHEMA_YAML = os.path.join(SCHEMA_PARENT_PATH, "schema/sssom.yaml")
