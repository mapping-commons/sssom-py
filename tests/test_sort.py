"""Test for sorting MappingSetDataFrame columns."""

import unittest

from sssom.parsers import parse_sssom_table
from sssom.schema import SCHEMA_DICT
from sssom.util import sort_df_rows_columns
from tests.constants import data_dir


class TestSort(unittest.TestCase):
    """A test case for sorting msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.msdf = parse_sssom_table(f"{data_dir}/basic6.tsv")

    def test_sort(self):
        """Test sorting of columns."""
        new_df = sort_df_rows_columns(self.msdf.df)
        column_sequence = [
            col for col in SCHEMA_DICT["slots"].keys() if col in new_df.columns
        ]
        self.assertListEqual(column_sequence, list(new_df.columns))
