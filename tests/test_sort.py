"""Test for sorting MappingSetDataFrame columns."""

import unittest

from sssom.constants import SCHEMA_DICT
from sssom.parsers import read_sssom_table
from sssom.util import sort_df_columns
from tests.constants import data_dir


class TestSort(unittest.TestCase):
    """A test case for sorting msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.msdf = read_sssom_table(f"{data_dir}/basic6.tsv")

    def test_sort(self):
        """Test sorting of columns."""
        new_df = sort_df_columns(self.msdf.df)
        column_sequence = [
            col for col in SCHEMA_DICT["slots"].keys() if col in new_df.columns
        ]
        self.assertListEqual(column_sequence, list(new_df.columns))
