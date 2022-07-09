"""Test for filtering MappingSetDataFrame columns."""

import unittest
from os.path import join

import numpy as np
from pandas.util.testing import assert_frame_equal

from sssom.io import filter_file
from sssom.parsers import read_sssom_table
from tests.constants import data_dir


class TestSort(unittest.TestCase):
    """A test case for filtering msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.input = join(data_dir, "basic.tsv")
        self.prefixes = ["x", "y", "z"]
        self.predicates = ["owl:subClassOf", "skos:exactMatch", "skos:broadMatch"]
        self.validation_file = join(data_dir, "test_filter_sssom.tsv")

    def test_sort(self):
        """Test sorting of columns."""
        filtered_msdf = filter_file(
            input=self.input, prefix=self.prefixes, predicate=self.predicates
        )
        validation_msdf = read_sssom_table(self.validation_file)

        # Drop empty columns since read_sssom_table drops them by default.
        filtered_df = filtered_msdf.df
        filtered_df.replace("", np.NAN, inplace=True)
        filtered_df.dropna(how="all", axis=1, inplace=True)
        filtered_df.fillna("", inplace=True)
        validation_msdf.df.fillna("", inplace=True)
        filtered_df = filtered_df.reset_index(drop=True)
        self.assertEqual(filtered_msdf.metadata, validation_msdf.metadata)
        self.assertEqual(filtered_msdf.prefix_map, validation_msdf.prefix_map)
        assert_frame_equal(filtered_df, validation_msdf.df)
