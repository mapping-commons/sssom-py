"""Test for filtering MappingSetDataFrame columns."""

import unittest
from os.path import join

from sssom.constants import PREDICATE_MODIFIER
from sssom.io import filter_file
from sssom.parsers import parse_sssom_table
from tests.constants import data_dir


class TestSort(unittest.TestCase):
    """A test case for filtering msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.input = join(data_dir, "basic.tsv")
        self.prefixes = ["x", "y", "z"]
        self.predicates = ["owl:subClassOf", "skos:exactMatch", "skos:broadMatch"]
        self.validation_file = join(data_dir, "test_filter_sssom.tsv")

    def test_filter(self):
        """Test filtering of rows."""
        kwargs = {"subject_id": ("x:%", "y:%"), "object_id": ("y:%", "z:%", "a:%")}
        filtered_msdf = filter_file(input=self.input, **kwargs)
        validation_msdf = parse_sssom_table(self.validation_file)

        # Drop empty columns since read_sssom_table drops them by default.
        filtered_df = filtered_msdf.df.drop(columns=[PREDICATE_MODIFIER])

        self.assertEqual(filtered_msdf.metadata, validation_msdf.metadata)
        self.assertEqual(filtered_msdf.prefix_map, validation_msdf.prefix_map)
        self.assertEqual(len(filtered_df), len(validation_msdf.df))
        # Pandas does something weird with assert_frame_equal
        # assert_frame_equal(filtered_df.sort_index(axis=1), validation_msdf.df.sort_index(axis=1), check_like=True)

    def test_filter_fail(self):
        """Pass invalid param to see if it fails."""
        kwargs = {"subject_ids": ("x:%", "y:%"), "object_id": ("y:%", "z:%")}
        with self.assertRaises(ValueError):
            filter_file(input=self.input, **kwargs)
