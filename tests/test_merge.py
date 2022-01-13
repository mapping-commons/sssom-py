"""Test for merging MappingSetDataFrames."""

import unittest

from sssom.parsers import read_sssom_table
from sssom.util import merge_msdf
from tests.constants import data_dir


class TestMerge(unittest.TestCase):
    """A test case for merging msdfs."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        msdf1 = read_sssom_table(f"{data_dir}/basic.tsv")
        msdf2 = read_sssom_table(f"{data_dir}/basic2.tsv")
        msdf3 = read_sssom_table(f"{data_dir}/basic3.tsv")
        self.msdfs = [msdf1, msdf2, msdf3]

    def test_merge(self):
        """Test merging of multiple msdfs."""
        merged_msdf = merge_msdf(self.msdfs)
        self.assertEqual(128, len(merged_msdf.df))
