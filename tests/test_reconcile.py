"""Tests for reconcilation utilities."""

import unittest

from sssom import filter_redundant_rows
from sssom.parsers import read_sssom_table
from sssom.util import deal_with_negation, merge_msdf
from tests.constants import data_dir


class TestReconcile(unittest.TestCase):
    """A test case for reconcilation utilities."""

    def setUp(self) -> None:
        """Test up the test case with the third basic example."""
        self.msdf = read_sssom_table(f"{data_dir}/basic3.tsv")

    def test_filter(self):
        """Test filtering returns the right number of rows."""
        df = filter_redundant_rows(self.msdf.df)
        self.assertEqual(10, len(df.index))

    def test_deal_with_negation(self):
        """Test handling negating returns the right number of rows."""
        df = deal_with_negation(self.msdf.df)
        self.assertEqual(7, len(df.index))

    def test_merge(self):
        """Test merging two tables."""
        msdf1 = read_sssom_table(f"{data_dir}/basic.tsv")
        msdf2 = read_sssom_table(f"{data_dir}/basic2.tsv")

        merged_msdf = merge_msdf(msdf1=msdf1, msdf2=msdf2)

        self.assertEqual(95, len(merged_msdf.df))

    def test_merge_no_reconcile(self):
        """Test merging two tables without reconciliation."""
        msdf1 = read_sssom_table(f"{data_dir}/basic4.tsv")
        msdf2 = read_sssom_table(f"{data_dir}/basic5.tsv")

        merged_msdf = merge_msdf(msdf1=msdf1, msdf2=msdf2, reconcile=False)

        self.assertEqual(53, len(msdf1.df))
        self.assertEqual(53, len(msdf2.df))
        self.assertEqual(len(merged_msdf.df), (len(msdf1.df) + len(msdf2.df)))
