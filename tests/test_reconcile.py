"""Tests for reconcilation utilities."""

import unittest

from sssom import filter_redundant_rows
from sssom.parsers import parse_sssom_table
from sssom.util import deal_with_negation, merge_msdf
from tests.constants import data_dir


class TestReconcile(unittest.TestCase):
    """A test case for reconcilation utilities."""

    def setUp(self) -> None:
        """Test up the test case with the third basic example."""
        self.msdf1 = parse_sssom_table(data_dir / "basic3.tsv")
        self.msdf2 = parse_sssom_table(data_dir / "basic7.tsv")

    def test_filter(self):
        """Test filtering returns the right number of rows."""
        df1 = filter_redundant_rows(self.msdf1.df)
        self.assertEqual(10, len(df1.index))
        df2 = filter_redundant_rows(self.msdf2.df)
        self.assertEqual(18, len(df2.index))

    def test_deal_with_negation(self):
        """Test handling negating returns the right number of rows."""
        df1 = deal_with_negation(self.msdf1.df)
        self.assertEqual(7, len(df1.index))
        df2 = deal_with_negation(self.msdf2.df)
        self.assertEqual(5, len(df2.index))

    def test_merge(self):
        """Test merging two tables."""
        msdf3 = parse_sssom_table(data_dir / "basic.tsv")
        merged_msdf1 = merge_msdf(self.msdf1, msdf3)

        self.assertEqual(98, len(merged_msdf1.df))

        merged_msdf2 = merge_msdf(self.msdf2, msdf3)
        self.assertEqual(107, len(merged_msdf2.df))

        merged_msdf3 = merge_msdf(self.msdf1, self.msdf2)
        self.assertEqual(18, len(merged_msdf3.df))

    def test_merge_no_reconcile(self):
        """Test merging two tables without reconciliation."""
        msdf1 = parse_sssom_table(data_dir / "basic4.tsv")
        msdf2 = parse_sssom_table(data_dir / "basic5.tsv")

        merged_msdf = merge_msdf(msdf1, msdf2, reconcile=False)

        self.assertEqual(53, len(msdf1.df))
        self.assertEqual(53, len(msdf2.df))
        self.assertEqual(len(merged_msdf.df), (len(msdf1.df) + len(msdf2.df)))
