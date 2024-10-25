"""Tests for reconcilation utilities."""

import unittest

from sssom.parsers import parse_sssom_table
from sssom.util import deal_with_negation, filter_redundant_rows, merge_msdf
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

        # Create a new dataframe with the confidence column having NaN values
        import numpy as np

        self.msdf1.df["confidence"] = np.nan
        df3 = filter_redundant_rows(self.msdf1.df)
        self.assertEqual(11, len(df3.index))

    def test_deal_with_negation(self):
        """Test handling negating returns the right number of rows."""
        df1 = deal_with_negation(self.msdf1.df)
        self.assertEqual(8, len(df1.index))
        df2 = deal_with_negation(self.msdf2.df)
        self.assertEqual(12, len(df2.index))

    def test_merge(self):
        """Test merging two tables."""
        msdf3 = parse_sssom_table(data_dir / "basic.tsv")
        merged_msdf1 = merge_msdf(self.msdf1, msdf3)

        self.assertEqual(152, len(merged_msdf1.df))

        merged_msdf2 = merge_msdf(self.msdf2, msdf3)
        self.assertEqual(174, len(merged_msdf2.df))

        merged_msdf3 = merge_msdf(self.msdf1, self.msdf2)
        self.assertEqual(34, len(merged_msdf3.df))

    def test_merge_with_reconcile(self):
        """Test merging two tables with reconciliation."""
        merged_msdf = merge_msdf(self.msdf1, self.msdf2, reconcile=True)
        self.assertEqual(len(merged_msdf.df), 18)

    def test_merge_without_reconcile(self):
        """Test merging two tables without reconciliation."""
        merged_msdf = merge_msdf(self.msdf1, self.msdf2, reconcile=False)
        self.assertEqual(len(merged_msdf.df), 34)

    def test_merge_with_reconcile_without_confidence(self):
        """Test merging two tables without reconciliation."""
        msdf1 = parse_sssom_table(data_dir / "reconcile_1.tsv")
        msdf2 = parse_sssom_table(data_dir / "reconcile_2.tsv")

        merged_msdf = merge_msdf(msdf1, msdf2, reconcile=True)

        self.assertEqual(3, len(msdf1.df))
        self.assertEqual(4, len(msdf2.df))
        self.assertEqual(len(merged_msdf.df), (len(msdf1.df) + len(msdf2.df)))
