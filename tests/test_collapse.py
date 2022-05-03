"""Test various grouping functionalities."""

import unittest

import yaml
from pandasql import sqldf

from sssom import (
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    parse,
    reconcile_prefix_and_data,
)
from sssom.parsers import parse_sssom_table
from tests.constants import data_dir, prefix_recon_yaml


class TestCollapse(unittest.TestCase):
    """Test various grouping functionalities."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.df = parse(data_dir / "basic.tsv")

    def test_row_count(self):
        """Test the dataframe has the correct number of rows."""
        df = self.df
        self.assertEqual(
            len(df),
            141,
            f"Dataframe should have a different number of rows {df.head(10)}",
        )

    def test_collapse(self):
        """Test the row count after collapsing the dataframe."""
        df = collapse(self.df)
        self.assertEqual(
            len(df), 91, f"Dataframe should have a different {df.head(10)}"
        )

    def test_filter(self):
        """Test the row count after filtering redundant rows."""
        df = filter_redundant_rows(self.df)
        self.assertEqual(len(df), 91)

    def test_ptable(self):
        """Test the row count of the ptable export."""
        rows = dataframe_to_ptable(self.df)
        self.assertEqual(91, len(rows))

    def test_groupings(self):
        """Test the row count after grouping mappings."""
        mappings = group_mappings(self.df)
        self.assertEqual(len(mappings), 91)

    def test_diff(self):
        """Test the comparison between two dataframes."""
        diff = compare_dataframes(self.df, self.df)
        self.assertEqual(0, len(diff.unique_tuples1))
        self.assertEqual(0, len(diff.unique_tuples2))
        self.assertEqual(91, len(diff.common_tuples))
        diff_df = diff.combined_dataframe
        # print(len(diff_df.index))
        # print(diff_df[0:20])
        self.assertLess(100, len(diff_df.index))
        for c in diff_df["comment"]:
            self.assertTrue(c.startswith("COMMON_TO_BOTH"))
        output = sqldf("select * from diff_df where comment != ''")
        print(output)
        # print(diff)

        df2 = parse(data_dir / "basic2.tsv")
        diff = compare_dataframes(self.df, df2)
        # print(len(diff.unique_tuples1))
        # print(len(diff.unique_tuples2))
        # print(len(diff.common_tuples))
        self.assertEqual(15, len(diff.unique_tuples1))
        self.assertEqual(3, len(diff.unique_tuples2))
        self.assertEqual(76, len(diff.common_tuples))
        # totlen = len(diff.unique_tuples1) + len(diff.unique_tuples2) + len(diff.common_tuples)
        # self.assertEqual(totlen, len(self.df.index) + len(df2.index))
        diff_df = diff.combined_dataframe
        print(len(diff_df.index))
        # print(diff_df[0:10])

    def test_reconcile_prefix(self):
        """Test curie reconciliation is performing as expected."""
        msdf = parse_sssom_table(data_dir / "basic3.tsv")

        with open(prefix_recon_yaml) as pref_rec:
            prefix_reconciliation = yaml.safe_load(pref_rec)

        recon_msdf = reconcile_prefix_and_data(msdf, prefix_reconciliation)

        prefix_expansion = prefix_reconciliation["prefix_expansion_reconciliation"]

        for pfx, exp in prefix_expansion.items():
            if pfx in recon_msdf.prefix_map.keys():
                self.assertEqual(recon_msdf.prefix_map[pfx], exp)
