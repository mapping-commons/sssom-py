import os
import unittest

from pandasql import sqldf

from sssom import (
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    parse,
)

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, "data")


class TestCollapse(unittest.TestCase):
    def setUp(self) -> None:
        self.df = parse(f"{data_dir}/basic.tsv")

    def test_df(self):
        df = self.df
        self.assertEquals(
            len(df),
            141,
            f"Dataframe should have a different number of rows {df.head(10)}",
        )

    def test_collapse(self):
        df = collapse(self.df)
        self.assertEquals(
            len(df), 91, f"Dataframe should have a different {df.head(10)}"
        )

    def test_filter(self):
        df = filter_redundant_rows(self.df)
        self.assertEqual(len(df.index), 92)

    def test_ptable(self):
        rows = dataframe_to_ptable(self.df)
        for row in rows[0:10]:
            print("\t".join(row))
        self.assertEqual(91, len(rows))

    def test_groupings(self):
        mappings = group_mappings(self.df)
        self.assertEqual(len(mappings), 91)

    def test_diff(self):
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

        df2 = parse(f"{data_dir}/basic2.tsv")
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
