import os
import unittest

from pandasql import sqldf

from sssom import (
    parse,
    collapse,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    compare_dataframes,
)

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, "data")


class TestCollapse(unittest.TestCase):
    def setUp(self) -> None:
        self.df = parse(f"{data_dir}/basic.tsv")

    def test_df(self):
        df = self.df
        print(df[0:20])
        self.assertTrue(True)

    def test_collapse(self):
        df = collapse(self.df)
        print(df[0:20])

    def test_filter(self):
        df = filter_redundant_rows(self.df)
        print(df[0:20])
        self.assertEqual(len(df.index), 141)

    def test_ptable(self):
        rows = dataframe_to_ptable(self.df)
        for row in rows[0:10]:
            print("\t".join(row))
        assert len(rows) == 91

    def test_groupings(self):
        mappings = group_mappings(self.df)
        self.assertEqual(len(mappings), 3)

    def test_diff(self):
        diff = compare_dataframes(self.df, self.df)
        assert len(diff.unique_tuples1) == 0
        assert len(diff.unique_tuples2) == 0
        assert len(diff.common_tuples) == 91
        diff_df = diff.combined_dataframe
        print(len(diff_df.index))
        print(diff_df[0:20])
        assert len(diff_df.index) > 100
        for c in diff_df["comment"]:
            assert c.startswith("COMMON_TO_BOTH")
        output = sqldf("select * from diff_df where comment != ''")
        print(output)
        print(diff)

        df2 = parse(f"{data_dir}/basic2.tsv")
        diff = compare_dataframes(self.df, df2)
        print(len(diff.unique_tuples1))
        print(len(diff.unique_tuples2))
        print(len(diff.common_tuples))
        assert len(diff.unique_tuples1) == 15
        assert len(diff.unique_tuples2) == 3
        assert len(diff.common_tuples) == 76
        # totlen = len(diff.unique_tuples1) + len(diff.unique_tuples2) + len(diff.common_tuples)
        # assert totlen == len(self.df.index) + len(df2.index)
        diff_df = diff.combined_dataframe
        print(len(diff_df.index))
        print(diff_df[0:10])
