import os
import unittest

from sssom import (
    filter_redundant_rows,
)
from sssom.parsers import read_sssom_tsv
from sssom.util import merge_msdf, deal_with_negation

# from pandasql import sqldf

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, "data")


class TestReconcile(unittest.TestCase):
    def setUp(self) -> None:
        self.msdf = read_sssom_tsv(f"{data_dir}/basic3.tsv")

    def test_df(self):
        df = self.msdf.df
        print(df[0:20])
        self.assertTrue(True)

    def test_filter(self):
        df = filter_redundant_rows(self.msdf.df)
        print(df[0:20])
        assert len(df.index) == 10

    def test_deal_with_negation(self):
        df = deal_with_negation(self.msdf.df)
        assert len(df.index) == 7

    def test_merge(self):
        msdf1 = read_sssom_tsv(f"{data_dir}/basic.tsv")
        msdf2 = read_sssom_tsv(f"{data_dir}/basic2.tsv")

        merged_msdf = merge_msdf(msdf1=msdf1, msdf2=msdf2)

        assert len(merged_msdf.df) == 95

    def test_merge_no_reconcile(self):
        msdf1 = read_sssom_tsv(f"{data_dir}/basic.tsv")
        msdf2 = read_sssom_tsv(f"{data_dir}/basic4.tsv")

        merged_msdf = merge_msdf(msdf1=msdf1, msdf2=msdf2)
        print(f"LEN: {len(msdf1.df)}")
        print(f"LEN MERGED: {len(merged_msdf.df)}")
        assert len(msdf1.df) == 141
        assert len(msdf2.df) == 141
        assert len(merged_msdf.df) == (len(msdf1.df)+len(msdf2.df))
