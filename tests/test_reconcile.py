from sssom.parsers import from_tsv
from sssom import parse, collapse, dataframe_to_ptable, filter_redundant_rows, group_mappings, compare_dataframes
from sssom.util import merge_msdf, deal_with_negation

import unittest
import os
#from pandasql import sqldf
import logging
cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, 'data')


class TestReconcile(unittest.TestCase):

    def setUp(self) -> None:
        self.msdf = from_tsv(f'{data_dir}/basic3.tsv')
        

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
        msdf1 = from_tsv(f'{data_dir}/basic.tsv')
        msdf2 = from_tsv(f'{data_dir}/basic2.tsv')
        
        merged_msdf = merge_msdf(msdf1=msdf1, msdf2=msdf2)

        assert len(merged_msdf.df) == 94
