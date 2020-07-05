from sssom import parse, collapse, export_ptable

import unittest

import logging

class TestCollapse(unittest.TestCase):

    def setUp(self) -> None:
        self.df = parse('data/basic.tsv')

    def test_df(self):
        df = self.df
        print(df[0:20])
        self.assertTrue(True)
        
    def test_collapse(self):
        df = collapse(self.df)
        print(df[0:20])

    def test_ptable(self):
        export_ptable(self.df)
