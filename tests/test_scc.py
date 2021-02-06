from sssom import parse, collapse, export_ptable
from sssom.parsers import from_tsv
from sssom.cliques import split_into_cliques

import unittest
import os

import logging
cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, 'data')

class TestSCC(unittest.TestCase):

    def setUp(self) -> None:
        self.mset = from_tsv(f'{data_dir}/basic.tsv')

    def test_scc(self):
        cliquedocs = split_into_cliques(self.mset)
        for d in cliquedocs:
            print(f'D: {len(d.mapping_set.mappings)}')
