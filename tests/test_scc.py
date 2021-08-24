import os
import unittest

from sssom.cliques import split_into_cliques, summarize_cliques
from sssom.parsers import read_sssom_table

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, "data")


class TestSCC(unittest.TestCase):
    """
    Tests strongly connected components
    """

    def setUp(self) -> None:
        self.mset = read_sssom_table(f"{data_dir}/basic.tsv")

    def test_scc(self):
        cliquedocs = split_into_cliques(self.mset)
        for d in cliquedocs:
            print(f"D: {len(d.mapping_set.mappings)}")

    def test_cliquesummary(self):
        df = summarize_cliques(self.mset)
        df.to_csv(f"{data_dir}/basic-cliquesummary.tsv", sep="\t")
        df.describe().transpose().to_csv(
            f"{data_dir}/basic-cliquesummary-stats.tsv", sep="\t"
        )
