"""Test case for splitting strongly connected components."""

import unittest

from sssom.cliques import split_into_cliques, summarize_cliques
from sssom.parsers import read_sssom_table
from tests.constants import data_dir


class TestSCC(unittest.TestCase):
    """Test case for splitting strongly connected components."""

    def setUp(self) -> None:
        """Set up the test case by reading the basic SSSOM example."""
        self.mset = read_sssom_table(f"{data_dir}/basic.tsv")

    def test_scc(self):
        """Test splitting into cliques."""
        cliquedocs = split_into_cliques(self.mset)
        for d in cliquedocs:
            print(f"D: {len(d.mapping_set.mappings)}")

    def test_cliquesummary(self):
        """Test summarizing cliques."""
        df = summarize_cliques(self.mset)
        df.to_csv(f"{data_dir}/basic-cliquesummary.tsv", sep="\t")
        df.describe().transpose().to_csv(
            f"{data_dir}/basic-cliquesummary-stats.tsv", sep="\t"
        )
