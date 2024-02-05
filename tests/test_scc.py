"""Test case for splitting strongly connected components."""

import unittest

from sssom.cliques import split_into_cliques, summarize_cliques
from sssom.parsers import parse_sssom_table
from tests.constants import data_dir


class TestSCC(unittest.TestCase):
    """Test case for splitting strongly connected components."""

    def setUp(self) -> None:
        """Set up the test case by reading the basic SSSOM example."""
        self.mset = parse_sssom_table(data_dir / "basic.tsv")

    def test_scc(self):
        """Test splitting into cliques."""
        cliquedocs = split_into_cliques(self.mset)
        expected = [38, 36, 5, 8, 8, 10, 14, 8, 8, 2, 4]
        self.assertEqual(sorted(expected), sorted(len(d.mapping_set.mappings) for d in cliquedocs))

    def test_cliquesummary(self):
        """Test summarizing cliques."""
        df = summarize_cliques(self.mset)
        df.to_csv(data_dir / "basic-cliquesummary.tsv", sep="\t")
        df.describe().transpose().to_csv(data_dir / "basic-cliquesummary-stats.tsv", sep="\t")
