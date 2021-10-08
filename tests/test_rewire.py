"""Tests for rewiring utilities."""

import unittest

from rdflib import Graph

from sssom.parsers import read_sssom_table
from sssom.rdf_util import rewire_graph
from tests.constants import data_dir, test_out_dir


class TestRewire(unittest.TestCase):
    """Test case for rewiring utilities."""

    def setUp(self) -> None:
        """Set up the test case with the COB mappings et and OWL graph."""
        self.mset = read_sssom_table(f"{data_dir}/cob-to-external.tsv")
        g = Graph()
        g.parse(f"{data_dir}/cob.owl", format="xml")
        self.graph = g

    def test_rewire(self):
        """Test running the require function."""
        with self.assertRaises(ValueError):
            # we expect this to fail due to PR/CHEBI ambiguity
            rewire_graph(self.graph, self.mset)

        n = rewire_graph(self.graph, self.mset, precedence=["PR"])
        print(f"Num changed = {n}")
        with open(f"{test_out_dir}/rewired-cob.ttl", "w") as stream:
            stream.write(self.graph.serialize(format="turtle").decode())
