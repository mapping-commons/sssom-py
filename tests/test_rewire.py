"""Tests for rewiring utilities."""

import os
import unittest

from rdflib import Graph

from sssom.parsers import parse_sssom_table
from sssom.rdf_util import rewire_graph
from tests.constants import data_dir, test_out_dir


class TestRewire(unittest.TestCase):
    """Test case for rewiring utilities."""

    def setUp(self) -> None:
        """Set up the test case with the COB mappings et and OWL graph."""
        self.mset = parse_sssom_table(data_dir / "cob-to-external.tsv")
        g = Graph()
        g.parse(os.path.join(data_dir, "cob.owl"), format="xml")
        self.graph = g

    def test_rewire(self):
        """Test running the require function."""
        with self.assertRaises(ValueError):
            # we expect this to fail due to PR/CHEBI ambiguity
            rewire_graph(self.graph, self.mset)

        n = rewire_graph(self.graph, self.mset, precedence=["PR"])
        self.assertLessEqual(0, n)
        with open(test_out_dir / "rewired-cob.ttl", "w") as stream:
            stream.write(self.graph.serialize(format="turtle"))
