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

    def test_rewire(self) -> None:
        """Test running the require function."""
        with self.assertRaises(ValueError):
            # we expect this to fail due to PR/CHEBI ambiguity
            rewire_graph(self.graph, self.mset)

        n = rewire_graph(self.graph, self.mset, precedence=["PR"])
        self.assertLessEqual(0, n)
        with open(test_out_dir / "rewired-cob.ttl", "w") as stream:
            stream.write(self.graph.serialize(format="turtle"))


class TestRewireNoMapping(unittest.TestCase):
    """Test case for a mapping set with no mapping that rewiring uses.

    The MONDO to NANDO mapping set has only ``skos:exactMatch`` mappings.
    """

    def setUp(self) -> None:
        """Load the MONDO to NANDO mapping set."""
        self.msdf = parse_sssom_table(data_dir / "mondo-nando.sssom.tsv")

    def test_warning(self) -> None:
        """Test that rewiring warns when no mapping is left to rewire by."""
        with self.assertLogs(level="WARNING") as logs:
            n = rewire_graph(Graph(), self.msdf)
        self.assertEqual(0, n)
        self.assertIn("No mapping to rewire by", logs.output[0])

    def test_strict(self) -> None:
        """Test that strict rewiring fails when no mapping is left to rewire by."""
        with self.assertRaises(ValueError):
            rewire_graph(Graph(), self.msdf, strict=True)
