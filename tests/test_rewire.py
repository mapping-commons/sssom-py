"""Tests for rewiring utilities."""

import os
import unittest
from typing import Collection

from rdflib import OWL, RDF, Graph, Namespace

from sssom.constants import SKOS_BROAD_MATCH, SKOS_BROAD_MATCH_URI
from sssom.parsers import parse_sssom_table
from sssom.rdf_util import REWIRE_FLAVORS, rewire_graph
from tests.constants import data_dir, test_out_dir

SOURCE = Namespace("https://example.org/source/")
TARGET = Namespace("https://example.org/target/")
ALTERNATIVE = Namespace("https://example.org/alternative/")

#: The local names that the source and target terms of the predicates test data share.
TERMS = ("Equivalent", "Exact", "Broad", "Narrow", "Close", "Unmapped")


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


class TestRewirePredicates(unittest.TestCase):
    """Test case for choosing the mapping predicates that rewiring uses.

    The mapping set has one mapping per predicate from a source term to its target term. It also
    has a second, agreeing mapping for the equivalent term, a weaker broad mapping for the exact
    term, and two competing close mappings for the close term.
    """

    def setUp(self) -> None:
        """Load the predicates mapping set and the small source graph."""
        self.mset = parse_sssom_table(data_dir / "rewire-predicates.tsv")
        self.graph = Graph()
        self.graph.parse(data_dir / "rewire-predicates.ttl", format="turtle")

    def assert_rewired(self, graph: Graph, rewired: Collection[str]) -> None:
        """Assert that exactly the given source terms were replaced by their target terms.

        :param graph: The graph after rewiring.
        :param rewired: Local names of the terms that should now be in the target namespace.
        """
        nodes = set(graph.all_nodes())
        for term in TERMS:
            if term in rewired:
                self.assertNotIn(SOURCE[term], nodes, f"{term} was not rewired")
                self.assertIn(TARGET[term], nodes, f"{term} was not rewired")
            else:
                self.assertIn(SOURCE[term], nodes, f"{term} was rewired")
                self.assertNotIn(TARGET[term], nodes, f"{term} was rewired")

    def test_default_is_equivalence(self) -> None:
        """Test that only the OWL equivalence mappings are used when no predicates are given."""
        n = rewire_graph(self.graph, self.mset)
        self.assertEqual(2, n)
        self.assert_rewired(self.graph, {"Equivalent"})

    def test_exact(self) -> None:
        """Test that the exact flavor adds skos:exactMatch and accepts an agreeing second mapping."""
        n = rewire_graph(self.graph, self.mset, predicates=REWIRE_FLAVORS["exact"])
        self.assertEqual(6, n)
        self.assert_rewired(self.graph, {"Equivalent", "Exact"})

    def test_exact_broad(self) -> None:
        """Test that the exact-broad flavor adds skos:broadMatch and chooses exact over broad."""
        n = rewire_graph(self.graph, self.mset, predicates=REWIRE_FLAVORS["exact-broad"])
        self.assertEqual(8, n)
        self.assert_rewired(self.graph, {"Equivalent", "Exact", "Broad"})
        self.assertNotIn(TARGET.ExactParent, set(self.graph.all_nodes()))

    def test_any(self) -> None:
        """Test that the any flavor uses every SKOS predicate and needs precedence for a tie."""
        with self.assertRaises(ValueError):
            rewire_graph(self.graph, self.mset, predicates=REWIRE_FLAVORS["any"])
        self.assert_rewired(self.graph, set())
        n = rewire_graph(
            self.graph, self.mset, predicates=REWIRE_FLAVORS["any"], precedence=["tgt"]
        )
        self.assertEqual(12, n)
        self.assert_rewired(self.graph, {"Equivalent", "Exact", "Broad", "Narrow", "Close"})

    def test_precedence_decides_between_equal_predicates(self) -> None:
        """Test that precedence picks between two skos:closeMatch replacements for one term."""
        rewire_graph(self.graph, self.mset, predicates={"skos:closeMatch"}, precedence=["alt"])
        nodes = set(self.graph.all_nodes())
        self.assertIn(ALTERNATIVE.Close, nodes)
        self.assertNotIn(TARGET.Close, nodes)
        self.assertNotIn(SOURCE.Close, nodes)

    def test_explicit_predicates(self) -> None:
        """Test that given predicates, spelled as a CURIE or an IRI, replace the default ones."""
        for predicate in (SKOS_BROAD_MATCH, SKOS_BROAD_MATCH_URI):
            graph = Graph()
            graph.parse(data_dir / "rewire-predicates.ttl", format="turtle")
            n = rewire_graph(graph, self.mset, predicates={predicate})
            self.assertEqual(6, n, predicate)
            nodes = set(graph.all_nodes())
            self.assertIn(TARGET.Broad, nodes, predicate)
            self.assertIn(TARGET.ExactParent, nodes, predicate)
            self.assertNotIn(TARGET.Exact, nodes, predicate)
            self.assertIn(SOURCE.Equivalent, nodes, predicate)

    def test_hierarchical_mappings_are_not_reversed(self) -> None:
        """Test that broad and narrow mappings are skipped when rewiring from object to subject."""
        graph = Graph()
        for term in TERMS:
            graph.add((TARGET[term], RDF.type, OWL.Class))
        n = rewire_graph(
            graph, self.mset, subject_to_object=False, predicates=REWIRE_FLAVORS["any"]
        )
        self.assertEqual(3, n)
        nodes = set(graph.all_nodes())
        for term in ("Equivalent", "Exact", "Close"):
            self.assertIn(SOURCE[term], nodes, f"{term} was not rewired")
            self.assertNotIn(TARGET[term], nodes, f"{term} was not rewired")
        for term in ("Broad", "Narrow", "Unmapped"):
            self.assertIn(TARGET[term], nodes, f"{term} was rewired")
            self.assertNotIn(SOURCE[term], nodes, f"{term} was rewired")
