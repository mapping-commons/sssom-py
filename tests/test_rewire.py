"""Tests for rewiring utilities."""

import os
import unittest
from typing import Any, Tuple

from rdflib import OWL, RDF, Graph

from sssom.constants import PREDICATE_MODIFIER_NOT
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


class TestRewireBasic(unittest.TestCase):
    """Test case for rewiring by rows of the basic mapping set."""

    def setUp(self) -> None:
        """Load the basic mapping set."""
        self.msdf = parse_sssom_table(data_dir / "basic.tsv")

    def rewire(self, *subjects: str, **kwargs: Any) -> Tuple[Graph, int]:
        """Rewire a graph of the given classes by their own mappings in the basic mapping set.

        :param subjects: The CURIEs of the classes, and of the mapping subjects to keep.
        :param kwargs: Further arguments to :func:`rewire_graph`.
        :return: The rewired graph, and the number of triples that changed.
        """
        # Keep only the rows of these subjects, because other rows tie and raise an error.
        self.msdf.df = self.msdf.df[self.msdf.df["subject_id"].isin(subjects)]
        graph = Graph()
        for subject in subjects:
            graph.add((self.msdf.converter.expand_rdflib(subject), RDF.type, OWL.Class))
        return graph, rewire_graph(graph, self.msdf, **kwargs)

    def test_agreeing_mappings_are_no_tie(self) -> None:
        """Test that two mappings to the same replacement are not a tie."""
        graph, n = self.rewire("x:hindlimb")
        self.assertEqual(1, n)
        self.assertEqual({self.msdf.converter.expand_rdflib("z:hindlimb")}, set(graph.subjects()))

    def test_precedence_decides_a_tie(self) -> None:
        """Test that two different replacements are a tie, which precedence decides."""
        with self.assertRaises(ValueError):
            self.rewire("x:organ")
        graph, n = self.rewire("x:organ", precedence=["z"])
        self.assertEqual(1, n)
        self.assertEqual({self.msdf.converter.expand_rdflib("z:organ")}, set(graph.subjects()))

    def test_negated_mapping_is_skipped(self) -> None:
        """Test that a negated mapping is not applied."""
        graph, n = self.rewire("x:eyeball", "d:something")
        self.assertEqual(1, n)
        self.assertEqual(
            {
                self.msdf.converter.expand_rdflib("z:eyeball"),
                self.msdf.converter.expand_rdflib("d:something"),
            },
            set(graph.subjects()),
        )

    def test_negated_mapping_is_no_tie(self) -> None:
        """Test that a negated mapping does not tie with a positive mapping for the same node."""
        # When rewiring from object to subject, y:organ has two candidates: x:organ and z:organ.
        df = self.msdf.df[self.msdf.df["object_id"] == "y:organ"].copy()
        # Negate the z:organ row, so that x:organ is the only candidate left.
        df.loc[df["subject_id"] == "z:organ", "predicate_modifier"] = PREDICATE_MODIFIER_NOT
        self.msdf.df = df
        graph = Graph()
        graph.add((self.msdf.converter.expand_rdflib("y:organ"), RDF.type, OWL.Class))
        n = rewire_graph(graph, self.msdf, subject_to_object=False)
        self.assertEqual(1, n)
        self.assertEqual({self.msdf.converter.expand_rdflib("x:organ")}, set(graph.subjects()))
