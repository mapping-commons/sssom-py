"""Tests for rewiring utilities."""

import os
import unittest
from typing import Any, Dict, Tuple

from rdflib import OWL, RDF, Graph, Namespace, URIRef

from sssom.constants import (
    PREDICATE_ID,
    PREDICATE_MODIFIER_NOT,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_EXACT_MATCH_URI,
    SKOS_NARROW_MATCH,
    SKOS_RELATED_MATCH,
)
from sssom.parsers import parse_sssom_table
from sssom.rdf_util import rewire_graph
from tests.constants import data_dir, test_out_dir

AIRO = Namespace("https://w3id.org/airo#")
ETICAS = Namespace("https://taxonomy.eticas.ai/")
IBM_RISK_ATLAS = Namespace("https://www.ibm.com/docs/en/watsonx/saas?topic=")
MIT_AI_RISK_REPOSITORY = Namespace("https://airisk.mit.edu/")
AIUC1 = Namespace("https://www.aiuc1.com/framework/")
OWASP_LLM = Namespace("https://owasp.org/www-project-top-10-for-large-language-model-applications/")

#: The three Eticas risks that the predicates test data maps.
PERFORMANCE_EQUITY = ETICAS["eticas-performance-equity"]
HARMFUL_CONTENT = ETICAS["eticas-harmful-content-toxicity"]
SENTIMENT_FAIRNESS = ETICAS["eticas-sentiment-fairness"]
RISKS = (PERFORMANCE_EQUITY, HARMFUL_CONTENT, SENTIMENT_FAIRNESS)

#: Some of the entities they are mapped to.
OUTPUT_BIAS = IBM_RISK_ATLAS["atlas-output-bias"]
TOXIC_OUTPUT = IBM_RISK_ATLAS["atlas-toxic-output"]
HARMFUL_OUTPUT = IBM_RISK_ATLAS["atlas-harmful-output"]
UNFAIR_DISCRIMINATION = MIT_AI_RISK_REPOSITORY["mit-ai-risk-subdomain-1.1"]
UNEQUAL_PERFORMANCE = MIT_AI_RISK_REPOSITORY["mit-ai-risk-subdomain-1.3"]


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


class TestRewirePredicates(unittest.TestCase):
    """Test case for choosing the mapping predicates that rewiring uses.

    The mapping set holds the curated mappings of three risks of the Eticas AI Risk Taxonomy,
    from IBM AI Atlas Nexus. Differential performance has an exact and a close candidate, harmful
    content has an exact and three close candidates, and sentiment disparity has a narrow one.
    """

    def setUp(self) -> None:
        """Load the Eticas mapping set and the part of the taxonomy it maps."""
        self.mset = parse_sssom_table(data_dir / "rewire-predicates.tsv")
        self.graph = self.load_graph()

    @staticmethod
    def load_graph() -> Graph:
        """Load the part of the Eticas taxonomy that the mapping set maps.

        :return: A new graph with the three risks and their groups.
        """
        return Graph().parse(data_dir / "rewire-predicates.ttl", format="turtle")

    def assert_rewired(self, graph: Graph, replacements: Dict[URIRef, URIRef]) -> None:
        """Assert that exactly the given risks were replaced, each by the given entity.

        :param graph: The graph after rewiring.
        :param replacements: The replacement of each risk that should have been rewired.
        """
        nodes = set(graph.all_nodes())
        for risk in RISKS:
            if risk in replacements:
                self.assertNotIn(risk, nodes, f"{risk} was not rewired")
                self.assertIn(replacements[risk], nodes, f"{risk} was not rewired")
            else:
                self.assertIn(risk, nodes, f"{risk} was rewired")

    def test_default_is_equivalence(self) -> None:
        """Test that nothing is rewired by default, because no mapping is an OWL equivalence."""
        with self.assertLogs(level="WARNING") as logs:
            n = rewire_graph(self.graph, self.mset)
        self.assertEqual(0, n)
        self.assert_rewired(self.graph, {})
        self.assertIn("No mapping to rewire by", logs.output[0])

    def test_strict(self) -> None:
        """Test that strict rewiring fails when no mapping has one of the chosen predicates."""
        with self.assertRaises(ValueError):
            rewire_graph(self.graph, self.mset, strict=True)
        self.assert_rewired(self.graph, {})

    def test_predicate_that_does_not_expand(self) -> None:
        """Test that a chosen predicate whose prefix is not in the prefix map fails loudly."""
        with self.assertRaisesRegex(ValueError, "Could not expand predicate skoss:exactMatch"):
            rewire_graph(self.graph, self.mset, predicates=["skoss:exactMatch"])
        self.assert_rewired(self.graph, {})

    def test_mapping_predicate_that_does_not_expand(self) -> None:
        """Test that a mapping whose predicate prefix is not in the prefix map fails loudly.

        The last mapping, of sentiment disparity, gets a predicate whose prefix is not in the
        prefix map. Rewiring by ``skos:exactMatch`` would not use that mapping, but it still fails.
        """
        self.mset.df.loc[self.mset.df.index[-1], PREDICATE_ID] = "foo:customMatch"
        with self.assertRaisesRegex(ValueError, "Could not expand predicate foo:customMatch"):
            rewire_graph(self.graph, self.mset, predicates=[SKOS_EXACT_MATCH])
        self.assert_rewired(self.graph, {})

    def test_exact(self) -> None:
        """Test rewiring by skos:exactMatch, given as a CURIE or as an IRI."""
        for predicate in (SKOS_EXACT_MATCH, SKOS_EXACT_MATCH_URI):
            with self.subTest(predicate=predicate):
                graph = self.load_graph()
                n = rewire_graph(graph, self.mset, predicates=[predicate])
                self.assertEqual(6, n)
                self.assert_rewired(
                    graph, {PERFORMANCE_EQUITY: OUTPUT_BIAS, HARMFUL_CONTENT: TOXIC_OUTPUT}
                )

    def test_single_predicate_as_string(self) -> None:
        """Test that a single predicate given as a string is not read character by character."""
        n = rewire_graph(self.graph, self.mset, predicates=SKOS_EXACT_MATCH)
        self.assertEqual(6, n)
        self.assert_rewired(
            self.graph, {PERFORMANCE_EQUITY: OUTPUT_BIAS, HARMFUL_CONTENT: TOXIC_OUTPUT}
        )

    def test_preferred_predicate_wins_over_precedence(self) -> None:
        """Test that an exact candidate, listed first, wins over close ones with precedence."""
        n = rewire_graph(
            self.graph,
            self.mset,
            predicates=[SKOS_EXACT_MATCH, SKOS_CLOSE_MATCH],
            precedence=["mit-ai-risk-repository"],
        )
        self.assertEqual(6, n)
        self.assert_rewired(
            self.graph, {PERFORMANCE_EQUITY: OUTPUT_BIAS, HARMFUL_CONTENT: TOXIC_OUTPUT}
        )

    def test_preferred_predicate_wins_whatever_the_row_order(self) -> None:
        """Test that close candidates listed before the one exact candidate make no tie."""
        n = rewire_graph(self.graph, self.mset, predicates=[SKOS_EXACT_MATCH, SKOS_CLOSE_MATCH])
        self.assertEqual(6, n)
        self.assert_rewired(
            self.graph, {PERFORMANCE_EQUITY: OUTPUT_BIAS, HARMFUL_CONTENT: TOXIC_OUTPUT}
        )

    def test_order_of_predicates_is_the_preference(self) -> None:
        """Test that listing skos:closeMatch before skos:exactMatch prefers the close candidates.

        The close candidates of harmful content then tie, and precedence decides between them.
        """
        with self.assertRaises(ValueError):
            rewire_graph(self.graph, self.mset, predicates=[SKOS_CLOSE_MATCH, SKOS_EXACT_MATCH])
        self.assert_rewired(self.graph, {})
        n = rewire_graph(
            self.graph,
            self.mset,
            predicates=[SKOS_CLOSE_MATCH, SKOS_EXACT_MATCH],
            precedence=["ibm-risk-atlas"],
        )
        self.assertEqual(6, n)
        self.assert_rewired(
            self.graph, {PERFORMANCE_EQUITY: UNEQUAL_PERFORMANCE, HARMFUL_CONTENT: HARMFUL_OUTPUT}
        )

    def test_precedence_decides_a_tie(self) -> None:
        """Test that precedence chooses between the three close candidates of harmful content."""
        with self.assertRaises(ValueError):
            rewire_graph(self.graph, self.mset, predicates=[SKOS_CLOSE_MATCH])
        self.assert_rewired(self.graph, {})
        n = rewire_graph(
            self.graph, self.mset, predicates=[SKOS_CLOSE_MATCH], precedence=["ibm-risk-atlas"]
        )
        self.assertEqual(6, n)
        self.assert_rewired(
            self.graph, {PERFORMANCE_EQUITY: UNEQUAL_PERFORMANCE, HARMFUL_CONTENT: HARMFUL_OUTPUT}
        )

    def test_agreeing_mappings_are_no_tie(self) -> None:
        """Test that two mappings to the same replacement under two predicates are not a tie."""
        mset = parse_sssom_table(data_dir / "rewire-agreeing.tsv")
        risk = IBM_RISK_ATLAS["atlas-lack-of-model-transparency"]
        requirement = AIUC1["aiuc1-req-e017"]
        graph = Graph()
        graph.add((risk, RDF.type, AIRO.Risk))
        n = rewire_graph(graph, mset, predicates=[SKOS_CLOSE_MATCH, SKOS_RELATED_MATCH])
        self.assertEqual(1, n)
        self.assertEqual({(requirement, RDF.type, AIRO.Risk)}, set(graph))

    def test_reverse(self) -> None:
        """Test applying exact and narrow mappings from object to subject."""
        graph = Graph()
        for entity in (OUTPUT_BIAS, TOXIC_OUTPUT, UNFAIR_DISCRIMINATION):
            graph.add((entity, RDF.type, AIRO.Risk))
        n = rewire_graph(
            graph,
            self.mset,
            subject_to_object=False,
            predicates=[SKOS_EXACT_MATCH, SKOS_NARROW_MATCH],
        )
        self.assertEqual(3, n)
        self.assertEqual({(risk, RDF.type, AIRO.Risk) for risk in RISKS}, set(graph))


class TestRewireNoMatch(unittest.TestCase):
    """Test case for mappings that record that there is no match.

    The mapping set holds the curated mappings of two IBM risks to the OWASP Top 10 for LLM
    applications, from IBM AI Atlas Nexus. The curators accepted one candidate for attribute
    inference and rejected another, and found no candidate for a lack of model transparency.
    """

    def setUp(self) -> None:
        """Load the OWASP mapping set and a graph with the IBM risk."""
        self.mset = parse_sssom_table(data_dir / "rewire-no-match.tsv")
        self.graph = Graph()
        self.graph.add((IBM_RISK_ATLAS["atlas-attribute-inference-attack"], RDF.type, AIRO.Risk))

    def test_negated_mapping_is_skipped(self) -> None:
        """Test that a rejected candidate is skipped, so it does not tie with the accepted one."""
        n = rewire_graph(self.graph, self.mset, predicates=[SKOS_RELATED_MATCH])
        self.assertEqual(1, n)
        disclosure = OWASP_LLM["llm022025-sensitive-information-disclosure"]
        self.assertEqual({(disclosure, RDF.type, AIRO.Risk)}, set(self.graph))

    def test_no_term_found_is_skipped(self) -> None:
        """Test that a risk mapped to sssom:NoTermFound is left as it is."""
        risk = IBM_RISK_ATLAS["atlas-lack-of-model-transparency"]
        self.graph.add((risk, RDF.type, AIRO.Risk))
        n = rewire_graph(self.graph, self.mset, predicates=[SKOS_RELATED_MATCH])
        self.assertEqual(1, n)
        self.assertIn((risk, RDF.type, AIRO.Risk), set(self.graph))
