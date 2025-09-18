"""Tests for conversion utilities."""

import unittest

import curies
import pandas as pd
import rdflib

from sssom import MappingSetDataFrame
from sssom.constants import (
    MAPPING_JUSTIFICATION,
    OBJECT_ID,
    PREDICATE_ID,
    PREDICATE_MODIFIER,
    SEMAPV,
    SUBJECT_ID,
)
from sssom.parsers import parse_sssom_table
from sssom.writers import to_json, to_owl_graph, to_rdf_graph
from tests.constants import data_dir


class TestConvert(unittest.TestCase):
    """A test case for conversion utilities."""

    def setUp(self) -> None:
        """Set up the test case with two tables."""
        self.msdf = parse_sssom_table(data_dir / "basic.tsv")
        self.cob = parse_sssom_table(data_dir / "cob-to-external.tsv")

    def test_df(self) -> None:
        """Test the dataframe has the right number of mappings."""
        df = self.msdf.df
        self.assertEqual(
            len(df.index),
            141,
            "The tested SSSOM file has an unexpected number of mappings.",
        )

    def test_to_owl(self) -> None:
        """Test converting the basic example to an OWL RDF graph."""
        g = to_owl_graph(self.msdf)
        results = g.query(
            """SELECT DISTINCT ?e1 ?e2
                WHERE {
                  ?e1 <http://www.w3.org/2002/07/owl#equivalentClass> ?e2 .
                }"""
        )
        size = len(results)
        self.assertEqual(size, 90)

    def test_cob_to_owl(self) -> None:
        """Test converting the COB example to an OWL RDF graph."""
        g = to_owl_graph(self.cob)
        results = g.query(
            """SELECT DISTINCT ?e1 ?e2
                WHERE {
                  ?e1 <http://www.w3.org/2002/07/owl#equivalentClass> ?e2 .
                }"""
        )
        # g.serialize(destination="tmp/cob-external-test.owl", format="turtle")
        size = len(results)
        self.assertEqual(size, 60)

        results = g.query(
            """SELECT DISTINCT ?e1 ?e2
                WHERE {
                  ?e1 <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?e2 .
                }"""
        )
        size = len(results)
        self.assertEqual(size, 22)

        results = g.query(
            """SELECT DISTINCT ?e1 ?e2
                WHERE {
                  ?e1 <https://w3id.org/sssom/superClassOf> ?e2 .
                }"""
        )
        size = len(results)
        self.assertEqual(size, 0)

    def test_to_rdf(self) -> None:
        """Test converting the basic example to a basic RDF graph."""
        g = to_rdf_graph(self.msdf)
        results = g.query(
            """SELECT DISTINCT ?e1 ?e2 WHERE {
                ?ax a owl:Axiom ;
                        owl:annotatedSource ?e1 ;
                        owl:annotatedProperty owl:equivalentClass ;
                        owl:annotatedTarget ?e2 .
                }"""
        )
        size = len(results)
        self.assertEqual(size, 90)

    def test_to_json(self) -> None:
        """Test converting the basic example to a JSON object."""
        json_obj = to_json(self.msdf)
        self.assertIsInstance(json_obj, dict)
        self.assertIsNotNone(json_obj["mapping_set_id"])
        self.assertIsNotNone(json_obj["license"])
        self.assertGreater(len(json_obj["mappings"]), 100)
        # TODO: add inject_type=False to arguments to dumper
        # self.assertNotIn("@type", json_obj)
        for m in json_obj["mappings"]:
            self.assertIn("subject_id", m)
            # ensure no JSON-LD strangeness
            for k in m.keys():
                self.assertFalse(k.startswith("@"))

    def test_to_rdf_hydrated(self) -> None:
        """Test converting to RDF with hydration."""
        rows = [
            (
                "DOID:0050601",
                "skos:exactMatch",
                "UMLS:C1863204",
                SEMAPV.ManualMappingCuration.value,
                "",
            ),
            (
                "mesh:C562684",
                "skos:exactMatch",
                "HP:0003348",
                SEMAPV.ManualMappingCuration.value,
                "NOT",
            ),
            (
                "mesh:C563052",
                "skos:exactMatch",
                "sssom:NoTermFound",
                SEMAPV.ManualMappingCuration.value,
                "",
            ),
            (
                "sssom:NoTermFound",
                "skos:exactMatch",
                "mesh:C562684",
                SEMAPV.ManualMappingCuration.value,
                "",
            ),
        ]
        columns = [
            SUBJECT_ID,
            PREDICATE_ID,
            OBJECT_ID,
            MAPPING_JUSTIFICATION,
            PREDICATE_MODIFIER,
        ]
        df = pd.DataFrame(rows, columns=columns)
        converter = curies.Converter.from_prefix_map(
            {
                "DOID": "http://purl.obolibrary.org/obo/DOID_",
                "HP": "http://purl.obolibrary.org/obo/HP_",
                "UMLS": "https://uts.nlm.nih.gov/uts/umls/concept/",
                "mesh": "http://id.nlm.nih.gov/mesh/",
                "sssom": "https://w3id.org/sssom/",
            }
        )
        msdf = MappingSetDataFrame(df, converter=converter)
        graph = to_rdf_graph(msdf, hydrate=False)
        self.assertIn("sssom", {p for p, _ in graph.namespaces()})
        self.assert_not_ask(
            graph,
            "ASK { DOID:0050601 skos:exactMatch UMLS:C1863204 }",
            msg="hydration should not have occurred",
        )
        self.assert_not_ask(graph, "ASK { mesh:C562684 skos:exactMatch HP:0003348 }")
        self.assert_not_ask(graph, "ASK { mesh:C563052 skos:exactMatch sssom:NoTermFound }")
        self.assert_not_ask(graph, "ASK { sssom:NoTermFound skos:exactMatch mesh:C564625 }")

        graph = to_rdf_graph(msdf, hydrate=True)
        self.assertIn("sssom", {p for p, _ in graph.namespaces()})
        self.assert_ask(
            graph,
            "ASK { DOID:0050601 skos:exactMatch UMLS:C1863204 }",
            msg="regular triple should be hydrated",
        )
        self.assert_not_ask(
            graph,
            "ASK { mesh:C562684 skos:exactMatch HP:0003348 }",
            msg="triple with NOT modifier should not be hydrated",
        )
        self.assert_not_ask(
            graph,
            "ASK { mesh:C563052 skos:exactMatch sssom:NoTermFound }",
            msg="triple with NoTermFound as object should not be hydrated",
        )
        self.assert_not_ask(
            graph,
            "ASK { sssom:NoTermFound skos:exactMatch mesh:C564625 }",
            msg="triple with NoTermFound as subject should not be hydrated",
        )

    def assert_ask(self, graph: rdflib.Graph, query: str, *, msg: str | None = None) -> None:
        """Assert that the query returns a true answer."""
        self.assertTrue(graph.query(query).askAnswer, msg=msg)

    def assert_not_ask(self, graph: rdflib.Graph, query: str, *, msg: str | None = None) -> None:
        """Assert that the query returns a false answer."""
        self.assertFalse(graph.query(query).askAnswer, msg=msg)
