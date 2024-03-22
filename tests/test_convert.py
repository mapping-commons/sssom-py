"""Tests for conversion utilities."""

import unittest

from sssom.parsers import parse_sssom_table
from sssom.writers import to_json, to_owl_graph, to_rdf_graph
from tests.constants import data_dir


class TestConvert(unittest.TestCase):
    """A test case for conversion utilities."""

    def setUp(self) -> None:
        """Set up the test case with two tables."""
        self.msdf = parse_sssom_table(data_dir / "basic.tsv")
        self.cob = parse_sssom_table(data_dir / "cob-to-external.tsv")

    def test_df(self):
        """Test the dataframe has the right number of mappings."""
        df = self.msdf.df
        self.assertEqual(
            len(df.index),
            141,
            "The tested SSSOM file has an unexpected number of mappings.",
        )

    def test_to_owl(self):
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

    def test_cob_to_owl(self):
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

    def test_to_rdf(self):
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

    def test_to_json(self):
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
