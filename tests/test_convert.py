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

    def test_uberon_to_rdf(self) -> None:
        """Test to ensure that a mixed case prefix is not upper cased by LinkMLs rdflib_dumper."""
        from linkml_runtime.dumpers import rdflib_dumper
        from linkml_runtime.utils.schemaview import SchemaView

        from sssom.constants import SCHEMA_YAML
        from sssom.parsers import to_mapping_set_document

        # Load the SSSOM table and then ensure that the associated prefixmap has the correct mixed case prefix (it does)
        uberon = parse_sssom_table(data_dir / "uberon.sssom.tsv")
        self.assertNotIn(
            "HSAPDV", uberon.converter.bimap, msg="Namespace 'HSAPDV' should NOT exist in the MSDF"
        )
        self.assertIn(
            "HsapDv",
            uberon.converter.bimap,
            msg="Namespace 'HsapDv' should exist in the MSDF",
        )

        # Convert TSV file to YAML and ensure prefix is still there
        doc = to_mapping_set_document(uberon)
        self.assertIn(
            "HsapDv",
            doc.mapping_set.mappings[0].object_id,
            msg="Namespace 'HsapDv' should exist in the MSDF",
        )

        # Run LinkML conversion
        g = rdflib_dumper.as_rdf_graph(
            element=doc.mapping_set,
            schemaview=SchemaView(SCHEMA_YAML),
            prefix_map=uberon.converter.bimap,
        )

        namespaces = list(g.namespace_manager.namespaces())
        self.assertTrue(
            any(prefix == "HsapDv" for prefix, _ in namespaces),
            msg="Namespace 'HsapDv' should exist in the graph (and it does)",
        )

        # To show that the assert works, I am checking a random prefix to ensure it does not exist
        self.assertFalse(
            any(prefix == "RANDOMRR" for prefix, _ in namespaces),
            msg="Namespace 'RANDOM' should exist in the graph, and indeed, it does not",
        )

        # Here is where the problem happens:
        self.assertFalse(
            any(prefix == "HSAPDV" for prefix, _ in namespaces),
            msg="Namespace 'HSAPDV' should NOT exist in the graph, but I show it here to proof my point",
        )

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
