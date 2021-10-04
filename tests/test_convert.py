"""Tests for conversion utilities."""

import unittest

from sssom.parsers import read_sssom_table
from sssom.writers import to_owl_graph, to_rdf_graph
from tests.constants import data_dir


class TestConvert(unittest.TestCase):
    """A test case for conversion utilities."""

    def setUp(self) -> None:
        """Set up the test case with two tables."""
        self.msdf = read_sssom_table(f"{data_dir}/basic.tsv")
        self.cob = read_sssom_table(f"{data_dir}/cob-to-external.tsv")

    def test_df(self):
        df = self.msdf.df
        self.assertEqual(
            len(df.index),
            141,
            "The tested SSSOM file has an unexpected number of mappings.",
        )

    def test_to_owl(self):
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
        g = to_owl_graph(self.cob)
        results = g.query(
            """SELECT DISTINCT ?e1 ?e2
                WHERE {
                  ?e1 <http://www.w3.org/2002/07/owl#equivalentClass> ?e2 .
                }"""
        )
        # g.serialize(destination="tmp/cob-external-test.owl", format="turtle")
        size = len(results)
        self.assertEqual(size, 61)

    def test_to_rdf(self):
        g = to_rdf_graph(self.msdf)
        results = g.query(
            """SELECT DISTINCT ?e1 ?e2
                WHERE {
                  ?e1 <http://www.w3.org/2002/07/owl#equivalentClass> ?e2 .
                }"""
        )
        size = len(results)
        self.assertEqual(size, 90)
