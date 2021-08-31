import os
import unittest

from sssom.parsers import read_sssom_table
from sssom.writers import to_owl_graph, to_rdf_graph

# from pandasql import sqldf

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, "data")


class TestConvert(unittest.TestCase):
    def setUp(self) -> None:
        self.msdf = read_sssom_table(f"{data_dir}/basic.tsv")

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
