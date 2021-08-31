import os
import unittest

from rdflib import Graph

from sssom.parsers import read_sssom_table
from sssom.rdf_util import rewire_graph

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, "data")
test_out_dir = os.path.join(cwd, "tmp")


class TestRewire(unittest.TestCase):
    """
    Tests strongly connected components
    """

    def setUp(self) -> None:
        self.mset = read_sssom_table(f"{data_dir}/cob-to-external.tsv")
        g = Graph()
        g.parse(f"{data_dir}/cob.owl", format="xml")
        self.graph = g

    def test_rewire(self):
        expected_exception = False
        try:
            rewire_graph(self.graph, self.mset)
        except Exception:
            # we expect this to fail due to PR/CHEBI ambiguity
            expected_exception = True
        assert expected_exception
        n = rewire_graph(self.graph, self.mset, precedence=["PR"])
        print(f"Num changed = {n}")
        with open(f"{test_out_dir}/rewired-cob.ttl", "w") as stream:
            stream.write(self.graph.serialize(format="turtle").decode())
