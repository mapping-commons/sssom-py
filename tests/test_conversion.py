import json
import logging
import filecmp
import unittest
import pandas as pd

from rdflib import Graph
from sssom.sssom_document import MappingSetDocument
from sssom.parsers import read_pandas, get_parsing_function
from sssom.writers import to_owl_graph, to_rdf_graph, to_dataframe, to_jsonld_dict
from sssom.writers import write_json, write_rdf, write_owl, write_tsv
from .test_data import ensure_test_dir_exists, SSSOMTestCase, get_all_test_cases


class SSSOMReadWriteTestSuite(unittest.TestCase):
    def test(self):
        ensure_test_dir_exists()
        test_cases = get_all_test_cases()
        self.assertTrue(len(test_cases) > 2, "Less than 2 testcases in the test suite!")
        for test in test_cases:
            with self.subTest():
                read_func = get_parsing_function(test.inputformat, test.filepath)
                mdoc = read_func(test.filepath, curie_map=test.curie_map)
                logging.info(f"Testing {test.filepath}")
                self.assertEqual(len(mdoc.mapping_set.mappings), test.ct_data_frame_rows,
                                 f"Wrong number of mappings in MappingSet of {test.filename}")
                logging.info("Testing OWL export")
                self._test_to_owl_graph(mdoc, test)
                logging.info("Testing RDF export")
                self._test_to_rdf_graph(mdoc, test)
                logging.info("Testing CSV export")
                self._test_to_dataframe(mdoc, test)
                logging.info("Testing JSON export")
                self._test_to_json_dict(mdoc, test)

    def _test_to_owl_graph(self, mdoc, test):
        g = to_owl_graph(mdoc)
        file_format = "owl"
        self._test_graph_roundtrip(g, test, file_format)
        write_owl(mdoc, test.get_out_file(file_format), test.graph_serialisation)
        self._test_load_graph_size(test.get_out_file(file_format), test.graph_serialisation,
                                   getattr(test, f"ct_graph_queries_owl"))
        # self._test_files_equal(test.get_out_file(file_format), test.get_validate_file(file_format))

    def _test_to_rdf_graph(self, mdoc, test):
        g = to_rdf_graph(mdoc)
        file_format = "rdf"
        self._test_graph_roundtrip(g, test, file_format)
        write_rdf(mdoc, test.get_out_file(file_format), test.graph_serialisation)
        self._test_load_graph_size(test.get_out_file(file_format), test.graph_serialisation,
                                   getattr(test, f"ct_graph_queries_rdf"))
        # self._test_files_equal(test.get_out_file(file_format), test.get_validate_file(file_format))

    def _test_graph_roundtrip(self, g: Graph, test: SSSOMTestCase, file_format: str):
        self._test_graph_size(g, getattr(test, f"ct_graph_queries_{file_format}"), test.filename)
        f_roundtrip = test.get_out_file(f"roundtrip.{file_format}")
        g.serialize(destination=f_roundtrip, format=test.graph_serialisation)
        self._test_load_graph_size(f_roundtrip, test.graph_serialisation,
                                   getattr(test, f"ct_graph_queries_{file_format}"))

    def _test_files_equal(self, f1, f2):
        self.assertTrue(filecmp.cmp(f1, f2), f"{f1} and {f2} are not the same!")

    def _test_load_graph_size(self, file: str, graph_serialisation: str, queries: list):
        g = Graph()
        g.parse(file, format=graph_serialisation)
        self._test_graph_size(g, queries, file)

    def _test_graph_size(self, graph: Graph, queries: list, file: str):
        for query, size in queries:
            self.assertEqual(len(graph.query(query)), size,
                             f"Graph query {query} does not return the expected number of triples for {file}")

    def _test_to_dataframe(self, mdoc, test):
        df = to_dataframe(mdoc)
        self.assertEqual(len(df), test.ct_data_frame_rows,
                         f"The pandas data frame has less elements than the orginal one for {test.filename}")
        df.to_csv(test.get_out_file("roundtrip.tsv"), sep="\t")
        data = pd.read_csv(test.get_out_file("roundtrip.tsv"), sep="\t")
        self.assertEqual(len(data), test.ct_data_frame_rows,
                         f"The re-serialised pandas data frame has less elements than the orginal one for {test.filename}")
        write_tsv(mdoc, test.get_out_file("tsv"))
        # self._test_files_equal(test.get_out_file("tsv"), test.get_validate_file("tsv"))
        df = read_pandas(test.get_out_file("tsv"))
        self.assertEqual(len(df), test.ct_data_frame_rows,
                         f"The exported pandas data frame has less elements than the orginal one for {test.filename}")

    def _test_to_json_dict(self, mdoc: MappingSetDocument, test: SSSOMTestCase):
        json_dict = to_jsonld_dict(mdoc)
        self.assertEqual(len(json_dict), test.ct_json_elements,
                         f"JSON document has less elements than the orginal one for {test.filename}")

        with open(test.get_out_file("roundtrip.json"), 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)

        with open(test.get_out_file("roundtrip.json")) as json_file:
            data = json.load(json_file)

        self.assertEqual(len(data), test.ct_json_elements,
                         f"The re-serialised JSON file has less elements than the orginal one for {test.filename}")
        write_json(mdoc, test.get_out_file("json"))
        with open(test.get_out_file("json")) as json_file:
            data = json.load(json_file)
        # self._test_files_equal(test.get_out_file("json"), test.get_validate_file("json"))
        self.assertEqual(len(data), test.ct_json_elements,
                         f"The exported JSON file has less elements than the orginal one for {test.filename}")


if __name__ == '__main__':
    unittest.main()
