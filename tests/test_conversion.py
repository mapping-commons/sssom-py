"""Tests for conversion utilities."""

import filecmp
import json
import logging
import unittest

from rdflib import Graph

from sssom.parsers import get_parsing_function, to_mapping_set_document
from sssom.sssom_document import MappingSetDocument
from sssom.util import read_pandas, to_mapping_set_dataframe
from sssom.writers import (
    to_dataframe,
    to_json,
    to_owl_graph,
    to_rdf_graph,
    write_json,
    write_owl,
    write_rdf,
    write_table,
)

from .test_data import SSSOMTestCase, get_all_test_cases


class SSSOMReadWriteTestSuite(unittest.TestCase):
    """A test case for conversion utilities."""

    def test_conversion(self):
        """Run all conversion tests."""
        test_cases = get_all_test_cases()
        self.assertTrue(len(test_cases) > 2, "Less than 2 testcases in the test suite!")
        for test in test_cases:
            with self.subTest(test=test.id):
                read_func = get_parsing_function(test.inputformat, test.filepath)
                msdf = read_func(test.filepath, prefix_map=test.prefix_map)
                mdoc = to_mapping_set_document(msdf)
                logging.info(f"Testing {test.filepath}")
                self.assertEqual(
                    len(mdoc.mapping_set.mappings),
                    test.ct_data_frame_rows,
                    f"Wrong number of mappings in MappingSet of {test.filename}",
                )
                logging.info("Testing OWL export")
                self._test_to_owl_graph(mdoc, test)
                logging.info("Testing RDF export")
                self._test_to_rdf_graph(mdoc, test)
                logging.info("Testing CSV export")
                self._test_to_dataframe(mdoc, test)
                logging.info("Testing JSON export")
                self._test_to_json_dict(mdoc, test)
                self._test_to_json(mdoc, test)

    def _test_to_owl_graph(self, mdoc, test):
        msdf = to_mapping_set_dataframe(mdoc)
        g = to_owl_graph(msdf)
        file_format = "owl"
        self._test_graph_roundtrip(g, test, file_format)
        path = test.get_out_file(file_format)
        with open(path, "w") as file:
            write_owl(msdf, file, test.graph_serialisation)
        self._test_load_graph_size(
            path,
            test.graph_serialisation,
            test.ct_graph_queries_owl,
        )
        # self._test_files_equal(test.get_out_file(file_format), test.get_validate_file(file_format))

    def _test_to_json(self, mdoc, test: SSSOMTestCase):
        msdf = to_mapping_set_dataframe(mdoc)
        jsonob = to_json(msdf)
        self.assertEqual(len(jsonob), test.ct_json_elements)
        with open(test.get_out_file("json"), "w") as file:
            write_json(msdf, file, serialisation="json")

    def _test_to_rdf_graph(self, mdoc, test):
        msdf = to_mapping_set_dataframe(mdoc)
        g = to_rdf_graph(msdf)
        file_format = "rdf"
        self._test_graph_roundtrip(g, test, file_format)
        path = test.get_out_file(file_format)
        with open(path, "w") as file:
            write_rdf(msdf, file, test.graph_serialisation)
        self._test_load_graph_size(
            path,
            test.graph_serialisation,
            test.ct_graph_queries_rdf,
        )
        # self._test_files_equal(test.get_out_file(file_format), test.get_validate_file(file_format))

    def _test_graph_roundtrip(self, g: Graph, test: SSSOMTestCase, file_format: str):
        self._test_graph_size(
            g, getattr(test, f"ct_graph_queries_{file_format}"), test.filename
        )
        f_roundtrip = test.get_out_file(f"roundtrip.{file_format}")
        g.serialize(destination=f_roundtrip, format=test.graph_serialisation)
        self._test_load_graph_size(
            f_roundtrip,
            test.graph_serialisation,
            getattr(test, f"ct_graph_queries_{file_format}"),
        )

    def _test_files_equal(self, f1, f2):
        self.assertTrue(filecmp.cmp(f1, f2), f"{f1} and {f2} are not the same!")

    def _test_load_graph_size(self, file: str, graph_serialisation: str, queries: list):
        g = Graph()
        g.parse(file, format=graph_serialisation)
        self._test_graph_size(g, queries, file)

    def _test_graph_size(self, graph: Graph, queries: list, file: str):
        for query, size in queries:
            self.assertEqual(
                len(graph.query(query)),
                size,
                f"Graph query {query} does not return the expected number of triples for {file}",
            )

    def _test_to_dataframe(self, mdoc, test):
        msdf = to_mapping_set_dataframe(mdoc)
        df = to_dataframe(msdf)
        self.assertEqual(
            len(df),
            test.ct_data_frame_rows,
            f"The pandas data frame has less elements than the orginal one for {test.filename}",
        )
        df.to_csv(test.get_out_file("roundtrip.tsv"), sep="\t")
        # data = pd.read_csv(test.get_out_file("roundtrip.tsv"), sep="\t")
        data = read_pandas(test.get_out_file("roundtrip.tsv"))
        self.assertEqual(
            len(data),
            test.ct_data_frame_rows,
            f"The re-serialised pandas data frame has less elements than the orginal one for {test.filename}",
        )
        path = test.get_out_file("tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        # self._test_files_equal(test.get_out_file("tsv"), test.get_validate_file("tsv"))
        df = read_pandas(path)
        self.assertEqual(
            len(df),
            test.ct_data_frame_rows,
            f"The exported pandas data frame has less elements than the orginal one for {test.filename}",
        )

    def _test_to_json_dict(self, mdoc: MappingSetDocument, test: SSSOMTestCase):
        msdf = to_mapping_set_dataframe(mdoc)
        json_dict = to_json(msdf)
        self.assertTrue("mappings" in json_dict)

        self.assertEqual(
            len(json_dict),
            test.ct_json_elements,
            f"JSON document has less elements than the orginal one for {test.filename}. Json: {json.dumps(json_dict)}",
        )

        self.assertEqual(
            len(json_dict["mappings"]),
            len(msdf.df),
            f"JSON document has less mappings than the orginal ({test.filename}). Json: {json.dumps(json_dict)}",
        )

        with open(test.get_out_file("roundtrip.json"), "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)

        with open(test.get_out_file("roundtrip.json")) as json_file:
            data = json.load(json_file)

        self.assertEqual(
            len(data),
            test.ct_json_elements,
            f"The re-serialised JSON file has less elements than the orginal one for {test.filename}",
        )
        path = test.get_out_file("jsonld")
        with open(path, "w") as file:
            write_json(msdf, file)
        with open(path) as json_file:
            data = json.load(json_file)
        # self._test_files_equal(test.get_out_file("json"), test.get_validate_file("json"))
        self.assertEqual(
            len(data),
            test.ct_json_elements,
            f"The exported JSON file has less elements than the orginal one for {test.filename}",
        )
