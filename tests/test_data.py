"""Tests for loading and processing data."""

import os
from typing import Any, List, Mapping

import yaml

from sssom.constants import CURIE_MAP
from tests.constants import cwd, data_dir, test_out_dir

test_validate_dir = os.path.join(cwd, "validate_data")
schema_dir = os.path.join(cwd, os.pardir, "schema")
DEFAULT_CONTEXT_PATH = os.path.join(schema_dir, "sssom.context.jsonld")
TEST_CONFIG_PATH = os.path.join(cwd, "test_config.yaml")
with open(TEST_CONFIG_PATH) as file:
    TEST_CONFIG = yaml.safe_load(file)
RECON_YAML = os.path.join(cwd, "data", "prefix_reconciliation.yaml")


def get_test_file(filename: str) -> str:
    """Get a test file path inside the test data directory."""
    return os.path.join(data_dir, filename)


class SSSOMTestCase:
    """A dynamic test case for data tests."""

    def __init__(self, config: Mapping[str, Any], queries: Mapping[str, str]):
        """Initialize the SSSOM test case.

        :param config: A dictionary of configuration values
        :param queries: A mapping from query name to SPARQL query strings
        """
        self.filepath = get_test_file(config["filename"])
        self.filename = config["filename"]
        self.id = config.get("id", self.filename)
        self.metadata_file = config.get("metadata_file")
        self.graph_serialisation = "turtle"
        self.ct_json_elements = config["ct_json_elements"]
        self.ct_data_frame_rows = config["ct_data_frame_rows"]
        self.inputformat = config.get("inputformat")
        self.ct_graph_queries_owl = self._query_tuple(config, "ct_graph_queries_owl", queries)
        self.ct_graph_queries_rdf = self._query_tuple(config, "ct_graph_queries_rdf", queries)
        self.prefix_map = config.get(CURIE_MAP)

    @staticmethod
    def _query_tuple(config, tuple_id, queries_dict):
        queries = []
        for t in config[tuple_id]:
            query = queries_dict[t]
            queries.append((query, config[tuple_id][t]))
        return queries

    def get_out_file(self, extension: str) -> str:
        """Get the output file path."""
        return os.path.join(test_out_dir, f"{self.filename}.{extension}")

    def get_validate_file(self, extension: str) -> str:
        """Get the validation file path."""
        return os.path.join(test_validate_dir, f"{self.filename}.{extension}")

    def __str__(self) -> str:  # noqa:D105
        return f"Testcase {self.id} (Filepath: {self.filepath})"


def get_all_test_cases() -> List[SSSOMTestCase]:
    """Get a list of all test cases."""
    test_cases = []
    for test in TEST_CONFIG["tests"]:
        test_cases.append(SSSOMTestCase(test, TEST_CONFIG["queries"]))
    return test_cases


def get_multiple_input_test_cases() -> Mapping[str, SSSOMTestCase]:
    """Get a mapping from identifiers to test cases that require multiple parameters."""
    test_cases = dict()
    for test in TEST_CONFIG["tests"]:
        if test["multiple_input"]:
            test = SSSOMTestCase(test, TEST_CONFIG["queries"])
            test_cases[test.id] = test
    return test_cases
