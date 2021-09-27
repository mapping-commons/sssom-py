import os
from typing import Any, Mapping

import yaml

from sssom.util import PREFIX_MAP_KEY

cwd = os.path.abspath(os.path.dirname(__file__))
test_data_dir = os.path.join(cwd, "data")
test_out_dir = os.path.join(cwd, "tmp")
os.makedirs(test_out_dir, exist_ok=True)
test_validate_dir = os.path.join(cwd, "validate_data")
schema_dir = os.path.join(cwd, "../schema")
TEST_CONFIG = os.path.join(cwd, "test_config.yaml")
DEFAULT_CONTEXT_PATH = os.path.join(schema_dir, "sssom.context.jsonld")


def get_test_file(filename: str) -> str:
    """Get a test file path inside the test data directory."""
    return os.path.join(test_data_dir, filename)


def load_config():
    """Load configuration.

    :return: Confiuration
    :rtype: Any
    """
    with open(TEST_CONFIG) as file:
        config = yaml.safe_load(file)
    return config


def get_all_test_cases():
    """Get all test cases.

    :return: List of test cases
    :rtype: List[SSOMTestCase]
    """
    test_cases = []
    config = load_config()
    for test in config["tests"]:
        test_cases.append(SSSOMTestCase(test, config["queries"]))
    return test_cases


def get_multiple_input_test_cases():
    """Get multiple input test cases.

    :return: List of test cases
    :rtype: List[SSOMTestCase]
    """
    test_cases = dict()
    config = load_config()
    for test in config["tests"]:
        if test["multiple_input"]:
            test = SSSOMTestCase(test, config["queries"])
            test_cases[test.id] = test
    return test_cases


class SSSOMTestCase:
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
        self.ct_graph_queries_owl = self._query_tuple(
            config, "ct_graph_queries_owl", queries
        )
        self.ct_graph_queries_rdf = self._query_tuple(
            config, "ct_graph_queries_rdf", queries
        )
        self.prefix_map = config.get(PREFIX_MAP_KEY)

    def _query_tuple(self, config, tuple_id, queries_dict):
        queries = []
        for t in config[tuple_id]:
            query = queries_dict[t]
            queries.append((query, config[tuple_id][t]))
        return queries

    def get_out_file(self, extension):
        return os.path.join(test_out_dir, f"{self.filename}.{extension}")

    def get_validate_file(self, extension):
        return os.path.join(test_validate_dir, f"{self.filename}.{extension}")

    def __str__(self) -> str:  # noqa:D105
        return f"Testcase {self.id} (Filepath: {self.filepath})"
