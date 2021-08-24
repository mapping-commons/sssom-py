import os
import yaml

cwd = os.path.abspath(os.path.dirname(__file__))
test_data_dir = os.path.join(cwd, "data")
test_out_dir = os.path.join(cwd, "tmp")
test_validate_dir = os.path.join(cwd, "validate_data")
schema_dir = os.path.join(cwd, "../schema")
TEST_CONFIG = os.path.join(cwd, "test_config.yaml")
DEFAULT_CONTEXT_PATH = os.path.join(schema_dir, "sssom.context.jsonld")


def get_test_file(filename):
    return os.path.join(test_data_dir, filename)


def ensure_test_dir_exists():
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)


def load_config():
    with open(TEST_CONFIG) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def get_all_test_cases():
    test_cases = []
    config = load_config()
    for test in config["tests"]:
        test_cases.append(SSSOMTestCase(test, config["queries"]))
    return test_cases


def get_multiple_input_test_cases():
    test_cases = dict()
    config = load_config()
    for test in config["tests"]:
        if test["multiple_input"]:
            test = SSSOMTestCase(test, config["queries"])
            test_cases[test.id] = test
    return test_cases


class SSSOMTestCase:
    def __init__(self, config, queries):
        self.filepath = get_test_file(config["filename"])
        self.filename = config['filename']
        if "id" in config:
            self.id = config['id']
        else:
            self.id = config['filename']

        if "metadata_file" in config:
            self.metadata_file = config["metadata_file"]
        else:
            self.metadata_file = None
        self.graph_serialisation = "turtle"
        self.ct_json_elements = config["ct_json_elements"]
        self.ct_data_frame_rows = config["ct_data_frame_rows"]
        if "inputformat" in config:
            self.inputformat = config["inputformat"]
        else:
            self.inputformat = None
        self.ct_graph_queries_owl = self._query_tuple(
            config, "ct_graph_queries_owl", queries
        )
        self.ct_graph_queries_rdf = self._query_tuple(
            config, "ct_graph_queries_rdf", queries
        )
        if "curie_map" in config:
            self.curie_map = config["curie_map"]
        else:
            self.curie_map = None

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
