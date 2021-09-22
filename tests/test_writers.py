import os
import unittest

from sssom.parsers import read_sssom_json, read_sssom_rdf, read_sssom_table
from sssom.writers import write_json, write_owl, write_rdf, write_table
from tests.test_data import test_data_dir, test_out_dir


class TestWrite(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists(test_out_dir):
            os.mkdir(test_out_dir)
        self.msdf = read_sssom_table(f"{test_data_dir}/basic.tsv")
        # self.msdf = read_sssom_table(f"{test_data_dir}/basic-simple.tsv")
        self.mapping_count = 141  # 141 for basic.tsv

    def test_write_sssom_dataframe(self):
        tmp_path = os.path.join(test_out_dir, "test_write_sssom_dataframe.tsv")
        with open(tmp_path, "w") as tmp_file:
            write_table(self.msdf, tmp_file)
        msdf = read_sssom_table(tmp_path)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{tmp_file} has the wrong number of mappings.",
        )

    def test_write_sssom_rdf(self):
        path_1 = os.path.join(test_out_dir, "test_write_sssom_rdf.rdf")
        with open(path_1, "w") as file:
            write_rdf(self.msdf, file)
        msdf = read_sssom_rdf(path_1, self.msdf.prefix_map)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{path_1} has the wrong number of mappings.",
        )

        # TODO this test doesn't make sense
        path_2 = os.path.join(test_out_dir, "test_write_sssom_rdf.rdf.tsv")
        with open(path_2, "w") as file:
            write_table(self.msdf, file)

    def test_write_sssom_json(self):
        path = os.path.join(test_out_dir, "test_write_sssom_json.json")
        with open(path, "w") as file:
            write_json(self.msdf, file)
        msdf = read_sssom_json(path)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{path} has the wrong number of mappings.",
        )

    def test_write_sssom_owl(self):
        tmp_file = os.path.join(test_out_dir, "test_write_sssom_owl.owl")
        with open(tmp_file, "w") as file:
            write_owl(self.msdf, file)
        # FIXME this test doesn't test anything
        # TODO implement "read_owl" function
        self.assertEqual(1, 1)
